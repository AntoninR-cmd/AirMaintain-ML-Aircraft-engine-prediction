import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_pinball_loss
from time import perf_counter
import joblib
from sklearn.dummy import DummyRegressor


SENSOR_COLUMNS = [
    f"MesureCapteur{i:02d}"
    for i in range(1, 22)
]

BASE_FEATURE_COLUMNS = [
    "Cycle",
    "ParameterOpe1",
    "ParameterOpe2",
    "ParameterOpe3",
    *SENSOR_COLUMNS
]

COLUMNS = [
    "IdMoteur",
    *BASE_FEATURE_COLUMNS
]

WINDOWS = (5, 20)

DATASET_FOLDER = Path(
    r"C:\Users\Antonin\Mon Drive\AirMaintain"
    r"\AirMaintain-ML-Aircraft-engine-prediction\dataset"
)

RESULTS_FOLDER = DATASET_FOLDER.parent / "results"
MODELS_FOLDER = RESULTS_FOLDER / "models"
PREDICTIONS_FOLDER = RESULTS_FOLDER / "predictions"

RESULTS_FOLDER.mkdir(exist_ok=True)
MODELS_FOLDER.mkdir(exist_ok=True)
PREDICTIONS_FOLDER.mkdir(exist_ok=True)


def get_temporal_feature_columns(windows):
    temporal_columns = []

    long_window = max(windows)

    for sensor in SENSOR_COLUMNS:
        temporal_columns.append(
            f"{sensor}_delta1"
        )

        for window in windows:
            temporal_columns.append(
                f"{sensor}_mean_{window}"
            )
            temporal_columns.append(
                f"{sensor}_std_{window}"
            )

        temporal_columns.append(
            f"{sensor}_trend_{long_window}"
        )

    return temporal_columns


TEMPORAL_FEATURE_COLUMNS = get_temporal_feature_columns(
    WINDOWS
)

MODEL_FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    + TEMPORAL_FEATURE_COLUMNS
)

def add_temporal_features(df, windows=(5, 20)):
    """
    Crée des caractéristiques temporelles en utilisant uniquement
    le cycle actuel et les cycles précédents du même moteur.
    """

    df = df.sort_values(
        ["FD", "IdMoteur", "Cycle"]
    ).copy()

    grouped = df.groupby(
        ["FD", "IdMoteur"],
        sort=False
    )

    new_features = {}

    for sensor in SENSOR_COLUMNS:
        # Variation depuis le cycle précédent
        new_features[f"{sensor}_delta1"] = (
            grouped[sensor]
            .diff()
            .fillna(0)
        )

        for window in windows:
            # Moyenne mobile
            new_features[f"{sensor}_mean_{window}"] = (
                grouped[sensor]
                .transform(
                    lambda serie: serie.rolling(
                        window=window,
                        min_periods=1
                    ).mean()
                )
            )

            # Écart-type mobile
            new_features[f"{sensor}_std_{window}"] = (
                grouped[sensor]
                .transform(
                    lambda serie: serie.rolling(
                        window=window,
                        min_periods=2
                    ).std(ddof=0)
                )
                .fillna(0)
            )

        # Tendance approximative sur la plus grande fenêtre
        long_window = max(windows)

        new_features[f"{sensor}_trend_{long_window}"] = (
            grouped[sensor]
            .diff(long_window - 1)
            .div(long_window - 1)
            .fillna(0)
        )

    temporal_df = pd.DataFrame(
        new_features,
        index=df.index
    )

    return pd.concat(
        [df, temporal_df],
        axis=1
    )


def import_train():
    folder = DATASET_FOLDER / "train"
    dataframes = []

    for file in sorted(folder.glob("train_FD*.txt")):
        fd = file.stem[-3:]

        train_file = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            names=COLUMNS
        )

        train_file["FD"] = fd
        dataframes.append(train_file)

    if not dataframes:
        raise FileNotFoundError(
            f"Aucun fichier d'entraînement trouvé dans {folder}"
        )

    train = pd.concat(
        dataframes,
        ignore_index=True
    )

    train["RUL"] = (
        train.groupby(
            ["FD", "IdMoteur"]
        )["Cycle"].transform("max")
        - train["Cycle"]
    )

    train = add_temporal_features(
        train,
        windows=WINDOWS
    )

    X_train = train[
        MODEL_FEATURE_COLUMNS
    ].copy()

    y_train = train["RUL"].copy()

    return X_train, y_train


def import_X_test_val():
    folder = DATASET_FOLDER / "X_test"
    dataframes = []

    for file in sorted(folder.glob("test_FD*.txt")):
        test_file = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            names=COLUMNS
        )

        test_file["FD"] = file.stem[-3:]
        dataframes.append(test_file)

    if not dataframes:
        raise FileNotFoundError(
            f"Aucun fichier trouvé dans {folder}"
        )

    test = pd.concat(
        dataframes,
        ignore_index=True
    )

    test = add_temporal_features(
        test,
        windows=WINDOWS
    )

    return test


def import_y_test_val(test):
    folder = DATASET_FOLDER / "y_test"
        
    dataframes = []
    for file in folder.iterdir():
        fd = file.stem[-3:]

        RUL_file = pd.read_csv(file, sep=r"\s+", header=None, names=["RUL"])

        RUL_file['FD'] = fd
        RUL_file["IdMoteur"] = range(1, len(RUL_file)+1)

        dataframes.append(RUL_file)

    if not dataframes:
        raise FileNotFoundError(
            f"Aucun fichier trouvé dans {folder}"
        )

    y_test = pd.concat(dataframes, ignore_index=True)
    
    test = test.merge(
        y_test[["FD", "IdMoteur", "RUL"]], 
        on=["FD", "IdMoteur"],
        how="left",
        validate="many_to_one"
    )

    if test["RUL"].isna().any():
        lignes_manquantes = test.loc[
            test["RUL"].isna(),
            ["FD", "IdMoteur"]
        ].drop_duplicates()

        raise ValueError(
            f"RUL introuvable pour certains moteurs :\n{lignes_manquantes}"
        )

    test = test.rename(columns={"RUL": "RUL_fin"})
    test["Cycle_max"] = test.groupby(["IdMoteur", "FD"])["Cycle"].transform("max")
    test["RUL"] = (
        test["RUL_fin"] + (test["Cycle_max"] - test["Cycle"])
    )

    return test

def split(test):
    moteurs = test[
        ["FD", "IdMoteur"]
    ].drop_duplicates()

    moteurs_test, moteurs_validation = train_test_split(
        moteurs,
        test_size=0.5,
        random_state=42,
        stratify=moteurs["FD"]
    )

    test_final = test.merge(
        moteurs_test,
        on=["FD", "IdMoteur"],
        how="inner",
        validate="many_to_one"
    )

    validation = test.merge(
        moteurs_validation,
        on=["FD", "IdMoteur"],
        how="inner",
        validate="many_to_one"
    )

    cles_test = set(
        map(
            tuple,
            test_final[
                ["FD", "IdMoteur"]
            ].drop_duplicates().to_numpy()
        )
    )

    cles_validation = set(
        map(
            tuple,
            validation[
                ["FD", "IdMoteur"]
            ].drop_duplicates().to_numpy()
        )
    )

    assert cles_test.isdisjoint(
        cles_validation
    )

    X_test = test_final[
        MODEL_FEATURE_COLUMNS
    ].copy()

    y_test = test_final["RUL"].copy()

    X_val = validation[
        MODEL_FEATURE_COLUMNS
    ].copy()

    y_val = validation["RUL"].copy()

    return X_test, y_test, X_val, y_val


def train_val(name, model, X_train, y_train, X_val, y_val):
    debut = perf_counter()

    model.fit(X_train, y_train)

    temps_entrainement = perf_counter() - debut

    debut_prediction = perf_counter()
    y_pred = model.predict(X_val)
    temps_prediction = perf_counter() - debut_prediction

    # Sauvegarder le modèle entraîné
    joblib.dump(
        model,
        MODELS_FOLDER / f"{name}_validation.joblib",
        compress=3
    )

    # Sauvegarder les prédictions
    predictions = pd.DataFrame({
        "y_true": y_val.to_numpy(),
        "y_pred": y_pred
    })

    predictions.to_csv(
        PREDICTIONS_FOLDER / f"{name}_validation.csv",
        index=False
    )

    return {
        "Modèle": name,
        "RMSE": root_mean_squared_error(y_val, y_pred),
        "MAE": mean_absolute_error(y_val, y_pred),
        "R2": r2_score(y_val, y_pred),
        "Temps entraînement": temps_entrainement,
        "Temps prédiction": temps_prediction
    }
    

def evaluate_test(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    predictions = pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "y_pred": y_pred
    })

    predictions.to_csv(
        PREDICTIONS_FOLDER / f"{name}_test.csv",
        index=False
    )

    return {
        "Modèle": name,
        "RMSE": root_mean_squared_error(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }


def analyse_bias(name):
    predictions = pd.read_csv(
        PREDICTIONS_FOLDER / f"{name}_test.csv"
    )

    erreur = predictions["y_pred"] - predictions["y_true"]

    print("\n=== Analyse des erreurs ===")
    print(f"Erreur moyenne : {erreur.mean():.2f} cycles")
    print(f"Erreur médiane : {erreur.median():.2f} cycles")

    print(f"Surestimations : {(erreur > 0).mean() * 100:.1f} %")
    print(f"Sous-estimations : {(erreur < 0).mean()* 100:.1f} %")

    if (erreur > 0).any():
        print(f"Surestimation moyenne : {erreur[erreur > 0].mean():.2f} cycles")
    if (erreur < 0).any():
        print(f"Sous-estimation moyenne : {-erreur[erreur < 0].mean():.2f} cycles")


def create_quantile_models():
    common_params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,
        "random_state":42
    }

    models = {
        "q10": GradientBoostingRegressor(
            loss="quantile",
            alpha=0.10,
            ** common_params
        ),
        "q50": GradientBoostingRegressor(
            loss="quantile",
            alpha=0.50,
            ** common_params
        ),
        "q90": GradientBoostingRegressor(
            loss="quantile",
            alpha=0.90,
            ** common_params
        )
    }

    return models


def predict_quantile(quantile_models, X):
    q10 = quantile_models["q10"].predict(X)
    q50 = quantile_models["q50"].predict(X)
    q90 = quantile_models["q90"].predict(X)

    return q10, q50, q90


def evaluate_quantiles(
    y_true,
    q10,
    q50,
    q90
):
    y_true = np.asarray(y_true)
    q10 = np.asarray(q10)
    q50 = np.asarray(q50)
    q90 = np.asarray(q90)

    # Vérification : les quantiles doivent être ordonnés.
    crossing_rate = np.mean(
        (q10 > q50)
        | (q50 > q90)
    )

    # La vraie RUL appartient-elle à [q10 ; q90] ?
    covered = (
        (y_true >= q10)
        & (y_true <= q90)
    )

    coverage = covered.mean()

    # Largeur de l'intervalle
    widths = q90 - q10
    mean_width = widths.mean()

    # Pinball loss de chaque quantile
    pinball_q10 = mean_pinball_loss(
        y_true,
        q10,
        alpha=0.10
    )

    pinball_q50 = mean_pinball_loss(
        y_true,
        q50,
        alpha=0.50
    )

    pinball_q90 = mean_pinball_loss(
        y_true,
        q90,
        alpha=0.90
    )

    # Calibration :
    # q10 devrait avoir environ 10 % des vraies RUL en dessous.
    calibration_q10 = np.mean(
        y_true <= q10
    )

    # q50 : environ 50 %
    calibration_q50 = np.mean(
        y_true <= q50
    )

    # q90 : environ 90 %
    calibration_q90 = np.mean(
        y_true <= q90
    )

    # Biais de la prédiction médiane
    signed_error = q50 - y_true

    mean_signed_error = signed_error.mean()

    overestimation_rate = np.mean(
        signed_error > 0
    )

    underestimation_rate = np.mean(
        signed_error < 0
    )

    return {
        "Couverture attendue": 0.80,
        "Couverture réelle": coverage,
        "Largeur moyenne": mean_width,

        "Pinball q10": pinball_q10,
        "Pinball q50": pinball_q50,
        "Pinball q90": pinball_q90,

        "Calibration q10": calibration_q10,
        "Calibration q50": calibration_q50,
        "Calibration q90": calibration_q90,

        "Erreur moyenne signée q50": mean_signed_error,
        "Taux surestimation q50": overestimation_rate,
        "Taux sous-estimation q50": underestimation_rate,

        "Taux croisement quantiles": crossing_rate
    }


def print_quantile_results(title, results):
    print(f"\n=== {title} ===")

    print(
        f"Couverture q10-q90 : "
        f"{results['Couverture réelle'] * 100:.2f} % "
        f"(cible : {results['Couverture attendue'] * 100:.0f} %)"
    )

    print(
        f"Largeur moyenne : "
        f"{results['Largeur moyenne']:.2f} cycles"
    )

    print(
        f"Pinball q10 : "
        f"{results['Pinball q10']:.3f}"
    )

    print(
        f"Pinball q50 : "
        f"{results['Pinball q50']:.3f}"
    )

    print(
        f"Pinball q90 : "
        f"{results['Pinball q90']:.3f}"
    )

    print("\nCalibration :")

    print(
        f"q10 : {results['Calibration q10'] * 100:.2f} % "
        f"(cible : 10 %)"
    )

    print(
        f"q50 : {results['Calibration q50'] * 100:.2f} % "
        f"(cible : 50 %)"
    )

    print(
        f"q90 : {results['Calibration q90'] * 100:.2f} % "
        f"(cible : 90 %)"
    )

    print("\nBiais de q50 :")

    print(
        f"Erreur moyenne signée : "
        f"{results['Erreur moyenne signée q50']:.2f} cycles"
    )

    print(
        f"Surestimation : "
        f"{results['Taux surestimation q50'] * 100:.2f} %"
    )

    print(
        f"Sous-estimation : "
        f"{results['Taux sous-estimation q50'] * 100:.2f} %"
    )

    print(
        f"Croisement des quantiles : "
        f"{results['Taux croisement quantiles'] * 100:.2f} %"
    )


def calibration_by_wear(
    y_true,
    q10,
    q50,
    q90
):
    df = pd.DataFrame({
        "RUL": np.asarray(y_true),
        "q10": np.asarray(q10),
        "q50": np.asarray(q50),
        "q90": np.asarray(q90)
    })

    df["Niveau_usure"] = pd.cut(
        df["RUL"],
        bins=[
            -np.inf,
            10,
            20,
            30,
            60,
            np.inf
        ],
        labels=[
            "RUL <= 10",
            "10 < RUL <= 20",
            "20 < RUL <= 30",
            "30 < RUL <= 60",
            "RUL > 60"
        ]
    )

    df["Couvert"] = (
        (df["RUL"] >= df["q10"])
        & (df["RUL"] <= df["q90"])
    )

    df["Largeur"] = (
        df["q90"]
        - df["q10"]
    )

    df["Erreur_q50"] = (
        df["q50"]
        - df["RUL"]
    )

    df["Sous_q10"] = (
        df["RUL"] <= df["q10"]
    )

    df["Sous_q50"] = (
        df["RUL"] <= df["q50"]
    )

    df["Sous_q90"] = (
        df["RUL"] <= df["q90"]
    )

    resultats = (
        df.groupby(
            "Niveau_usure",
            observed=True
        )
        .agg(
            Nombre=("RUL", "size"),
            Couverture=("Couvert", "mean"),
            Largeur_moyenne=("Largeur", "mean"),
            Calibration_q10=("Sous_q10", "mean"),
            Calibration_q50=("Sous_q50", "mean"),
            Calibration_q90=("Sous_q90", "mean"),
            Biais_q50=("Erreur_q50", "mean")
        )
    )

    return resultats


def main():
    test = import_X_test_val()
    test = import_y_test_val(test)

    X_test, y_test, X_val, y_val = split(test)

    X_train, y_train = import_train()

    print("Dimensions :")
    print("X_train :", X_train.shape)
    print("X_val   :", X_val.shape)
    print("X_test  :", X_test.shape)

    assert list(X_train.columns) == list(X_val.columns)
    assert list(X_train.columns) == list(X_test.columns)

    models = {
        "LinearRegression": LinearRegression(),

        "Ridge": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1.0)
        ),

        "Lasso": make_pipeline(
            StandardScaler(),
            Lasso(
                alpha=0.01,
                max_iter=10_000,
                selection="random",
                random_state=42
            )
        ),

        "DecisionTree": DecisionTreeRegressor(
            max_depth=15,
            min_samples_leaf=5,
            random_state=42
        ),

        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42
        ),

        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),

        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            early_stopping=True,
            random_state=42
        ),

        "SVR": make_pipeline(
            StandardScaler(),
            SVR(
                kernel="rbf",
                C=10,
                epsilon=5,
                gamma="scale",
                cache_size=4000
            )
        ),

        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(
                n_neighbors=10,
                weights="distance",
                n_jobs=-1
            )
        ),

        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=42
            )
        )
    }

    resultats = []
    for name, model in models.items():
        print(f"Entraînement de {name}...")

        resultat = train_val(
            name,
            model,
            X_train,
            y_train,
            X_val,
            y_val
        )

        resultats.append(resultat)

    resultats = pd.DataFrame(resultats)
    resultats = resultats.sort_values("RMSE")

    resultats.to_csv(
        RESULTS_FOLDER / "resultats_validation.csv",
        index=False
    )

    print(resultats.to_string(index=False))

    best_name = resultats.iloc[0]["Modèle"]
    best_model = models[best_name]

    X_train_final = pd.concat(
        [X_train, X_val],
        ignore_index=True
    )

    y_train_final = pd.concat(
        [y_train, y_val],
        ignore_index=True
    )

    best_model.fit(
        X_train_final,
        y_train_final
    )

    #Sauvegarde du meilleur modèle
    joblib.dump(
        best_model,
        MODELS_FOLDER / "best_model_final.joblib",
        compress=3
    )

    with open(
        RESULTS_FOLDER / "best_model_name.txt",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(best_name)

    resultat_test = evaluate_test(
        best_name,
        best_model,
        X_test,
        y_test
    )

    print("\nRésultat final sur TEST :")
    print(resultat_test)

    dummy = DummyRegressor(
        strategy="mean"
    )

    dummy.fit(
        X_train_final,
        y_train_final
    )

    y_pred_dummy = dummy.predict(
        X_test
    )

    print(
        "Dummy RMSE :",
        root_mean_squared_error(
            y_test,
            y_pred_dummy
        )
    )

    print(
        "Dummy MAE :",
        mean_absolute_error(
            y_test,
            y_pred_dummy
        )
    )

    print(
        "Dummy R2 :",
        r2_score(
            y_test,
            y_pred_dummy
        )
    )

    analyse_bias(best_name)

    quantile_models = create_quantile_models()

    for name, model in quantile_models.items():
        model.fit(X_train_final, y_train_final)

        joblib.dump(
            model,
            MODELS_FOLDER / f"{name}_quantile.joblib",
            compress=3
        )

    q10, q50, q90 = predict_quantile(quantile_models, X_test)

    quantile_results = evaluate_quantiles(
        y_test,
        q10,
        q50,
        q90
    )

    print_quantile_results(
        "RESULTATS QUANTILES SUR TEST",
        quantile_results
    )

    pd.DataFrame(
        [quantile_results]
    ).to_csv(
        RESULTS_FOLDER / "resultats_quantiles_test.csv",
        index=False
    )

    quantile_predictions = pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "q10": q10,
        "q50": q50,
        "q90": q90
    })

    quantile_predictions.to_csv(
        PREDICTIONS_FOLDER / "quantiles_test.csv",
        index=False
    )

    wear_results = calibration_by_wear(
        y_test,
        q10,
        q50,
        q90
    )

    print(
        "\n=== CALIBRATION PAR NIVEAU D'USURE ==="
    )

    print(
        wear_results.to_string()
    )

main()