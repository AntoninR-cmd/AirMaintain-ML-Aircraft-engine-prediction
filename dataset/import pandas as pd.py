import pandas as pd
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
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from time import perf_counter


FEATURE_COLUMNS = [
    "Cycle",
    "ParameterOpe1",
    "ParameterOpe2",
    "ParameterOpe3",
    *[f"MesureCapteur{i:02d}" for i in range(1, 22)]
]

COLUMNS = ["IdMoteur", *FEATURE_COLUMNS]

DATASET_FOLDER = Path(
    r"C:\Users\Antonin\Mon Drive\AirMaintain"
    r"\AirMaintain-ML-Aircraft-engine-prediction\dataset"
)


def import_train():

    folder = DATASET_FOLDER / "train"

    dataframes = []
    for file in folder.iterdir():
        fd = file.stem[-3:]

        train_file = pd.read_csv(file, sep=r"\s+", header=None, names=COLUMNS)

        train_file['RUL'] = (
            train_file.groupby('IdMoteur')['Cycle'].transform('max') - train_file['Cycle']
        )
        train_file['FD'] = fd

        dataframes.append(train_file)

    if not dataframes:
        raise FileNotFoundError(
            f"Aucun fichier d'entraînement trouvé dans {folder}"
        )

    train = pd.concat(dataframes, ignore_index=True)

    X_train = train[FEATURE_COLUMNS]
    y_train = train["RUL"]

    return X_train, y_train


def import_X_test_val():
    folder = DATASET_FOLDER / "X_test"
    
    dataframe = []
    for file in folder.iterdir():
        test_file = pd.read_csv(file, sep=r"\s+", header=None, names=COLUMNS)

        test_file['FD'] = file.stem[-3:]

        dataframe.append(test_file)

    if not dataframe:
        raise FileNotFoundError(
            f"Aucun fichier trouvé dans {folder}"
        )

    test = pd.concat(dataframe, ignore_index=True)

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
    moteurs = test[["IdMoteur", "FD"]].drop_duplicates()

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

    feature_columns = [
        "Cycle",
        "ParameterOpe1",
        "ParameterOpe2",
        "ParameterOpe3",
        *[f"MesureCapteur{i:02d}" for i in range(1, 22)]
    ]

    X_test = test_final[feature_columns]
    y_test = test_final["RUL"]
    X_val = validation[feature_columns]
    y_val = validation["RUL"]

    cles_test = set(
        map(tuple, test_final[["FD", "IdMoteur"]].drop_duplicates().to_numpy())
    )

    cles_validation = set(
        map(tuple, validation[["FD", "IdMoteur"]].drop_duplicates().to_numpy())
    )

    assert cles_test.isdisjoint(cles_validation)

    return X_test, y_test, X_val, y_val


def train_val(name, model, X_train, y_train, X_val, y_val):
    debut = perf_counter()

    model.fit(X_train, y_train)

    temps_entrainement = perf_counter() - debut

    debut_prediction = perf_counter()
    y_pred = model.predict(X_val)
    temps_prediction = perf_counter() - debut_prediction

    return {
        "Modèle": name,
        "RMSE": root_mean_squared_error(y_val, y_pred),
        "MAE": mean_absolute_error(y_val, y_pred),
        "Temps entraînement": temps_entrainement,
        "Temps prédiction": temps_prediction
    }
    


def main():   
    test = import_X_test_val()
    test = import_y_test_val(test)

    X_test, y_test, X_val, y_val = split(test)

    X_train, y_train = import_train()

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
                cache_size=1000
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

    print(resultats.to_string(index=False))

main()