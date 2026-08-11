import pandas as pd
import numpy as np
from aeromaintain.data.loaders import load_train_data, load_X_test_data, load_y_test_data
from aeromaintain.data.split import split_validation_test_by_engine
from aeromaintain.models.registry import create_models
from aeromaintain.models.training import train_val
from aeromaintain.models.evaluation import evaluate_test, analyse_bias
from aeromaintain.models.quantiles import create_quantile_models,  predict_quantile, evaluate_quantiles, calibration_by_wear
from aeromaintain.config import RESULTS_FOLDER, MODELS_FOLDER, PREDICTIONS_FOLDER
import joblib
from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


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


def main():
    test = load_X_test_data()
    test = load_y_test_data(test)

    X_test, y_test, X_val, y_val = split_validation_test_by_engine(test)

    X_train, y_train = load_train_data()

    print("Dimensions :")
    print("X_train :", X_train.shape)
    print("X_val   :", X_val.shape)
    print("X_test  :", X_test.shape)

    assert list(X_train.columns) == list(X_val.columns)
    assert list(X_train.columns) == list(X_test.columns)

    models = create_models()

    resultats = []
    for name, model in models.items():
        print(f"Training of {name}...")

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

    print("\nFinal results on TEST :")
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

    print(analyse_bias(best_name))

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

if __name__ == "__main__":
    main()