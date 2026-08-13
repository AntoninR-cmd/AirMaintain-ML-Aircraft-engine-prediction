import pandas as pd
from aeromaintain.config import PREDICTIONS_FOLDER
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


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

    return {
        "erreur moyenne": erreur.mean(),
        "erreur mediane": erreur.median(),
        "taux_surestimation": (erreur > 0).mean(),
        "taux_sous-estimation": (erreur < 0).mean()
    }
