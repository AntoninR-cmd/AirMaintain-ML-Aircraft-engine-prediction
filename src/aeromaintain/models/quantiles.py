from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
from sklearn.metrics import mean_pinball_loss
import pandas as pd


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
