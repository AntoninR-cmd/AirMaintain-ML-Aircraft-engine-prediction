import pandas as pd
from time import perf_counter
import joblib
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from aeromaintain.config import MODELS_FOLDER, PREDICTIONS_FOLDER


def train_val(name, model, X_train, y_train, X_val, y_val):
    debut = perf_counter()

    model.fit(X_train, y_train)

    temps_entrainement = perf_counter() - debut

    debut_prediction = perf_counter()
    y_pred = model.predict(X_val)
    temps_prediction = perf_counter() - debut_prediction

    save_model(name, model)

    # Sauvegarder les prédictions
    predictions = pd.DataFrame({
        "y_true": y_val.to_numpy(),
        "y_pred": y_pred
    })

    save_predictions(name, predictions)

    return {
        "Modèle": name,
        "RMSE": root_mean_squared_error(y_val, y_pred),
        "MAE": mean_absolute_error(y_val, y_pred),
        "R2": r2_score(y_val, y_pred),
        "Temps entraînement": temps_entrainement,
        "Temps prédiction": temps_prediction
    }


def save_model(name, model):
    joblib.dump(
        model,
        MODELS_FOLDER / f"{name}_validation.joblib",
        compress=3
    )


def save_predictions(name, predictions):
    predictions.to_csv(
        PREDICTIONS_FOLDER / f"{name}_validation.csv",
        index=False
    )