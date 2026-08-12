import numpy as np
import pandas as pd

from aeromaintain.config import SENSOR_COLUMNS, WINDOWS
from aeromaintain.features.temporal import add_temporal_features
from aeromaintain.inference.predictor import (
    get_latest_engine_cycles,
    predict_from_dataframe,
)


class FakeModel:
    """
    Faux modèle utilisé uniquement pour tester le code d'inférence.

    Il possède une méthode predict(), comme un modèle scikit-learn,
    mais renvoie toujours la même valeur.
    """

    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(
            len(X),
            self.value,
            dtype=float,
        )


def create_engines():
    """
    Crée deux moteurs artificiels contenant les mêmes colonnes brutes
    que les fichiers C-MAPSS.

    Moteur 1 : cycles 1 à 3
    Moteur 2 : cycles 1 à 4
    """

    rows = []

    engine_cycles = {
        1: 3,
        2: 4,
    }

    for engine_id, max_cycle in engine_cycles.items():
        for cycle in range(1, max_cycle + 1):
            row = {
                "FD": "001",
                "IdMoteur": engine_id,
                "Cycle": cycle,
                "ParameterOpe1": 0.0,
                "ParameterOpe2": 0.0,
                "ParameterOpe3": 100.0,
            }

            for sensor_index, sensor in enumerate(
                SENSOR_COLUMNS,
                start=1,
            ):
                row[sensor] = (
                    100.0
                    + sensor_index
                    + engine_id
                    + cycle
                )

            rows.append(row)

    return pd.DataFrame(rows)


def test_latest_cycle():
    engines = create_engines()

    latest = get_latest_engine_cycles(
        engines
    )

    assert len(latest) == 2

    engine_1 = latest[
        latest["IdMoteur"] == 1
    ].iloc[0]

    engine_2 = latest[
        latest["IdMoteur"] == 2
    ].iloc[0]

    assert engine_1["Cycle"] == 3
    assert engine_2["Cycle"] == 4


def test_prediction():
    engines = create_engines()

    # predict_from_dataframe() attend un DataFrame contenant déjà
    # les features utilisées par les modèles.
    engines = add_temporal_features(
        engines,
        windows=WINDOWS,
    )

    models = {
        "rul": FakeModel(30),
        "q10": FakeModel(20),
        "q50": FakeModel(28),
        "q90": FakeModel(40),
    }

    result = predict_from_dataframe(
        engines,
        models,
    )

    assert len(result) == 2

    assert result["RUL_predite"].tolist() == [
        30.0,
        30.0,
    ]

    assert result["q10"].tolist() == [
        20.0,
        20.0,
    ]

    assert result["q50"].tolist() == [
        28.0,
        28.0,
    ]

    assert result["q90"].tolist() == [
        40.0,
        40.0,
    ]

    result_by_engine = result.set_index(
        "IdMoteur"
    )

    assert result_by_engine.loc[1, "Cycle"] == 3
    assert result_by_engine.loc[2, "Cycle"] == 4