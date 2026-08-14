from fastapi.testclient import TestClient

from aeromaintain.api.main import app
from aeromaintain.config import WINDOWS
from aeromaintain.features.temporal import MODEL_FEATURE_COLUMNS
import numpy as np

client = TestClient(app)

def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


class FakeModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(
            len(X),
            self.value,
            dtype=float
        )


def create_reading(cycle):
    reading = {
        "cycle": cycle,
        "parameter_ope1": 0.0,
        "parameter_ope2": 0.0,
        "parameter_ope3": 100.0,
    }

    for i in range(1, 22):
        reading[f"sensor_{i:02d}"] = 100 + i*(10 + cycle)

    return reading


def test_models():
    app.state.models = {
        "rul": FakeModel(30),
        "q10": FakeModel(20),
        "q50": FakeModel(28),
        "q90": FakeModel(40),
    }

    payload = {
        "fd": "001",
        "engine_id": 1,
        "history": [
            create_reading(1),
            create_reading(2)
        ]
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200
    print(response.text)

    assert response.json() == {
        "fd": "001",
        "engine_id":1,
        "cycle":2,
        "rul":30.0,
        "q10":20.0,
        "q50":28.0,
        "q90":40.0
    }


def test_predict_invalide_engine():
    app.state.models = {
        "rul": FakeModel(30),
        "q10": FakeModel(20),
        "q50": FakeModel(28),
        "q90": FakeModel(40),
    }

    payload = {
        "fd": "001",
        "engine_id": -1,
        "history": [
            create_reading(1)
        ]
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


def test_empty_history():
    app.state.models = {
        "rul": FakeModel(30),
        "q10": FakeModel(20),
        "q50": FakeModel(28),
        "q90": FakeModel(40),
    }

    payload = {
        "fd": "001",
        "engine_id": 1,
        "history": []
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


def test_missing_sensor():
    app.state.models = {
        "rul": FakeModel(30),
        "q10": FakeModel(20),
        "q50": FakeModel(28),
        "q90": FakeModel(40),
    }

    reading = create_reading(1)
    del reading["sensor_12"]

    payload = {
        "fd": "001",
        "engine_id": 1,
        "history": [
            reading
        ]
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 422


def test_model_info():
    app.state.models = {
        "rul": FakeModel(30),
        "q10": FakeModel(20),
        "q50": FakeModel(28),
        "q90": FakeModel(40),
    }

    response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["loaded"] is True
    assert data["rul_model"] == "FakeModel"
    assert data["q10_model"] == "FakeModel"
    assert data["q50_model"] == "FakeModel"
    assert data["q90_model"] == "FakeModel"
    assert data["windows"] == [5, 20]
    assert data["feature_count"] == len(MODEL_FEATURE_COLUMNS)