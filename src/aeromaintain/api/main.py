from fastapi import FastAPI, Request
from aeromaintain.api.schemas import (
    PredictionRequest,
    PredictionResponse
)
from aeromaintain.api.converters import request_to_dataframe
from aeromaintain.features.temporal import add_temporal_features
from aeromaintain.config import WINDOWS
from contextlib import asynccontextmanager
from aeromaintain.inference.predictor import (
    load_prediction_models,
    predict_from_dataframe,
)


@asynccontextmanager
async def lifespan(app: FastAPI):

    models = load_prediction_models()

    app.state.models = models

    yield

app = FastAPI(
    title="AeroMaintain API",
    version= "0.1.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"version": "0.1.0"}

@app.get("/engine/{engine_id}")
def engine_id(engine_id : int):
    return {
        "IdMoteur": engine_id
    }

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(request_data: PredictionRequest,
                request: Request):
    raw_df = request_to_dataframe(request_data)

    featured_df = add_temporal_features(raw_df, windows=WINDOWS)

    models = request.app.state.models

    prediction_df = predict_from_dataframe(
        featured_df,
        models
    )

    prediction = prediction_df.iloc[0]

    return {
        "fd": prediction["FD"],
        "engine_id": int(prediction["IdMoteur"]),
        "cycle": int(prediction["Cycle"]),
        "rul": float(prediction["RUL_predite"]),
        "q10": float(prediction["q10"]),
        "q50": float(prediction["q50"]),
        "q90": float(prediction["q90"])
    }