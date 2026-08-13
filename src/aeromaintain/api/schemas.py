from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    cycle: int = Field(gt=0)

    parameter_ope1: float
    parameter_ope2: float
    parameter_ope3: float

    sensor_01: float
    sensor_02: float
    sensor_03: float
    sensor_04: float
    sensor_05: float
    sensor_06: float
    sensor_07: float
    sensor_08: float
    sensor_09: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):
    fd: str
    engine_id: int = Field(gt=0)
    history: list[SensorReading] = Field(min_length=1)


class PredictionResponse(BaseModel):
    fd: str
    engine_id: int = Field(gt=0)
    cycle: int = Field(gt=0)
    rul: float
    q10: float
    q50: float
    q90: float
