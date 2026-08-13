from aeromaintain.api.converters import request_to_dataframe
from aeromaintain.api.schemas import PredictionRequest
import pandas as pd


def test_converter():
    reading ={
        "cycle": 40,
        "parameter_ope1": 0.0,
        "parameter_ope2": 0.0,
        "parameter_ope3": 100.0,

        "sensor_01": 518.67,
        "sensor_02": 642.1,
        "sensor_03": 1590.0,
        "sensor_04": 1400.0,
        "sensor_05": 14.62,
        "sensor_06": 21.61,
        "sensor_07": 553.5,
        "sensor_08": 2388.0,
        "sensor_09": 9046.0,
        "sensor_10": 1.3,
        "sensor_11": 47.4,
        "sensor_12": 521.5,
        "sensor_13": 2388.0,
        "sensor_14": 8138.0,
        "sensor_15": 8.42,
        "sensor_16": 0.03,
        "sensor_17": 392.0,
        "sensor_18": 2388.0,
        "sensor_19": 100.0,
        "sensor_20": 39.05,
        "sensor_21": 23.42
        }

    request = PredictionRequest(
        fd="001",
        engine_id=4,
        history=[reading]
    )

    df = request_to_dataframe(request)

    assert len(df) == 1

    assert df.iloc[0]["FD"] == "001"

    assert df.iloc[0]["IdMoteur"] == 4

    assert df.iloc[0]["Cycle"] == 40

    assert df.iloc[0]["MesureCapteur01"] == 518.67