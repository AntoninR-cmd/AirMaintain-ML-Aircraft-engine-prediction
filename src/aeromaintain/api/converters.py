import pandas as pd
from aeromaintain.api.schemas import PredictionRequest


def request_to_dataframe(
    request: PredictionRequest
) -> pd.DataFrame:
    rows = []

    for reading in request.history:
        row = {
            "FD": request.fd,
            "IdMoteur": request.engine_id,
            "Cycle": reading.cycle,
            "ParameterOpe1": reading.parameter_ope1,
            "ParameterOpe2": reading.parameter_ope2,
            "ParameterOpe3": reading.parameter_ope3
        }

        for i in range(1, 22):
            row[f"MesureCapteur{i:02d}"] = getattr(
                reading, 
                f"sensor_{i:02d}"
            )

        rows.append(row)

    return pd.DataFrame(rows)