from pathlib import Path
from features.temporal import get_temporal_feature_columns

FILE_PATH = Path(__file__).resolve()

SENSOR_COLUMNS = [
    f"MesureCapteur{i:02d}"
    for i in range(1, 22)
]

BASE_FEATURE_COLUMNS = [
    "Cycle",
    "ParameterOpe1",
    "ParameterOpe2",
    "ParameterOpe3",
    *SENSOR_COLUMNS
]

COLUMNS = [
    "IdMoteur",
    *BASE_FEATURE_COLUMNS
]

WINDOWS = (5, 20)

DATASET_FOLDER = FILE_PATH.parents[2]
print(DATASET_FOLDER)

RESULTS_FOLDER = DATASET_FOLDER / "results"
MODELS_FOLDER = RESULTS_FOLDER / "models"
PREDICTIONS_FOLDER = RESULTS_FOLDER / "predictions"

RESULTS_FOLDER.mkdir(exist_ok=True)
MODELS_FOLDER.mkdir(exist_ok=True)
PREDICTIONS_FOLDER.mkdir(exist_ok=True)


TEMPORAL_FEATURE_COLUMNS = get_temporal_feature_columns(
    WINDOWS
)

MODEL_FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    + TEMPORAL_FEATURE_COLUMNS
)
