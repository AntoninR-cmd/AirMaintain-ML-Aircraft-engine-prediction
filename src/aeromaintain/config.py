from pathlib import Path

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

DATASET_FOLDER = FILE_PATH.parents[2] / "dataset"

RESULTS_FOLDER = DATASET_FOLDER / "results"
MODELS_FOLDER = RESULTS_FOLDER / "models"
PREDICTIONS_FOLDER = RESULTS_FOLDER / "predictions"

RESULTS_FOLDER.mkdir(exist_ok=True)
MODELS_FOLDER.mkdir(exist_ok=True)
PREDICTIONS_FOLDER.mkdir(exist_ok=True)

