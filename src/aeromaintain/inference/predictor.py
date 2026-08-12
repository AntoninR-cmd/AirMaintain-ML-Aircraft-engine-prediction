import joblib
from pathlib import Path
import pandas as pd

from aeromaintain.config import (
    DATASET_FOLDER,
    COLUMNS,
    WINDOWS,
    MODELS_FOLDER,
)
from aeromaintain.features.temporal import (
    add_temporal_features,
    MODEL_FEATURE_COLUMNS,
)


def load_engine_data(folder: Path = DATASET_FOLDER / "entry") -> pd.DataFrame:
    """
    Charge les fichiers .txt contenant l'historique d'un ou plusieurs moteurs,
    ajoute la colonne FD, concatène les fichiers puis calcule les features temporelles.

    Les fichiers doivent utiliser le même format que les fichiers C-MAPSS NASA.
    """

    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Le dossier d'entrée n'existe pas : {folder}"
        )

    dataframes = []

    for file in sorted(folder.glob("*.txt")):
        fd = file.stem[-3:]

        engine_file = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            names=COLUMNS,
        )

        engine_file["FD"] = fd
        dataframes.append(engine_file)

    if not dataframes:
        raise FileNotFoundError(
            f"Aucun fichier moteur .txt trouvé dans {folder}"
        )

    engines = pd.concat(
        dataframes,
        ignore_index=True,
    )

    engines = add_temporal_features(
        engines,
        windows=WINDOWS,
    )

    return engines


def load_prediction_models(
    models_folder: Path = MODELS_FOLDER,
) -> dict:
    """
    Charge le meilleur modèle de RUL ainsi que les trois modèles quantiles.
    """

    models_folder = Path(models_folder)

    model_paths = {
        "rul": models_folder / "best_model_final.joblib",
        "q10": models_folder / "q10_quantile.joblib",
        "q50": models_folder / "q50_quantile.joblib",
        "q90": models_folder / "q90_quantile.joblib",
    }

    missing_models = [
        str(path)
        for path in model_paths.values()
        if not path.exists()
    ]

    if missing_models:
        raise FileNotFoundError(
            "Certains modèles sont introuvables :\n"
            + "\n".join(missing_models)
        )

    return {
        name: joblib.load(path)
        for name, path in model_paths.items()
    }


def predict_from_dataframe(engines, models):
    latest_cycles = get_latest_engine_cycles(engines)

    missing_columns = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column not in engines.columns
    ]

    if missing_columns:
        raise ValueError(
            "Certaines features nécessaires au modèle sont absentes : "
            + ", ".join(missing_columns)
        )

    X = latest_cycles[
        MODEL_FEATURE_COLUMNS
    ].copy()

    predictions = latest_cycles[
        ["FD", "IdMoteur", "Cycle"]
    ].copy()

    predictions["RUL_predite"] = models["rul"].predict(X)
    predictions["q10"] = models["q10"].predict(X)
    predictions["q50"] = models["q50"].predict(X)
    predictions["q90"] = models["q90"].predict(X)

    return predictions


def predict_engines(
    folder: Path = DATASET_FOLDER / "entry",
    models_folder: Path = MODELS_FOLDER,
) -> pd.DataFrame:
    """
    Prédit la RUL actuelle et les quantiles q10/q50/q90
    pour chaque moteur présent dans le dossier d'entrée.

    Une seule prédiction est produite par moteur : celle correspondant
    à son dernier cycle disponible.
    """

    engines = load_engine_data(folder)
    models = load_prediction_models(models_folder)

    predictions = predict_from_dataframe(engines, models)

    return predictions.reset_index(drop=True)


def get_latest_engine_cycles(engines):
    # Le modèle doit prédire l'état courant du moteur :
    # on conserve donc uniquement le dernier cycle connu de chaque moteur.
    latest_cycles = (
        engines
        .sort_values(["FD", "IdMoteur", "Cycle"])
        .groupby(["FD", "IdMoteur"], as_index=False)
        .tail(1)
        .copy()
    )

    return latest_cycles

if __name__ == "__main__":
    results = predict_engines()
    print(results.to_string(index=False))