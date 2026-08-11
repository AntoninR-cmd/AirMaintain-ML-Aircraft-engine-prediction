import pandas as pd
from aeromaintain.config import SENSOR_COLUMNS, WINDOWS, BASE_FEATURE_COLUMNS


def get_temporal_feature_columns(windows):
    temporal_columns = []

    long_window = max(windows)

    for sensor in SENSOR_COLUMNS:
        temporal_columns.append(
            f"{sensor}_delta1"
        )

        for window in windows:
            temporal_columns.append(
                f"{sensor}_mean_{window}"
            )
            temporal_columns.append(
                f"{sensor}_std_{window}"
            )

        temporal_columns.append(
            f"{sensor}_trend_{long_window}"
        )

    return temporal_columns


def add_temporal_features(df, windows=(5, 20)):
    """
    Crée des caractéristiques temporelles en utilisant uniquement
    le cycle actuel et les cycles précédents du même moteur.
    """

    df = df.sort_values(
        ["FD", "IdMoteur", "Cycle"]
    ).copy()

    grouped = df.groupby(
        ["FD", "IdMoteur"],
        sort=False
    )

    new_features = {}

    for sensor in SENSOR_COLUMNS:
        # Variation depuis le cycle précédent
        new_features[f"{sensor}_delta1"] = (
            grouped[sensor]
            .diff()
            .fillna(0)
        )

        for window in windows:
            # Moyenne mobile
            new_features[f"{sensor}_mean_{window}"] = (
                grouped[sensor]
                .transform(
                    lambda serie: serie.rolling(
                        window=window,
                        min_periods=1
                    ).mean()
                )
            )

            # Écart-type mobile
            new_features[f"{sensor}_std_{window}"] = (
                grouped[sensor]
                .transform(
                    lambda serie: serie.rolling(
                        window=window,
                        min_periods=2
                    ).std(ddof=0)
                )
                .fillna(0)
            )

        # Tendance approximative sur la plus grande fenêtre
        long_window = max(windows)

        new_features[f"{sensor}_trend_{long_window}"] = (
            grouped[sensor]
            .diff(long_window - 1)
            .div(long_window - 1)
            .fillna(0)
        )

    temporal_df = pd.DataFrame(
        new_features,
        index=df.index
    )

    return pd.concat(
        [df, temporal_df],
        axis=1
    )


TEMPORAL_FEATURE_COLUMNS = get_temporal_feature_columns(
    WINDOWS
)

MODEL_FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    + TEMPORAL_FEATURE_COLUMNS
)