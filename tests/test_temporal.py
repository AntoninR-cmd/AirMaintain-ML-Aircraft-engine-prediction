import pandas as pd
import pytest
from aeromaintain.features.temporal import add_temporal_features
from aeromaintain.config import SENSOR_COLUMNS


def create_engine_dataframe():
    data = {
        "FD": ["001"] * 4,
        "IdMoteur": [1] * 4,
        "Cycle": [1, 2, 3, 4]
    }

    for sensor in SENSOR_COLUMNS:
        data[sensor] = [0.0, 0.0, 0.0, 0.0]

    data["MesureCapteur01"] = [
        10.0,
        12.0,
        14.0,
        20.0,
    ]

    return pd.DataFrame(data)


def test_delta_rolling_features():
    df = create_engine_dataframe()

    result = add_temporal_features(df, windows=(2, 3))

    assert result["MesureCapteur01_delta1"].tolist() == [
        0.0,
        2.0,
        2.0,
        6.0
    ]

    assert result["MesureCapteur01_mean_2"].tolist() == [
        10,
        11,
        13,
        17
    ]

    assert result["MesureCapteur01_trend_3"].tolist() == [
        0.0,
        0.0,
        2.0,
        4.0
    ]


def test_future():
    df = create_engine_dataframe()

    result_before = add_temporal_features(df, windows=(2, 3))

    modified_df = df.copy()

    modified_df.loc[
        modified_df["Cycle"] == 4,
        "MesureCapteur01"
    ] = 1_000_000

    result_after = add_temporal_features(
        modified_df,
        windows=(2, 3)
    )

    columns = [
        "MesureCapteur01_delta1",
        "MesureCapteur01_mean_2",
        "MesureCapteur01_std_2",
        "MesureCapteur01_mean_3",
        "MesureCapteur01_std_3",
        "MesureCapteur01_trend_3",
    ]

    pd.testing.assert_frame_equal(
        result_before.loc[
            result_before["Cycle"] <= 3,
            columns
        ].reset_index(drop=True),

        result_after.loc[
            result_after["Cycle"] <= 3,
            columns
        ].reset_index(drop=True),
    )


def test_features_by_engine():
    rows = []

    for engine_id, values in [
        (1, [10, 20]),
        (2, [100, 120])
    ]:
        for cycle, value in enumerate(
            values, 
            start=1
        ):
            row = {
                "FD": "001",
                "IdMoteur": engine_id,
                "Cycle": cycle
            }

            for sensor in SENSOR_COLUMNS:
                row[sensor] = 0.0

            row["MesureCapteur01"] = value

            rows.append(row)

    df = pd.DataFrame(rows)

    result = add_temporal_features(
        df, 
        windows=(2, )
    )

    first_cycle_engine_2 = result[
        (result["IdMoteur"] == 2) &
        (result["Cycle"] == 1)
    ].iloc[0]

    assert first_cycle_engine_2["MesureCapteur01_delta1"] == 0
