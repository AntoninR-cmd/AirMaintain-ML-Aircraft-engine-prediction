import pandas as pd
from aeromaintain.data.split import split_egine_key


def create_engines():
    rows = []

    for fd in ["001", "002"]:
        for engine_id in range(1, 5):
            rows.append(
                {
                    "FD": fd,
                    "IdMoteur": engine_id
                }
            )

    return pd.DataFrame(rows)


def test_engine_split_is_disjoint():
    engines = create_engines()

    test_engines, val_engines = split_egine_key(engines)

    test_keys = set(
        map(
            tuple,
            test_engines[
                ["FD", "IdMoteur"]
            ].to_numpy()
        )
    )

    val_keys = set(
        map(
            tuple,
            val_engines[
                ["FD", "IdMoteur"]
            ].to_numpy()
        )
    )

    assert test_keys.isdisjoint(val_keys)

    assert len(test_keys) + len(val_keys) == 8


def split_keeps_fd_distribution():
    engines = create_engines()

    test_engines, val_engines = split_egine_key(
        engines,
        test_size=0.5
    )

    assert (
        test_engines["FD"].value_counts().to_dict()
        ==
        {
            "001": 2,
            "002": 2
        }
    )

    assert (
        val_engines["FD"].value_counts().to_dict()
        ==
        {
            "001": 2,
            "002": 2
        }
    )