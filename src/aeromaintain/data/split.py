from sklearn.model_selection import train_test_split
from aeromaintain.config import MODEL_FEATURE_COLUMNS


def split_validation_test_by_engine(test):
    moteurs = test[
        ["FD", "IdMoteur"]
    ].drop_duplicates()

    moteurs_test, moteurs_validation = train_test_split(
        moteurs,
        test_size=0.5,
        random_state=42,
        stratify=moteurs["FD"]
    )

    test_final = test.merge(
        moteurs_test,
        on=["FD", "IdMoteur"],
        how="inner",
        validate="many_to_one"
    )

    validation = test.merge(
        moteurs_validation,
        on=["FD", "IdMoteur"],
        how="inner",
        validate="many_to_one"
    )

    cles_test = set(
        map(
            tuple,
            test_final[
                ["FD", "IdMoteur"]
            ].drop_duplicates().to_numpy()
        )
    )

    cles_validation = set(
        map(
            tuple,
            validation[
                ["FD", "IdMoteur"]
            ].drop_duplicates().to_numpy()
        )
    )

    assert cles_test.isdisjoint(
        cles_validation
    )

    X_test = test_final[
        MODEL_FEATURE_COLUMNS
    ].copy()

    y_test = test_final["RUL"].copy()

    X_val = validation[
        MODEL_FEATURE_COLUMNS
    ].copy()

    y_val = validation["RUL"].copy()

    return X_test, y_test, X_val, y_val
