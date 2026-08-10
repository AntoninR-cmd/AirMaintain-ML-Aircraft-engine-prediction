import pandas as pd
from aeromaintain.config import DATASET_FOLDER, COLUMNS, WINDOWS, MODEL_FEATURE_COLUMNS
from features.temporal import add_temporal_features



def load_train_data(folder = DATASET_FOLDER / "train"):
    '''
    This function allow to load the data to train and 
    validate the best model
    Arguments : 0
    Return X_train and y_train pd.Dataframe with the 
    factors (X_train) and the target (y_train)
    '''
    dataframes = []

    for file in sorted(folder.glob("train_FD*.txt")):
        fd = file.stem[-3:]

        train_file = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            names=COLUMNS
        )

        train_file["FD"] = fd
        dataframes.append(train_file)

    if not dataframes:
        raise FileNotFoundError(
            f"No training file found in {folder}"
        )

    train = pd.concat(
        dataframes,
        ignore_index=True
    )

    train["RUL"] = (
        train.groupby(
            ["FD", "IdMoteur"]
        )["Cycle"].transform("max")
        - train["Cycle"]
    )

    train = add_temporal_features(
        train,
        windows=WINDOWS
    )

    X_train = train[
        MODEL_FEATURE_COLUMNS
    ].copy()

    y_train = train["RUL"].copy()

    return X_train, y_train


def load_X_test_data(folder = DATASET_FOLDER / "X_test"):
    '''
    This function load the factors for the test phase.
    Arguments : 0
    Return test, a pd.Dataframe with all the factors to test
    '''
    
    dataframes = []

    for file in sorted(folder.glob("test_FD*.txt")):
        test_file = pd.read_csv(
            file,
            sep=r"\s+",
            header=None,
            names=COLUMNS
        )

        test_file["FD"] = file.stem[-3:]
        dataframes.append(test_file)

    if not dataframes:
        raise FileNotFoundError(
            f"No file found in {folder}"
        )

    test = pd.concat(
        dataframes,
        ignore_index=True
    )

    test = add_temporal_features(
        test,
        windows=WINDOWS
    )

    return test


def load_y_test_data(test, folder = DATASET_FOLDER / "y_test"):
    '''
    This function allow to load the data for the target for the test phase.
    Arguments : test, pd.Dataframe with the factors for the test phase
    Return test, pd.Dataframe with the factors and the target for the test phase
    '''
        
    dataframes = []
    for file in folder.iterdir():
        fd = file.stem[-3:]

        RUL_file = pd.read_csv(file, sep=r"\s+", header=None, names=["RUL"])

        RUL_file['FD'] = fd
        RUL_file["IdMoteur"] = range(1, len(RUL_file)+1)

        dataframes.append(RUL_file)

    if not dataframes:
        raise FileNotFoundError(
            f"No file found in {folder}"
        )

    y_test = pd.concat(dataframes, ignore_index=True)
    
    test = test.merge(
        y_test[["FD", "IdMoteur", "RUL"]], 
        on=["FD", "IdMoteur"],
        how="left",
        validate="many_to_one"
    )

    if test["RUL"].isna().any():
        lignes_manquantes = test.loc[
            test["RUL"].isna(),
            ["FD", "IdMoteur"]
        ].drop_duplicates()

        raise ValueError(
            f"RUL not found for some engines :\n{lignes_manquantes}"
        )

    test = test.rename(columns={"RUL": "RUL_fin"})
    test["Cycle_max"] = test.groupby(["IdMoteur", "FD"])["Cycle"].transform("max")
    test["RUL"] = (
        test["RUL_fin"] + (test["Cycle_max"] - test["Cycle"])
    )

    return test
