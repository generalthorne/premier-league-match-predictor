
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from prepare_data import FEATURE_COLUMNS
from train_model import train_model, evaluate_model

import numpy as np


DATA_DIR = Path("data/processed")


def main():
    """Train on seven seasons and test on 2025-26."""

    # Load all matches with their pre-built features.
    data = pd.read_csv(DATA_DIR / "matches_with_features.csv")

    data["Date"] = pd.to_datetime(data["Date"])

    # Identify each Premier League season.
    data["Season"] = (
        data["Date"].dt.year
        - (data["Date"].dt.month < 7).astype(int)
    )

    # Select 2018-19 through 2024-25 for training.
    train_data = data[
        (data["Season"] >= 2018)
        & (data["Season"] <= 2024)
    ].copy()

    # Select 2025-26 for testing.
    test_data = data[data["Season"] == 2025].copy()

    print("Final Test — 2025-26")
    print("---------------------")
    print(f"Training matches: {len(train_data)}")
    print(f"Testing matches: {len(test_data)}")

    # Create comparative features, as in prepare_data.py.
    for dataset in [train_data, test_data]:
        dataset["PointsDiff"] = (
            dataset["HomeAvgPoints"] - dataset["AwayAvgPoints"]
        )

        dataset["ShotsOnTargetDiff"] = (
            dataset["HomeAvgShotsOnTarget"]
            - dataset["AwayAvgShotsOnTarget"]
        )

    # Prepare numerical features.
    X_train_numeric = train_data[FEATURE_COLUMNS].fillna(0)
    X_test_numeric = test_data[FEATURE_COLUMNS].fillna(0)

    # Prepare team features.
    X_train_teams = train_data[["HomeTeam", "AwayTeam"]]
    X_test_teams = test_data[["HomeTeam", "AwayTeam"]]

    # Fit the encoder using training seasons only.
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_encoded = np.asarray(
        encoder.fit_transform(X_train_teams)
    )

    X_test_encoded = np.asarray(
        encoder.transform(X_test_teams)
    )

    # Get encoded feature names.
    encoded_columns = encoder.get_feature_names_out(
        ["HomeTeam", "AwayTeam"]
    )

    # Combine numerical and encoded features.
    X_train = pd.concat([
        X_train_numeric.reset_index(drop=True),
        pd.DataFrame(X_train_encoded, columns=encoded_columns)
    ], axis=1)

    X_test = pd.concat([
        X_test_numeric.reset_index(drop=True),
        pd.DataFrame(X_test_encoded, columns=encoded_columns)
    ], axis=1)

    # Prepare target labels.
    y_train = train_data["FTR"].map({"H": 0, "D": 1, "A": 2})
    y_test = test_data["FTR"].map({"H": 0, "D": 1, "A": 2})

    # Train and evaluate using your existing functions.
    model = train_model(X_train, y_train)

    print("\nFinal Test Results:")
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
