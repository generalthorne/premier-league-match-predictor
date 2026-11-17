
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path("data/processed")

FEATURE_COLUMNS = [
    "HomeAvgGoalsScored",
    "HomeAvgGoalsConceded",
    "HomeAvgShots",
    "HomeAvgShotsOnTarget",
    "HomeAvgPoints",
    "AwayAvgGoalsScored",
    "AwayAvgGoalsConceded",
    "AwayAvgShots",
    "AwayAvgShotsOnTarget",
    "AwayAvgPoints"
]


def prepare_data():
    """Prepare training and testing data for model training."""

    train_data = pd.read_csv(DATA_DIR / "train.csv")
    test_data = pd.read_csv(DATA_DIR / "test.csv")

    # Separate numerical and categorical features
    X_train_numeric = train_data[FEATURE_COLUMNS]
    X_test_numeric = test_data[FEATURE_COLUMNS]

    X_train_teams = train_data[["HomeTeam", "AwayTeam"]]
    X_test_teams = test_data[["HomeTeam", "AwayTeam"]]

    # Fit the encoder using training data only
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    X_train_encoded = encoder.fit_transform(X_train_teams)
    X_test_encoded = encoder.transform(X_test_teams)

    # Combine numerical and encoded team features
    encoded_columns = encoder.get_feature_names_out(["HomeTeam", "AwayTeam"])

    X_train = pd.concat([
        X_train_numeric.reset_index(drop=True),
        pd.DataFrame(X_train_encoded, columns=encoded_columns)
    ], axis=1)

    X_test = pd.concat([
        X_test_numeric.reset_index(drop=True),
        pd.DataFrame(X_test_encoded, columns=encoded_columns)
    ], axis=1)

    # Create target labels
    y_train = train_data["FTR"].map({"H": 0, "D": 1, "A": 2})
    y_test = test_data["FTR"].map({"H": 0, "D": 1, "A": 2})

    # Handle missing historical averages
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Save prepared datasets
    X_train.assign(Target=y_train.reset_index(drop=True)).to_csv(
        DATA_DIR / "train_model.csv", index=False
    )

    X_test.assign(Target=y_test.reset_index(drop=True)).to_csv(
        DATA_DIR / "test_model.csv", index=False
    )


if __name__ == "__main__":
    prepare_data()
