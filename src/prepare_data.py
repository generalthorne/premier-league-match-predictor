
from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

DATA_DIR = Path("data/processed")


FEATURE_COLUMNS = [
    # Previous five-match averages
    "HomeAvgGoalsScored",
    "HomeAvgGoalsConceded",
    "HomeAvgShots",
    "HomeAvgShotsOnTarget",
    "HomeAvgPoints",
    "AwayAvgGoalsScored",
    "AwayAvgGoalsConceded",
    "AwayAvgShots",
    "AwayAvgShotsOnTarget",
    "AwayAvgPoints",

    # Ten-match averages
    "HomeAvg10GoalsScored",
    "HomeAvg10GoalsConceded",
    "HomeAvg10Shots",
    "HomeAvg10ShotsOnTarget",
    "HomeAvg10Points",
    "AwayAvg10GoalsScored",
    "AwayAvg10GoalsConceded",
    "AwayAvg10Shots",
    "AwayAvg10ShotsOnTarget",
    "AwayAvg10Points",

    # Comparative features
    "PointsDiff",
    "ShotsOnTargetDiff",

    # Season-to-date averages
    "HomeSeasonAvgPoints",
    "HomeSeasonAvgGoalsScored",
    "HomeSeasonAvgGoalsConceded",
    "AwaySeasonAvgPoints",
    "AwaySeasonAvgGoalsScored",
    "AwaySeasonAvgGoalsConceded",

    # Season matches played
    "HomeSeasonMatchesPlayed",
    "AwaySeasonMatchesPlayed",

    # Venue-specific averages
    "HomeHomeAvgPoints",
    "AwayAwayAvgPoints",
    "HomeHomeAvgGoalsScored",
    "HomeHomeAvgGoalsConceded",
    "AwayAwayAvgGoalsScored",
    "AwayAwayAvgGoalsConceded"
]


def prepare_data():
    """Prepare training, validation, and test data."""

    train_data = pd.read_csv(DATA_DIR / "train.csv")
    validation_data = pd.read_csv(DATA_DIR / "validation.csv")
    test_data = pd.read_csv(DATA_DIR / "test.csv")

    datasets = [train_data, validation_data, test_data]

    # Create comparative features
    for data in datasets:
        data["PointsDiff"] = (
            data["HomeAvgPoints"] - data["AwayAvgPoints"]
        )

        data["ShotsOnTargetDiff"] = (
            data["HomeAvgShotsOnTarget"] - data["AwayAvgShotsOnTarget"]
        )

    # Separate numerical and categorical features
    X_train_numeric = train_data[FEATURE_COLUMNS].fillna(0)
    X_validation_numeric = validation_data[FEATURE_COLUMNS].fillna(0)
    X_test_numeric = test_data[FEATURE_COLUMNS].fillna(0)

    X_train_teams = train_data[["HomeTeam", "AwayTeam"]]
    X_validation_teams = validation_data[["HomeTeam", "AwayTeam"]]
    X_test_teams = test_data[["HomeTeam", "AwayTeam"]]

    # Fit the encoder using training data only
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_encoded = encoder.fit_transform(X_train_teams)
    X_validation_encoded = encoder.transform(X_validation_teams)
    X_test_encoded = encoder.transform(X_test_teams)

    # Save the fitted encoder for future predictions
    joblib.dump(encoder, DATA_DIR / "team_encoder.joblib")

    # Get encoded feature names
    encoded_columns = encoder.get_feature_names_out(
        ["HomeTeam", "AwayTeam"]
    )

    # Combine numerical and encoded team features
    def combine_features(numeric, encoded):
        return pd.concat([
            numeric.reset_index(drop=True),
            pd.DataFrame(encoded, columns=encoded_columns)
        ], axis=1)

    X_train = combine_features(X_train_numeric, X_train_encoded)
    X_validation = combine_features(
        X_validation_numeric, X_validation_encoded
    )
    X_test = combine_features(X_test_numeric, X_test_encoded)

    # Create target labels
    y_train = train_data["FTR"].map({"H": 0, "D": 1, "A": 2})
    y_validation = validation_data["FTR"].map({"H": 0, "D": 1, "A": 2})
    y_test = test_data["FTR"].map({"H": 0, "D": 1, "A": 2})

    # Save prepared datasets
    X_train.assign(Target=y_train.reset_index(drop=True)).to_csv(
        DATA_DIR / "train_split_model.csv", index=False
    )

    X_validation.assign(Target=y_validation.reset_index(drop=True)).to_csv(
        DATA_DIR / "validation_model.csv", index=False
    )

    X_test.assign(Target=y_test.reset_index(drop=True)).to_csv(
        DATA_DIR / "test_model.csv", index=False
    )

    print("Data preparation completed!")
    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_validation)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Total model features: {X_train.shape[1]}")


if __name__ == "__main__":
    prepare_data()
