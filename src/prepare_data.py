
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_PATH = Path("data/processed/matches_with_features.csv")
OUTPUT_PATH = Path("data/processed/model_data.csv")


def prepare_data():
    """Prepare features and target for model training."""

    data = pd.read_csv(DATA_PATH)

    feature_columns = [
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

    # Encode team names
    encoder = LabelEncoder()

    data["HomeTeamEncoded"] = encoder.fit_transform(data["HomeTeam"])
    data["AwayTeamEncoded"] = encoder.transform(data["AwayTeam"])

    # Create model features
    X = data[feature_columns + ["HomeTeamEncoded", "AwayTeamEncoded"]]

    # Create target labels
    y = data["FTR"].map({"H": 0, "D": 1, "A": 2})

    # Combine features and target
    model_data = X.copy()
    model_data["Target"] = y

    # Handle missing historical averages
    model_data = model_data.fillna(0)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    prepare_data()
