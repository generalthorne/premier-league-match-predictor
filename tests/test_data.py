import pandas as pd
import pytest
from pathlib import Path

DATA_DIR = Path("data/processed")

@pytest.fixture
def data():
    return pd.read_csv("data/processed/matches_with_features.csv")


def test_match_count(data):
    assert len(data) == 2280


def test_match_results_preserved(data):
    assert data["FTR"].notna().all()


def test_feature_columns_exist(data):
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

    assert all(column in data.columns for column in feature_columns)


def test_historical_averages_nonnegative(data):
    feature_columns = [col for col in data.columns if col.startswith(("HomeAvg", "AwayAvg"))]

    assert (data[feature_columns].dropna() >= 0).all().all()


def test_initial_matches_have_missing_averages(data):
    feature_columns = [col for col in data.columns if col.startswith(("HomeAvg", "AwayAvg"))]

    assert data[feature_columns].isna().any().any()



def test_rolling_average_uses_previous_matches():
    matches = pd.DataFrame({
        "Team": ["Arsenal", "Arsenal", "Arsenal", "Arsenal", "Arsenal", "Arsenal"],
        "GoalsScored": [2, 1, 3, 0, 2, 4]
    })

    rolling_average = (
        matches.groupby("Team")["GoalsScored"]
        .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    )

    assert pd.isna(rolling_average.iloc[0])
    assert rolling_average.iloc[1] == 2.0
    assert rolling_average.iloc[2] == 1.5
    assert rolling_average.iloc[5] == 1.6


def test_training_and_testing_files_exist():
    assert (DATA_DIR / "train_model.csv").exists()
    assert (DATA_DIR / "test_model.csv").exists()


def test_training_data_comes_before_testing_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    train["Date"] = pd.to_datetime(train["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    assert train["Date"].max() < test["Date"].min()


def test_training_and_testing_features_match():
    train = pd.read_csv(DATA_DIR / "train_model.csv")
    test = pd.read_csv(DATA_DIR / "test_model.csv")

    train_features = train.drop(columns=["Target"]).columns
    test_features = test.drop(columns=["Target"]).columns

    assert list(train_features) == list(test_features)


def test_target_labels_are_valid():
    valid_labels = {0, 1, 2}

    train = pd.read_csv(DATA_DIR / "train_model.csv")
    test = pd.read_csv(DATA_DIR / "test_model.csv")

    assert set(train["Target"].unique()).issubset(valid_labels)
    assert set(test["Target"].unique()).issubset(valid_labels)


def test_prepared_data_has_no_missing_values():
    train = pd.read_csv(DATA_DIR / "train_model.csv")
    test = pd.read_csv(DATA_DIR / "test_model.csv")

    assert not train.isna().any().any()
    assert not test.isna().any().any()


def test_training_and_testing_data_are_not_empty():
    train = pd.read_csv(DATA_DIR / "train_model.csv")
    test = pd.read_csv(DATA_DIR / "test_model.csv")

    assert len(train) > 0
    assert len(test) > 0


def test_team_features_are_one_hot_encoded():
    train = pd.read_csv(DATA_DIR / "train_model.csv")

    team_columns = [
        col for col in train.columns
        if col.startswith(("HomeTeam_", "AwayTeam_"))
    ]

    assert len(team_columns) > 0
    assert set(train[team_columns].stack().unique()).issubset({0, 1})