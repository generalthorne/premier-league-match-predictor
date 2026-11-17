import pandas as pd
import pytest

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
