
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/all_matches.csv")
OUTPUT_PATH = Path("data/processed/matches_with_features.csv")

STATS = ["GoalsScored", "GoalsConceded", "Shots", "ShotsOnTarget", "Points"]


def load_matches():
    """Load and prepare historical match data."""
    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    data["MatchID"] = data.index

    return data


def calculate_team_features(data):
    """Calculate each team's rolling averages from previous matches."""

    home = pd.DataFrame({
        "MatchID": data["MatchID"],
        "Date": data["Date"],
        "Team": data["HomeTeam"],
        "GoalsScored": data["FTHG"],
        "GoalsConceded": data["FTAG"],
        "Shots": data["HS"],
        "ShotsOnTarget": data["HST"],
        "Points": data["FTR"].map({"H": 3, "D": 1, "A": 0})
    })

    away = pd.DataFrame({
        "MatchID": data["MatchID"],
        "Date": data["Date"],
        "Team": data["AwayTeam"],
        "GoalsScored": data["FTAG"],
        "GoalsConceded": data["FTHG"],
        "Shots": data["AS"],
        "ShotsOnTarget": data["AST"],
        "Points": data["FTR"].map({"A": 3, "D": 1, "H": 0})
    })

    team_matches = pd.concat([home, away], ignore_index=True)
    team_matches = team_matches.sort_values(["Team", "Date", "MatchID"])

    for stat in STATS:
        team_matches[f"Avg{stat}"] = (
            team_matches.groupby("Team")[stat]
            .transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
        )

    return team_matches[["MatchID", "Team"] + [f"Avg{s}" for s in STATS]]


def build_features():
    """Build the final dataset with historical home and away statistics."""

    data = load_matches()
    team_features = calculate_team_features(data)

    home_features = team_features.rename(columns={
        "Team": "HomeTeam",
        **{f"Avg{s}": f"HomeAvg{s}" for s in STATS}
    })

    away_features = team_features.rename(columns={
        "Team": "AwayTeam",
        **{f"Avg{s}": f"AwayAvg{s}" for s in STATS}
    })

    data = data.merge(
        home_features,
        on=["MatchID", "HomeTeam"],
        how="left"
    )

    data = data.merge(
        away_features,
        on=["MatchID", "AwayTeam"],
        how="left"
    )

    data = data.drop(columns=["MatchID"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    build_features()
