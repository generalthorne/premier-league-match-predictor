
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/all_matches.csv")
OUTPUT_PATH = Path("data/processed/matches_with_features.csv")

STATS = ["GoalsScored", "GoalsConceded", "Shots", "ShotsOnTarget", "Points"]

VENUE_STATS = ["Points", "GoalsScored", "GoalsConceded"]


def load_matches():
    """Load and prepare historical match data."""
    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    data["MatchID"] = data.index

    return data


def calculate_team_features(data):
    """Calculate rolling and season-to-date averages from previous matches."""

    home = pd.DataFrame({
        "Venue": "Home",
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
        "Venue": "Away",
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

    # Calculate venue-specific averages from previous matches.
    for stat in VENUE_STATS:
        team_matches[f"VenueAvg{stat}"] = (
            team_matches.groupby(["Team", "Venue"])[stat]
            .transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            )
        )

    # Identify the starting year of each Premier League season.
    team_matches["Season"] = (
        team_matches["Date"].dt.year
        - (team_matches["Date"].dt.month < 7).astype(int)
    )

    for stat in STATS:
        # Calculate averages from the previous five matches.
        team_matches[f"Avg{stat}"] = (
            team_matches.groupby("Team")[stat]
            .transform(
                lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
            )
        )

        # Calculate averages from the previous ten matches.
        team_matches[f"Avg10{stat}"] = (
            team_matches.groupby("Team")[stat]
            .transform(
                lambda x: x.shift(1).rolling(window=10, min_periods=1).mean()
            )
        )

        # Calculate averages from all previous matches in the current season.
        team_matches[f"SeasonAvg{stat}"] = (
            team_matches.groupby(["Team", "Season"])[stat]
            .transform(
                lambda x: x.shift(1).expanding(min_periods=1).mean()
            )
        )

    # Count the team's previous matches in the current season.
    team_matches["SeasonMatchesPlayed"] = (
        team_matches.groupby(["Team", "Season"]).cumcount()
    )

    feature_columns = (
        [f"Avg{s}" for s in STATS]
        + [f"Avg10{s}" for s in STATS]
        + [
            "SeasonAvgPoints",
            "SeasonAvgGoalsScored",
            "SeasonAvgGoalsConceded",
            "SeasonMatchesPlayed"
        ]
        + [f"VenueAvg{s}" for s in VENUE_STATS]
    )

    return team_matches[["MatchID", "Team"] + feature_columns]


def build_features():
    """Build the final dataset with historical home and away statistics."""

    data = load_matches()
    team_features = calculate_team_features(data)

    home_features = team_features.rename(columns={
        "VenueAvgPoints": "HomeHomeAvgPoints",
        "VenueAvgGoalsScored": "HomeHomeAvgGoalsScored",
        "VenueAvgGoalsConceded": "HomeHomeAvgGoalsConceded",
        "Team": "HomeTeam",
        "SeasonMatchesPlayed": "HomeSeasonMatchesPlayed",
        **{f"Avg{s}": f"HomeAvg{s}" for s in STATS},
        **{f"Avg10{s}": f"HomeAvg10{s}" for s in STATS},
        **{
            f"SeasonAvg{s}": f"HomeSeasonAvg{s}"
            for s in ["Points", "GoalsScored", "GoalsConceded"]
        }
    })

    away_features = team_features.rename(columns={
        "VenueAvgPoints": "AwayAwayAvgPoints",
        "VenueAvgGoalsScored": "AwayAwayAvgGoalsScored",
        "VenueAvgGoalsConceded": "AwayAwayAvgGoalsConceded",
        "Team": "AwayTeam",
        "SeasonMatchesPlayed": "AwaySeasonMatchesPlayed",
        **{f"Avg{s}": f"AwayAvg{s}" for s in STATS},
        **{f"Avg10{s}": f"AwayAvg10{s}" for s in STATS},
        **{
            f"SeasonAvg{s}": f"AwaySeasonAvg{s}"
            for s in ["Points", "GoalsScored", "GoalsConceded"]
        }
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
