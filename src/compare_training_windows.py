
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path("data/processed")
DATA_PATH = DATA_DIR / "matches_with_features.csv"

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

WINDOW_SIZES = range(2, 9)

# Evaluation seasons: 2018-19 through 2024-25.
EVALUATION_SEASONS = range(2018, 2025)


def load_data() -> pd.DataFrame:
    """Load matches with features and identify their seasons."""

    data = pd.read_csv(DATA_PATH)

    data["Date"] = pd.to_datetime(data["Date"])

    data["Season"] = (
        data["Date"].dt.year
        - (data["Date"].dt.month < 7).astype(int)
    )

    data = data.sort_values("Date").reset_index(drop=True)

    # Create comparative features used by the existing model.
    data["PointsDiff"] = (
        data["HomeAvgPoints"] - data["AwayAvgPoints"]
    )

    data["ShotsOnTargetDiff"] = (
        data["HomeAvgShotsOnTarget"]
        - data["AwayAvgShotsOnTarget"]
    )

    return data


def prepare_features(
    train_data: pd.DataFrame,
    evaluation_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare features and fit the team encoder on training data only."""

    X_train_numeric = train_data[FEATURE_COLUMNS].fillna(0)
    X_evaluation_numeric = evaluation_data[FEATURE_COLUMNS].fillna(0)

    X_train_teams = train_data[["HomeTeam", "AwayTeam"]]
    X_evaluation_teams = evaluation_data[["HomeTeam", "AwayTeam"]]

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    # Convert encoder output to NumPy arrays for consistent typing.
    X_train_encoded = np.asarray(
        encoder.fit_transform(X_train_teams)
    )

    X_evaluation_encoded = np.asarray(
        encoder.transform(X_evaluation_teams)
    )

    encoded_columns = encoder.get_feature_names_out(
        ["HomeTeam", "AwayTeam"]
    )

    X_train = pd.concat([
        X_train_numeric.reset_index(drop=True),
        pd.DataFrame(
            X_train_encoded,
            columns=encoded_columns
        )
    ], axis=1)

    X_evaluation = pd.concat([
        X_evaluation_numeric.reset_index(drop=True),
        pd.DataFrame(
            X_evaluation_encoded,
            columns=encoded_columns
        )
    ], axis=1)

    return X_train, X_evaluation


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> XGBClassifier:
    """Train using the existing XGBoost configuration."""

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
        eval_metric="mlogloss"
    )

    class_weights = {
        0: 1.0,
        1: 1.0,
        2: 1.0
    }

    sample_weights = y_train.map(class_weights)

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    return model


def main() -> None:
    """Compare different training-window lengths."""

    data = load_data()

    detailed_results = []
    predictions_by_window = {}

    for window_size in WINDOW_SIZES:

        predictions_by_window[window_size] = {
            "actual": [],
            "predicted": [],
            "probabilities": []
        }

        print(f"\nTesting {window_size}-season training window")

        for evaluation_season in EVALUATION_SEASONS:

            first_training_season = evaluation_season - window_size

            train_data = data[
                data["Season"].between(
                    first_training_season,
                    evaluation_season - 1
                )
            ].copy()

            evaluation_data = data[
                data["Season"] == evaluation_season
            ].copy()

            # Verify that every intended training season is present.
            expected_seasons = set(
                range(first_training_season, evaluation_season)
            )

            actual_seasons = set(train_data["Season"].unique())

            if actual_seasons != expected_seasons:
                raise ValueError(
                    f"Missing training seasons for "
                    f"{window_size}-season window before "
                    f"{evaluation_season}."
                )

            if train_data.empty or evaluation_data.empty:
                raise ValueError(
                    f"Empty dataset for evaluation season "
                    f"{evaluation_season}."
                )

            y_train = train_data["FTR"].map({
                "H": 0,
                "D": 1,
                "A": 2
            })

            y_evaluation = evaluation_data["FTR"].map({
                "H": 0,
                "D": 1,
                "A": 2
            })

            if y_train.isna().any() or y_evaluation.isna().any():
                raise ValueError("Unexpected match result found.")

            X_train, X_evaluation = prepare_features(
                train_data,
                evaluation_data
            )

            model = train_model(X_train, y_train)

            predicted = model.predict(X_evaluation)
            probabilities = model.predict_proba(X_evaluation)

            accuracy = accuracy_score(
                y_evaluation,
                predicted
            )

            loss = log_loss(
                y_evaluation,
                probabilities,
                labels=[0, 1, 2]
            )

            detailed_results.append({
                "WindowSize": window_size,
                "EvaluationSeason": evaluation_season,
                "TrainingMatches": len(train_data),
                "EvaluationMatches": len(evaluation_data),
                "Accuracy": accuracy,
                "LogLoss": loss
            })

            predictions_by_window[window_size]["actual"].extend(
                y_evaluation.tolist()
            )

            predictions_by_window[window_size]["predicted"].extend(
                predicted.tolist()
            )

            predictions_by_window[window_size]["probabilities"].extend(
                probabilities.tolist()
            )

            print(
                f"  {evaluation_season}-{evaluation_season + 1}: "
                f"Accuracy = {accuracy:.4f}, "
                f"Log Loss = {loss:.4f}"
            )

    # Save individual evaluation-season results.
    detailed = pd.DataFrame(detailed_results)

    detailed.to_csv(
        DATA_DIR / "training_window_details.csv",
        index=False
    )

    # Aggregate results for each training-window size.
    summary_results = []

    for window_size in WINDOW_SIZES:

        window_data = detailed[
            detailed["WindowSize"] == window_size
        ]

        actual = predictions_by_window[window_size]["actual"]
        predicted = predictions_by_window[window_size]["predicted"]
        probabilities = predictions_by_window[window_size]["probabilities"]

        summary_results.append({
            "WindowSize": window_size,
            "MeanAccuracy": window_data["Accuracy"].mean(),
            "AccuracyStd": window_data["Accuracy"].std(),
            "MeanLogLoss": window_data["LogLoss"].mean(),
            "LogLossStd": window_data["LogLoss"].std(),
            "OverallAccuracy": accuracy_score(actual, predicted),
            "OverallLogLoss": log_loss(
                actual,
                probabilities,
                labels=[0, 1, 2]
            )
        })

    summary = pd.DataFrame(summary_results)

    summary.to_csv(
        DATA_DIR / "training_window_summary.csv",
        index=False
    )

    print("\nTraining Window Comparison:")
    print(summary.round(4).to_string(index=False))

    print("\nResults saved:")
    print("data/processed/training_window_details.csv")
    print("data/processed/training_window_summary.csv")


if __name__ == "__main__":
    main()
