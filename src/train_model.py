from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import Bunch
from typing import cast

DATA_DIR = Path("data/processed")

def load_data():
    """Load prepared training and validation datasets."""

    train_data = pd.read_csv(DATA_DIR / "train_split_model.csv")
    validation_data = pd.read_csv(DATA_DIR / "validation_model.csv")

    X_train = train_data.drop(columns=["Target"])
    y_train = train_data["Target"]

    X_validation = validation_data.drop(columns=["Target"])
    y_validation = validation_data["Target"]

    return X_train, X_validation, y_train, y_validation

def train_model(X_train, y_train):
    """Train an XGBoost classifier."""

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
        eval_metric="mlogloss"
    )

    # Give greater importance to underrepresented classes
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



def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on the testing dataset."""

    # Predict the most likely outcome
    predictions = model.predict(X_test)

    # Predict probabilities for all three outcomes
    probabilities = model.predict_proba(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    loss = log_loss(y_test, probabilities)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {loss:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Home Win", "Draw", "Away Win"]
    ))

        # Display confusion matrix
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2]
    )

    print("\nConfusion Matrix:")
    print(matrix)




def show_feature_importance(model, X_train):
    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nFeature Importance:")
    print(importance.to_string(index=False))

def show_permutation_importance(model, X_test, y_test):
    results = cast(Bunch, permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_log_loss",
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    ))

    importance = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": results.importances_mean,
        "Std": results.importances_std
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nPermutation Importance:")
    print(importance.to_string(index=False))


def check_draw_calibration(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        "DrawProbability": probabilities,
        "ActualDraw": (y_test == 1).astype(int)
    })

    results["ProbabilityBin"] = pd.cut(
        results["DrawProbability"],
        bins=[0, 0.1, 0.2, 0.3, 0.4, 1.0],
        labels=["0-10%", "10-20%", "20-30%", "30-40%", "40-100%"],
        include_lowest=True
    )

    calibration = results.groupby(
        "ProbabilityBin", observed=False
    ).agg(
        Matches=("ActualDraw", "count"),
        AveragePredicted=("DrawProbability", "mean"),
        ActualDrawRate=("ActualDraw", "mean")
    )

    calibration["AveragePredicted"] *= 100
    calibration["ActualDrawRate"] *= 100

    print("\nDraw Calibration:")
    print(calibration.round(2).to_string())


def analyze_actual_draws(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)

    results = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "HomeProbability": probabilities[:, 0],
        "DrawProbability": probabilities[:, 1],
        "AwayProbability": probabilities[:, 2]
    })

    actual_draws = results[results["Actual"] == 1].copy()

    actual_draws["DrawRank"] = actual_draws[
        ["HomeProbability", "DrawProbability", "AwayProbability"]
    ].rank(axis=1, ascending=False, method="min")["DrawProbability"]

    rank_counts = actual_draws["DrawRank"].value_counts()

    print("\nAnalysis of Actual Draws:")
    print(f"Total actual draws: {len(actual_draws)}")
    print(f"Average predicted draw probability: "
          f"{actual_draws['DrawProbability'].mean() * 100:.2f}%")

    print("\nDraw Probability Ranking:")
    for rank, label in [
        (1, "Highest"),
        (2, "Second highest"),
        (3, "Lowest")
    ]:
        count = rank_counts.get(rank, 0)
        percentage = count / len(actual_draws) * 100

        print(f"{label}: {count} matches ({percentage:.2f}%)")



def test_draw_thresholds(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)

    thresholds = [0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35]

    print("\nDraw Threshold Experiment:")
    print(
        f"{'Threshold':<12}"
        f"{'Accuracy':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
    )

    for threshold in thresholds:
        predictions = probabilities.argmax(axis=1)

        draw_mask = probabilities[:, 1] >= threshold
        predictions[draw_mask] = 1

        accuracy = accuracy_score(y_test, predictions)

        draw_precision = precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        )
        draw_recall = recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        )
        draw_f1 = f1_score(
            y_test, predictions, pos_label=1, zero_division=0
        )

        print(
            f"{threshold:<12.1%}"
            f"{accuracy:<12.4f}"
            f"{draw_precision:<12.4f}"
            f"{draw_recall:<12.4f}"
            f"{draw_f1:<12.4f}"
        )



def analyze_team_strength_difference():
    data = pd.read_csv(DATA_DIR / "validation.csv")

    data["PointsDiff"] = (
        data["HomeAvgPoints"] - data["AwayAvgPoints"]
    )

    data["AbsolutePointsDiff"] = data["PointsDiff"].abs()

    data["StrengthGroup"] = pd.cut(
        data["AbsolutePointsDiff"],
        bins=[-float("inf"), 0.5, 1.0, 1.5, float("inf")],
        labels=[
            "Very similar (0-0.5)",
            "Similar (0.5-1.0)",
            "Different (1.0-1.5)",
            "Very different (1.5+)"
        ]
    )

    results = data.groupby(
        "StrengthGroup",
        observed=False
    ).agg(
        Matches=("FTR", "count"),
        Draws=("FTR", lambda x: (x == "D").sum())
    )

    results["DrawRate"] = (
        results["Draws"] / results["Matches"] * 100
    )

    print("\nDraw Rate by Team Strength Difference:")
    print(results.round(2).to_string())
    print("\n")


def analyze_defensive_performance():
    data = pd.read_csv(DATA_DIR / "validation.csv")

    home_median = data["HomeAvgGoalsConceded"].median()
    away_median = data["AwayAvgGoalsConceded"].median()

    data["HomeDefence"] = (
        data["HomeAvgGoalsConceded"] <= home_median
    )
    data["AwayDefence"] = (
        data["AwayAvgGoalsConceded"] <= away_median
    )

    data["DefensiveGroup"] = data.apply(
        lambda row: (
            "Both strong"
            if row["HomeDefence"] and row["AwayDefence"]
            else "Home stronger"
            if row["HomeDefence"]
            else "Away stronger"
            if row["AwayDefence"]
            else "Both weaker"
        ),
        axis=1
    )

    results = data.groupby("DefensiveGroup").agg(
        Matches=("FTR", "count"),
        Draws=("FTR", lambda x: (x == "D").sum())
    )

    results["DrawRate"] = (
        results["Draws"] / results["Matches"] * 100
    )

    print("\nDraw Rate by Defensive Performance:")
    print(results.round(2).to_string())


def analyze_historical_draw_rate():
    data = pd.read_csv(DATA_DIR / "all_matches.csv")

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    home_rows = data[["Date", "HomeTeam", "FTR"]].copy()
    home_rows.columns = ["Date", "Team", "Result"]
    home_rows["IsDraw"] = (home_rows["Result"] == "D").astype(int)

    away_rows = data[["Date", "AwayTeam", "FTR"]].copy()
    away_rows.columns = ["Date", "Team", "Result"]
    away_rows["IsDraw"] = (away_rows["Result"] == "D").astype(int)

    team_matches = pd.concat(
        [home_rows, away_rows],
        ignore_index=True
    )

    team_matches = team_matches.sort_values(
        ["Team", "Date"]
    )

    team_matches["PreviousMatches"] = (
        team_matches.groupby("Team").cumcount()
    )

    team_matches["PreviousDraws"] = (
        team_matches.groupby("Team")["IsDraw"].cumsum()
        - team_matches["IsDraw"]
    )

    team_matches["HistoricalDrawRate"] = (
        team_matches["PreviousDraws"]
        / team_matches["PreviousMatches"].replace(0, float("nan"))
    )

    data = data.merge(
        team_matches[["Date", "Team", "HistoricalDrawRate"]]
        .rename(columns={
            "Team": "HomeTeam",
            "HistoricalDrawRate": "HomeHistoricalDrawRate"
        }),
        on=["Date", "HomeTeam"]
    )

    data = data.merge(
        team_matches[["Date", "Team", "HistoricalDrawRate"]]
        .rename(columns={
            "Team": "AwayTeam",
            "HistoricalDrawRate": "AwayHistoricalDrawRate"
        }),
        on=["Date", "AwayTeam"]
    )

    validation = pd.read_csv(DATA_DIR / "validation.csv")
    validation["Date"] = pd.to_datetime(validation["Date"])

    data = data.merge(
        validation[["Date", "HomeTeam", "AwayTeam"]],
        on=["Date", "HomeTeam", "AwayTeam"],
        how="inner"
    )

    data["AverageHistoricalDrawRate"] = (
        data["HomeHistoricalDrawRate"]
        + data["AwayHistoricalDrawRate"]
    ) / 2

    data = data.dropna(subset=["AverageHistoricalDrawRate"])

    data["HistoricalDrawGroup"] = pd.qcut(
        data["AverageHistoricalDrawRate"],
        q=4,
        duplicates="drop"
    )

    results = data.groupby(
        "HistoricalDrawGroup",
        observed=False
    ).agg(
        Matches=("FTR", "count"),
        Draws=("FTR", lambda x: (x == "D").sum()),
        AverageHistoricalRate=("AverageHistoricalDrawRate", "mean")
    )

    results["DrawRate"] = (
        results["Draws"] / results["Matches"] * 100
    )

    results["AverageHistoricalRate"] *= 100

    print("\nDraw Rate by Historical Team Draw Tendency:")
    print(results.round(2).to_string())








def main():
    """Run the model training and validation pipeline."""

    X_train, X_validation, y_train, y_validation = load_data()

    model = train_model(X_train, y_train)

    check_draw_calibration(model, X_validation, y_validation)

    analyze_actual_draws(model, X_validation, y_validation)

    #analyze_defensive_performance()

    #analyze_historical_draw_rate()

    #analyze_team_strength_difference()

    #test_draw_thresholds(model, X_validation, y_validation)

    show_feature_importance(model, X_train)

    #show_permutation_importance(model, X_validation, y_validation)

    evaluate_model(model, X_validation, y_validation)


if __name__ == "__main__":
    main()
