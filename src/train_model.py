from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss

DATA_DIR = Path("data/processed")

def load_data():
    """Load prepared training and testing datasets."""

    train_data = pd.read_csv(DATA_DIR / "train_model.csv")
    test_data = pd.read_csv(DATA_DIR / "test_model.csv")

    X_train = train_data.drop(columns=["Target"])
    y_train = train_data["Target"]

    X_test = test_data.drop(columns=["Target"])
    y_test = test_data["Target"]

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train an XGBoost classifier."""

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)

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


def main():
    """Run the model training and evaluation pipeline."""

    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()