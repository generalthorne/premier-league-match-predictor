
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/matches_with_features.csv")
OUTPUT_DIR = Path("data/processed")


def split_data():
    """Split match data chronologically into training and testing sets."""

    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    split_index = int(len(data) * 0.8)
    split_date = data.iloc[split_index]["Date"]

    train_data = data[data["Date"] < split_date]
    test_data = data[data["Date"] >= split_date]

    train_data.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_data.to_csv(OUTPUT_DIR / "test.csv", index=False)


if __name__ == "__main__":
    split_data()
