
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/model_data.csv")
OUTPUT_DIR = Path("data/processed")


def split_data():
    """Split match data chronologically into training and testing sets."""

    data = pd.read_csv(DATA_PATH)

    split_index = int(len(data) * 0.8)

    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]

    train_data.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_data.to_csv(OUTPUT_DIR / "test.csv", index=False)


if __name__ == "__main__":
    split_data()
