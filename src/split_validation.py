from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")


def split_validation():
    data = pd.read_csv(DATA_DIR / "train.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    split_index = int(len(data) * 0.8)
    split_date = data.iloc[split_index]["Date"]

    train_data = data[data["Date"] < split_date]
    validation_data = data[data["Date"] >= split_date]

    train_data.to_csv(DATA_DIR / "train_split.csv", index=False)
    validation_data.to_csv(DATA_DIR / "validation.csv", index=False)


if __name__ == "__main__":
    split_validation()