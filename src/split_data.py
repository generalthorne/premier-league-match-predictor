
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/processed/matches_with_features.csv")
OUTPUT_DIR = Path("data/processed")

start_season = 2022

def split_data():
    """Split matches into training, validation, and testing by season."""

    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    # Identify the starting year of each Premier League season.
    data["Season"] = (
        data["Date"].dt.year
        - (data["Date"].dt.month < 7).astype(int)
    )

    # 2010–11 through 2023–24
    train_data = data[data["Season"].between(start_season, 2023)]

    # 2024–25
    validation_data = data[data["Season"] == 2024]

    # 2025–26
    test_data = data[data["Season"] == 2025]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation_data.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test_data.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("Dataset split completed!")
    print(f"Training matches: {len(train_data)}")
    print(f"Validation matches: {len(validation_data)}")
    print(f"Test matches: {len(test_data)}")


if __name__ == "__main__":
    split_data()
