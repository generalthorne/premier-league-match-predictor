
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

# Columns we want to retain
columns = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HS",
    "AS",
    "HST",
    "AST"
]

# Load all historical seasons
files = sorted(DATA_DIR.glob("E0*.csv"))

if not files:
    raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

matches = []

for file in files:
    df = pd.read_csv(file)

    df["Date"] = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    # Keep only relevant columns
    missing_columns = [col for col in columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"{file.name} is missing columns: {missing_columns}")

    df = df[columns]

    # Remove rows without actual match results
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])

    matches.append(df)

# Combine seasons
data = pd.concat(matches, ignore_index=True)

# Sort chronologically
data = data.sort_values("Date").reset_index(drop=True)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Save combined dataset
data.to_csv(OUTPUT_DIR / "all_matches.csv", index=False)

# Display summary
print("Combined dataset created!")
print(f"Total matches: {len(data)}")
print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
print("\nMissing values:")
print(data.isna().sum())
print("\nFirst five matches:")
print(data.head())
