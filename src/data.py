from pathlib import Path

import pandas as pd


DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/"
    "trip-data/yellow_tripdata_2025-01.parquet"
)

DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "yellow_tripdata_2025-01.parquet"


def download_data() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists():
        print(f"Dataset already exists: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    print("Downloading January 2025 NYC Yellow Taxi data...")

    df = pd.read_parquet(DATA_URL)
    df.to_parquet(RAW_DATA_PATH, index=False)

    print(f"Saved dataset to: {RAW_DATA_PATH}")
    return RAW_DATA_PATH


def load_raw_data() -> pd.DataFrame:
    path = download_data()
    return pd.read_parquet(path)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.weekday

    df = df[
        (df["duration"] >= 1)
        & (df["duration"] <= 60)
        & (df["trip_distance"] > 0)
    ].copy()

    return df


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = prepare_data(raw_df)

    print(f"Raw rows:   {len(raw_df):,}")
    print(f"Clean rows: {len(clean_df):,}")
    print()

    print("Duration statistics:")
    print(clean_df["duration"].describe())
    print()

    print("Selected columns:")
    print(
        clean_df[
            [
                "PULocationID",
                "DOLocationID",
                "trip_distance",
                "pickup_hour",
                "pickup_weekday",
                "duration",
            ]
        ].head()
    )