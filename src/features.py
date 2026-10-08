FEATURE_COLUMNS = [
    "PULocationID",
    "DOLocationID",
    "trip_distance",
    "pickup_hour",
    "pickup_weekday",
]

TARGET_COLUMN = "duration"


def get_features_and_target(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y