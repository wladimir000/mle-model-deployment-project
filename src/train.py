from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.data import load_raw_data, prepare_data
from src.features import get_features_and_target


RANDOM_STATE = 42
MAX_ROWS = 500_000

MLFLOW_DB = Path("mlflow.db")
ARTIFACT_DIR = Path("artifacts")
FINAL_MODEL_PATH = ARTIFACT_DIR / "model.joblib"


def train_and_log_run(
    run_name: str,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    save_as_final: bool = False,
):
    print()
    print(f"Starting run: {run_name}")

    raw_df = load_raw_data()
    df = prepare_data(raw_df)

    if len(df) > MAX_ROWS:
        print(f"Sampling {MAX_ROWS:,} rows...")
        df = df.sample(
            n=MAX_ROWS,
            random_state=RANDOM_STATE,
        )

    X, y = get_features_and_target(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("Training RandomForestRegressor...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    rmse = root_mean_squared_error(
        y_val,
        predictions,
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param(
            "n_estimators",
            n_estimators,
        )
        mlflow.log_param(
            "max_depth",
            max_depth,
        )
        mlflow.log_param(
            "min_samples_leaf",
            min_samples_leaf,
        )
        mlflow.log_param(
            "max_rows",
            MAX_ROWS,
        )
        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        mlflow.log_metric(
            "rmse",
            rmse,
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree",
            ],
        )

    if save_as_final:
        ARTIFACT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            FINAL_MODEL_PATH,
        )

        print(
            f"Final model saved to: "
            f"{FINAL_MODEL_PATH}"
        )

    print(
        f"RMSE: {rmse:.4f} minutes"
    )

    print(
        f"Finished run: {run_name}"
    )

    return rmse


def main():
    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB.resolve().as_posix()}"
    )

    mlflow.set_experiment(
        "nyc-taxi-duration"
    )

    baseline_rmse = train_and_log_run(
        run_name="random-forest-baseline",
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        save_as_final=False,
    )

    improved_rmse = train_and_log_run(
        run_name="random-forest-improved",
        n_estimators=200,
        max_depth=25,
        min_samples_leaf=2,
        save_as_final=True,
    )

    print()
    print("Comparison")
    print("----------")

    print(
        f"Baseline RMSE: "
        f"{baseline_rmse:.4f}"
    )

    print(
        f"Improved RMSE: "
        f"{improved_rmse:.4f}"
    )

    improvement = (
        baseline_rmse - improved_rmse
    )

    print(
        f"Improvement: "
        f"{improvement:.4f} minutes"
    )


if __name__ == "__main__":
    main()