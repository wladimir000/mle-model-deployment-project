from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path("artifacts/model.joblib")


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}. "
        "Run the training first with: uv run python -m src.train"
    )


model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="NYC Taxi Duration Predictor",
    description=(
        "Local API for predicting NYC Yellow Taxi trip duration."
    ),
    version="1.0.0",
)


class TripRequest(BaseModel):
    PULocationID: int = Field(
        gt=0,
        description="NYC TLC pickup location ID",
    )

    DOLocationID: int = Field(
        gt=0,
        description="NYC TLC dropoff location ID",
    )

    trip_distance: float = Field(
        gt=0,
        description="Trip distance in miles",
    )

    pickup_hour: int = Field(
        ge=0,
        le=23,
        description="Pickup hour from 0 to 23",
    )

    pickup_weekday: int = Field(
        ge=0,
        le=6,
        description="Pickup weekday where Monday=0 and Sunday=6",
    )


class PredictionResponse(BaseModel):
    predicted_duration_minutes: float


@app.get("/")
def root():
    return {
        "message": "NYC Taxi Duration Prediction API",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "model_path": str(MODEL_PATH),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    trip: TripRequest,
):
    features = pd.DataFrame(
        [
            {
                "PULocationID": trip.PULocationID,
                "DOLocationID": trip.DOLocationID,
                "trip_distance": trip.trip_distance,
                "pickup_hour": trip.pickup_hour,
                "pickup_weekday": trip.pickup_weekday,
            }
        ]
    )

    prediction = model.predict(features)[0]

    return PredictionResponse(
        predicted_duration_minutes=round(
            float(prediction),
            2,
        )
    )