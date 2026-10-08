from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict():
    payload = {
        "PULocationID": 229,
        "DOLocationID": 237,
        "trip_distance": 1.6,
        "pickup_hour": 0,
        "pickup_weekday": 2,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert "predicted_duration_minutes" in body
    assert isinstance(
        body["predicted_duration_minutes"],
        float,
    )
    assert body["predicted_duration_minutes"] > 0


def test_invalid_pickup_hour():
    payload = {
        "PULocationID": 229,
        "DOLocationID": 237,
        "trip_distance": 1.6,
        "pickup_hour": 25,
        "pickup_weekday": 2,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422