from fastapi.testclient import TestClient

from src.app.main import app

client=TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data.get("model_version")
    assert data.get("mlflow_run_id")
    assert data.get("mlflow_model_version") is not None

def test_predict_accepts_valid_input():
    response = client.post(
        "/predict",
        json={
            "tenure_months": 12,
            "monthly_charges": 75.5,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
            "support_calls": 2,
        },
    )

    assert response.status_code == 200
    assert "churn_probability" in response.json()

def test_negative_tenure_is_rejected():
    response = client.post(
        "/predict",
        json={
            "tenure_months": -1,
            "monthly_charges": 75.5,
            "contract": "Month-to-month",
            "internet_service": "Fiber optic",
            "support_calls": 2,
        },
    )

    assert response.status_code == 422

def test_prediction_is_deterministic():
    customer = {
        "tenure_months": 12,
        "monthly_charges": 75.5,
        "contract": "Month-to-month",
        "internet_service": "Fiber optic",
        "support_calls": 2,
    }

    first_response = client.post("/predict", json=customer)
    second_response = client.post("/predict", json=customer)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert (
        first_response.json()["churn_probability"]
        == second_response.json()["churn_probability"]
    )
