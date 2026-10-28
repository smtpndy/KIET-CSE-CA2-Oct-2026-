
from fastapi.testclient import TestClient
from main import main

client = TestClient(main)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict():
    patient = {
        "Pregnancies": 1,
        "Glucose": 85,
        "BloodPressure": 66,
        "SkinThickness": 29,
        "Insulin": 0,
        "BMI": 26.6,
        "DiabetesPedigreeFunction": 0.351,
        "Age": 31
    }

    response = client.post("/predict", json=patient)
    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]


def test_invalid_input():
    response = client.post("/predict", json={"Age": -1})
    assert response.status_code == 422
