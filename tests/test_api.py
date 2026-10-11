from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    data = {
        "pclass": 3,
        "sex": "male",
        "age": 22,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S",
        "class_": "Third",
        "who": "man",
        "alone": "True"
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "probability" in result
    assert result["prediction"] in [0, 1]
    assert 0 <= result["probability"] <= 1


def test_invalid_request():
    data = {
        "pclass": "wrong",
        "sex": "male",
        "age": 22,
        "sibsp": 1,
        "parch": 0,
        "fare": 7.25,
        "embarked": "S",
        "class_": "Third",
        "who": "man",
        "alone": "True"
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 422