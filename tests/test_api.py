from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_health():
    """Vérifie que l'endpoint /health répond avec status 200."""
    r = client.get("/health")
    assert r.status_code == 200


def test_predict_positive():
    """Vérifie qu'une prédiction retourne une structure correcte."""
    r = client.post(
        "/predict",
        json={"text": "Ce produit est excellent !"}
    )

    assert r.status_code == 200

    data = r.json()

    # Attention: pas d'espaces dans les clés JSON
    assert data["label"] in ["POSITIVE", "NEGATIVE", "NEUTRAL"]

    # Score entre 0 et 1
    assert 0 <= data["score"] <= 1


def test_predict_empty_fails():
    """Vérifie que Pydantic rejette un texte vide (422)."""
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422