"""
Tests for FakeShield API.
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

# Train models once for the whole test session
from app.model_manager import ModelManager
_manager = ModelManager()
_manager.train(n_samples=400)   # small dataset for speed

# Patch the app's manager before importing main
import app.main as main_module
main_module.manager = _manager

from app.main import app

client = TestClient(app)


# ── Health ─────────────────────────────────────────────────────────────────────
def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True
    assert len(data["available_models"]) == 5


def test_list_models():
    r = client.get("/models")
    assert r.status_code == 200
    models = r.json()["models"]
    ids = [m["id"] for m in models]
    assert "ensemble" in ids
    assert "logistic_regression" in ids


# ── Single prediction ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("Scientists confirm vaccine safety in peer-reviewed study published in Nature", "REAL"),
    ("SHOCKING: Government hiding alien DNA in water supply – share before deleted!", "FAKE"),
    ("Federal Reserve raises interest rates by 25 basis points amid inflation concerns", "REAL"),
    ("PROOF: 5G towers secretly controlling your BRAIN every night!!", "FAKE"),
])
def test_predict_obvious_cases(text, expected):
    r = client.post("/predict", json={"text": text, "model": "ensemble"})
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] == expected
    assert 0 <= data["confidence"] <= 100
    assert abs(data["probability_real"] + data["probability_fake"] - 100) < 0.1


@pytest.mark.parametrize("model_id", [
    "ensemble", "logistic_regression", "linear_svm", "random_forest", "naive_bayes"
])
def test_all_models_respond(model_id):
    r = client.post("/predict", json={
        "text": "Scientists publish findings on climate change in leading journal",
        "model": model_id,
    })
    assert r.status_code == 200
    assert r.json()["prediction"] in ("REAL", "FAKE")


def test_predict_returns_latency():
    r = client.post("/predict", json={"text": "Test headline", "model": "ensemble"})
    assert r.status_code == 200
    assert r.json()["latency_ms"] is not None


# ── Validation errors ──────────────────────────────────────────────────────────
def test_empty_text_rejected():
    r = client.post("/predict", json={"text": "   ", "model": "ensemble"})
    assert r.status_code == 422


def test_text_too_short_rejected():
    r = client.post("/predict", json={"text": "Hi", "model": "ensemble"})
    assert r.status_code == 422


def test_invalid_model_rejected():
    r = client.post("/predict", json={"text": "Valid headline text here", "model": "gpt4"})
    assert r.status_code == 422


# ── Batch prediction ───────────────────────────────────────────────────────────
def test_batch_predict():
    texts = [
        "Scientists confirm vaccine safety in peer-reviewed study",
        "SHOCKING: Government cover-up exposed – media silent!",
        "Federal Reserve raises rates amid inflation concerns",
    ]
    r = client.post("/predict/batch", json={"texts": texts, "model": "ensemble"})
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 3
    assert data["total_latency_ms"] > 0


def test_batch_too_large_rejected():
    texts = ["headline"] * 51
    r = client.post("/predict/batch", json={"texts": texts, "model": "ensemble"})
    assert r.status_code == 422


# ── Stats ──────────────────────────────────────────────────────────────────────
def test_stats_update_after_predictions():
    before = client.get("/stats").json()["total_predictions"]
    client.post("/predict", json={"text": "Scientists publish major climate study findings", "model": "ensemble"})
    after = client.get("/stats").json()["total_predictions"]
    assert after == before + 1
