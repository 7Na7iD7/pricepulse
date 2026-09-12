import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

FEATURE_ORDER = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi",
]

VALID_PAYLOAD = {
    "battery_power": 1500, "blue": 1, "clock_speed": 2.0, "dual_sim": 1, "fc": 5,
    "four_g": 1, "int_memory": 32, "m_dep": 0.5, "mobile_wt": 150, "n_cores": 4,
    "pc": 10, "px_height": 800, "px_width": 1200, "ram": 3500, "sc_h": 12, "sc_w": 7,
    "talk_time": 15, "three_g": 1, "touch_screen": 1, "wifi": 1,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    X, y = make_classification(
        n_samples=200, n_features=20, n_informative=10, n_classes=4,
        n_clusters_per_class=1, random_state=42,
    )
    X_df = pd.DataFrame(X, columns=FEATURE_ORDER)
    scaler = StandardScaler().fit(X_df)
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(X_df), y)

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    model_path = artifacts_dir / "pricepulse_model.joblib"
    joblib.dump({"model": model, "scaler": scaler, "pca": None}, model_path)

    meta_path = artifacts_dir / "pricepulse_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({"metrics": {"accuracy": 0.9}}, f)

    monkeypatch.chdir(tmp_path)
    import serving.api as api_module
    api_module._bundle = None
    api_module.ARTIFACT_PATH = Path("artifacts/pricepulse_model.joblib")

    return TestClient(api_module.app)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["price_range"] in (0, 1, 2, 3)
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["class_probabilities"]) == 4


def test_predict_rejects_out_of_range_ram(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["ram"] = 999999
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_rejects_invalid_binary_field(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["wifi"] = 5
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_batch(client):
    r = client.post("/predict/batch", json={"items": [VALID_PAYLOAD, VALID_PAYLOAD]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["feature_count"] == 20


def test_predict_explain_returns_top_factors(client):
    r = client.post("/predict/explain", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["price_range"] in (0, 1, 2, 3)
    assert len(body["top_factors"]) == 5
    for item in body["top_factors"]:
        assert "feature" in item and "contribution" in item
    assert "explanation_note" in body


def test_predict_explain_rejects_invalid_input(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["ram"] = -100
    r = client.post("/predict/explain", json=bad_payload)
    assert r.status_code == 422
