import json

from config import RegistryConfig
from registry.model_registry import ModelRegistry
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification


def test_save_and_load_roundtrip(tmp_path):
    config = RegistryConfig(artifacts_dir=tmp_path / "artifacts")
    registry = ModelRegistry(config)

    X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
    model = LogisticRegression().fit(X, y)
    scaler = StandardScaler().fit(X)

    model_path, meta_path = registry.save(
        model=model, scaler=scaler, pca=None,
        feature_names=["a", "b", "c", "d", "e"],
        metrics={"accuracy": 0.9, "cv_mean": 0.88, "cv_std": 0.02},
    )

    assert model_path.exists()
    assert meta_path.exists()

    loaded = registry.load()
    assert loaded["model"] is not None
    assert loaded["scaler"] is not None

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["metrics"]["accuracy"] == 0.9
    assert metadata["feature_names"] == ["a", "b", "c", "d", "e"]


def test_load_raises_when_missing(tmp_path):
    config = RegistryConfig(artifacts_dir=tmp_path / "empty")
    registry = ModelRegistry(config)
    try:
        registry.load()
        assert False, "should have raised"
    except FileNotFoundError:
        pass
