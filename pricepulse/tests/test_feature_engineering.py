import numpy as np
import pandas as pd

from config import PCAConfig
from features.engineering import FeatureEngineer


def test_add_domain_features_columns():
    df = pd.DataFrame({
        "px_height": [800], "px_width": [1200], "sc_h": [12], "sc_w": [7],
        "ram": [3000], "n_cores": [4], "four_g": [1], "three_g": [1],
        "wifi": [1], "blue": [1], "dual_sim": [1], "pc": [10], "fc": [5],
        "battery_power": [1500], "mobile_wt": [150],
    })
    result = FeatureEngineer.add_domain_features(df)

    expected_new_cols = {
        "pixel_density", "screen_area", "ram_per_core",
        "connectivity_score", "camera_score", "battery_efficiency",
    }
    assert expected_new_cols.issubset(set(result.columns))
    assert result["screen_area"].iloc[0] == 84
    assert result["camera_score"].iloc[0] == 15
    assert result["connectivity_score"].iloc[0] == 5


def test_add_domain_features_does_not_mutate_input():
    df = pd.DataFrame({
        "px_height": [800], "px_width": [1200], "sc_h": [12], "sc_w": [7],
        "ram": [3000], "n_cores": [4], "four_g": [1], "three_g": [1],
        "wifi": [1], "blue": [1], "dual_sim": [1], "pc": [10], "fc": [5],
        "battery_power": [1500], "mobile_wt": [150],
    })
    original_cols = set(df.columns)
    FeatureEngineer.add_domain_features(df)
    assert set(df.columns) == original_cols


def test_pca_disabled_returns_input_unchanged():
    config = PCAConfig(enabled=False)
    engineer = FeatureEngineer(config)
    X = np.random.rand(20, 5)
    result = engineer.fit_pca(X)
    assert np.array_equal(result, X)
    assert engineer.explained_variance() is None


def test_pca_enabled_reduces_dimensions():
    config = PCAConfig(enabled=True, n_components=3)
    engineer = FeatureEngineer(config)
    X = np.random.rand(50, 10)
    result = engineer.fit_pca(X)
    assert result.shape == (50, 3)
    assert 0.0 <= engineer.explained_variance() <= 1.0
