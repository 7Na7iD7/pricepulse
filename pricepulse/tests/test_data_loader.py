import pandas as pd
import pytest

from config import DataConfig
from data.loader import DataLoader, DataLoadError


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "battery_power": [1000, 1500, 1200, 1800, 900, 1600, 1100, 1700, 1300, 1400,
                           1050, 1550, 1250, 1850, 950, 1650, 1150, 1750, 1350, 1450],
        "ram": [1000, 2000, 1500, 3000, 800, 2500, 1200, 2800, 1600, 1900,
                1050, 2050, 1550, 3050, 850, 2550, 1250, 2850, 1650, 1950],
        "price_range": [0, 2, 1, 3, 0, 3, 1, 3, 1, 2,
                         0, 2, 1, 3, 0, 3, 1, 3, 1, 2],
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_raw_missing_file_raises():
    config = DataConfig(csv_path="/nonexistent/path.csv")
    loader = DataLoader(config)
    with pytest.raises(DataLoadError):
        loader.load_raw()


def test_load_raw_missing_target_column(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    config = DataConfig(csv_path=str(path), target_column="price_range")
    loader = DataLoader(config)
    with pytest.raises(DataLoadError):
        loader.load_raw()


def test_split_and_scale_shapes(sample_csv):
    config = DataConfig(csv_path=sample_csv, test_size=0.3, random_state=42)
    loader = DataLoader(config)
    data = loader.split_and_scale()

    assert data["X_train"].shape[0] + data["X_test"].shape[0] == 20
    assert data["X_train"].shape[1] == 2
    assert len(data["y_train"]) == data["X_train"].shape[0]
    assert data["feature_names"] == ["battery_power", "ram"]


def test_split_and_scale_is_standardized(sample_csv):
    config = DataConfig(csv_path=sample_csv, test_size=0.3, random_state=42)
    loader = DataLoader(config)
    data = loader.split_and_scale()

    means = data["X_train"].mean(axis=0)
    assert all(abs(m) < 1.0 for m in means)
