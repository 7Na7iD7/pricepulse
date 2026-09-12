from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    csv_path: str = "mobile_price_data.csv"
    target_column: str = "price_range"
    test_size: float = 0.3
    random_state: int = 42


@dataclass
class PCAConfig:
    n_components: int = 10
    enabled: bool = False


@dataclass
class TuningConfig:
    n_trials: int = 40
    cv_folds: int = 5
    timeout_seconds: int = 300
    random_state: int = 42
    per_model_trials: dict = field(default_factory=lambda: {
        "logistic_regression": 30, "knn": 20, "naive_bayes": 20, "svm": 30,
        "random_forest": 30, "gradient_boosting": 30,
        "xgboost": 50, "lightgbm": 50, "catboost": 40, "mlp": 30,
    })


@dataclass
class RegistryConfig:
    artifacts_dir: Path = Path("artifacts")
    model_filename: str = "pricepulse_model.joblib"
    metadata_filename: str = "pricepulse_metadata.json"


@dataclass
class ExplainabilityConfig:
    enabled: bool = True
    background_sample_size: int = 50
    explain_sample_size: int = 30


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
