import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib

logger = logging.getLogger("pricepulse.registry")


class ModelRegistry:
    def __init__(self, registry_config):
        self.config = registry_config
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save(self, model, scaler, pca, feature_names, metrics, extra_metadata=None):
        model_path = self.config.artifacts_dir / self.config.model_filename
        bundle = {"model": model, "scaler": scaler, "pca": pca}
        joblib.dump(bundle, model_path)

        metadata = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "feature_names": feature_names,
            "metrics": {
                k: (float(v) if isinstance(v, (int, float)) else v)
                for k, v in metrics.items()
                if k in ("accuracy", "cv_mean", "cv_std")
            },
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        meta_path = self.config.artifacts_dir / self.config.metadata_filename
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"مدل در {model_path} و متادیتا در {meta_path} ذخیره شد.")
        return model_path, meta_path

    def load(self):
        model_path = self.config.artifacts_dir / self.config.model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"مدلی در {model_path} یافت نشد.")
        return joblib.load(model_path)
