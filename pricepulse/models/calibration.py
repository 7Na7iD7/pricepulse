import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import label_binarize

logger = logging.getLogger("pricepulse.calibration")


class ModelCalibrator:
    """
    کالیبراسیون احتمالات خروجی مدل.

    یک مدل با دقت بالا لزوماً احتمالات قابل‌اعتمادی تولید نمی‌کند. برای
    مثال وقتی مدل می‌گوید «۹۹٪ مطمئنم»، این باید در عمل هم نزدیک به ۹۹٪
    درست باشد. Brier Score این «قابل‌اعتماد بودن» احتمالات را می‌سنجد
    (هرچه کمتر، بهتر).
    """

    def __init__(self, method="sigmoid", cv=5):
        self.method = method
        self.cv = cv
        self.calibrated_model = None

    def calibrate(self, base_model, X_train, y_train):
        self.calibrated_model = CalibratedClassifierCV(
            estimator=base_model, method=self.method, cv=self.cv
        )
        self.calibrated_model.fit(X_train, y_train)
        return self.calibrated_model

    @staticmethod
    def multiclass_brier_score(model, X_test, y_test, n_classes):
        probabilities = model.predict_proba(X_test)
        y_binarized = label_binarize(y_test, classes=list(range(n_classes)))
        return float(np.mean(np.sum((probabilities - y_binarized) ** 2, axis=1)))

    def compare(self, base_model, calibrated_model, X_test, y_test, n_classes):
        base_brier = self.multiclass_brier_score(base_model, X_test, y_test, n_classes)
        calibrated_brier = self.multiclass_brier_score(calibrated_model, X_test, y_test, n_classes)
        return {
            "base_brier_score": base_brier,
            "calibrated_brier_score": calibrated_brier,
            "improved": calibrated_brier < base_brier,
        }
