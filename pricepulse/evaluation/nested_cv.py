import logging

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from models.base_models import get_model_registry, get_search_spaces

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("pricepulse.nested_cv")


class NestedCVEvaluator:
    """
    ارزیابی بدون سوگیری (unbiased) عملکرد یک مدل با تیونینگ ابرپارامتر.

    مشکل رایج: وقتی از همان داده برای تیونینگ ابرپارامتر (با CV) و سپس
    گزارش دقت (با همان CV) استفاده می‌شود، دقت گزارش‌شده خوش‌بینانه
    (optimistic) است. Nested CV با یک حلقه‌ی بیرونی مستقل از فرایند
    تیونینگ، تخمین واقعی‌تری از عملکرد روی داده‌ی دیده‌نشده ارائه می‌دهد.
    """

    def __init__(self, outer_folds=5, inner_folds=3, n_trials=10, timeout_seconds=20, random_state=42):
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.random_state = random_state
        self.model_registry = get_model_registry()
        self.search_spaces = get_search_spaces()

    def _tune_on_fold(self, model_name, X_train, y_train):
        model_cls = self.model_registry[model_name]
        space_fn = self.search_spaces[model_name]
        inner_cv = StratifiedKFold(
            n_splits=self.inner_folds, shuffle=True, random_state=self.random_state
        )

        def objective(trial):
            params = space_fn(trial)
            model = model_cls(**params)
            scores = cross_val_score(model, X_train, y_train, cv=inner_cv, scoring="accuracy")
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout_seconds, show_progress_bar=False)
        return study.best_params

    def evaluate(self, model_name, X, y):
        outer_cv = StratifiedKFold(
            n_splits=self.outer_folds, shuffle=True, random_state=self.random_state
        )
        fold_scores = []
        model_cls = self.model_registry[model_name]

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            best_params = self._tune_on_fold(model_name, X_train, y_train)
            model = model_cls(**best_params)
            model.fit(X_train, y_train)
            fold_accuracy = model.score(X_test, y_test)
            fold_scores.append(fold_accuracy)
            logger.info(f"  fold {fold_idx + 1}/{self.outer_folds}: accuracy={fold_accuracy:.4f}")

        fold_scores = np.array(fold_scores)
        return {
            "model_name": model_name,
            "fold_scores": fold_scores.tolist(),
            "mean_accuracy": float(fold_scores.mean()),
            "std_accuracy": float(fold_scores.std()),
        }
