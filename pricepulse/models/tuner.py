import logging
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

from models.base_models import get_model_registry, get_search_spaces, get_fixed_params, get_param_transforms

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger("pricepulse.tuner")


class OptunaTuner:
    def __init__(self, tuning_config):
        self.config = tuning_config
        self.model_registry = get_model_registry()
        self.search_spaces = get_search_spaces()
        self.fixed_params = get_fixed_params()
        self.param_transforms = get_param_transforms()
        self.best_estimators = {}
        self.best_scores = {}
        self.best_params = {}

    def _objective(self, trial, model_name, X, y, cv):
        model_cls = self.model_registry[model_name]
        params = self.search_spaces[model_name](trial)
        model = model_cls(**params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        return scores.mean()

    def tune_all(self, X_train, y_train, per_model_trials=None):
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds, shuffle=True, random_state=self.config.random_state
        )

        for model_name in self.model_registry:
            n_trials = (per_model_trials or {}).get(model_name, self.config.n_trials)
            logger.info(f"شروع تیونینگ مدل: {model_name} (n_trials={n_trials})")
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial: self._objective(trial, model_name, X_train, y_train, cv),
                n_trials=n_trials,
                timeout=self.config.timeout_seconds,
                show_progress_bar=False,
            )

            best_params = dict(study.best_params)
            for param_name, transform_fn in self.param_transforms.get(model_name, {}).items():
                if param_name in best_params:
                    best_params[param_name] = transform_fn(best_params[param_name])
            best_params.update(self.fixed_params.get(model_name, {}))

            model_cls = self.model_registry[model_name]
            best_model = model_cls(**best_params)
            best_model.fit(X_train, y_train)

            self.best_estimators[model_name] = best_model
            self.best_scores[model_name] = study.best_value
            self.best_params[model_name] = best_params

            logger.info(f"{model_name}: بهترین CV accuracy = {study.best_value * 100:.2f}%")

        return self.best_estimators, self.best_scores, self.best_params
