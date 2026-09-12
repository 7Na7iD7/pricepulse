from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_model_registry():
    return {
        "logistic_regression": LogisticRegression,
        "knn": KNeighborsClassifier,
        "naive_bayes": GaussianNB,
        "svm": SVC,
        "random_forest": RandomForestClassifier,
        "gradient_boosting": GradientBoostingClassifier,
        "xgboost": XGBClassifier,
        "lightgbm": LGBMClassifier,
        "catboost": CatBoostClassifier,
        "mlp": MLPClassifier,
    }


def get_fixed_params():
    return {
        "logistic_regression": {"max_iter": 2000},
        "knn": {},
        "naive_bayes": {},
        "svm": {"probability": True},
        "random_forest": {"random_state": 42},
        "gradient_boosting": {"random_state": 42},
        "xgboost": {"random_state": 42, "eval_metric": "mlogloss"},
        "lightgbm": {"random_state": 42, "verbose": -1},
        "catboost": {"random_state": 42, "verbose": False},
        "mlp": {"max_iter": 1000, "early_stopping": True, "random_state": 42},
    }


def get_search_spaces():
    return {
        "logistic_regression": lambda trial: {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "max_iter": 2000,
        },
        "knn": lambda trial: {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 25),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "p": trial.suggest_int("p", 1, 2),
        },
        "naive_bayes": lambda trial: {
            "var_smoothing": trial.suggest_float("var_smoothing", 1e-10, 1e-6, log=True),
        },
        "svm": lambda trial: {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "probability": True,
        },
        "random_forest": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "random_state": 42,
        },
        "gradient_boosting": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": 42,
        },
        "xgboost": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "random_state": 42,
            "eval_metric": "mlogloss",
        },
        "lightgbm": lambda trial: {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 12),
            "num_leaves": trial.suggest_int("num_leaves", 7, 127),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42,
            "verbose": -1,
        },
        "catboost": lambda trial: {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "depth": trial.suggest_int("depth", 3, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "random_state": 42,
            "verbose": False,
        },
        "mlp": lambda trial: {
            "hidden_layer_sizes": {
                "small": (64,), "medium": (128,), "deep_small": (64, 32),
                "deep_medium": (128, 64), "deep_large": (128, 64, 32),
            }[trial.suggest_categorical(
                "hidden_layer_sizes", ["small", "medium", "deep_small", "deep_medium", "deep_large"]
            )],
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            "max_iter": 1000,
            "early_stopping": True,
            "random_state": 42,
        },
    }


MLP_HIDDEN_LAYER_MAP = {
    "small": (64,), "medium": (128,), "deep_small": (64, 32),
    "deep_medium": (128, 64), "deep_large": (128, 64, 32),
}


def get_param_transforms():
    return {
        "mlp": {"hidden_layer_sizes": lambda label: MLP_HIDDEN_LAYER_MAP[label]},
    }
