import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from ensemble.combiner import EnsembleCombiner


def _toy_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=6, n_classes=4,
        n_clusters_per_class=1, random_state=42,
    )
    return X[:140], X[140:], y[:140], y[140:]


def test_weighted_voting_weights_sum_to_one():
    X_train, X_test, y_train, y_test = _toy_data()
    estimators = {
        "lr": LogisticRegression(max_iter=1000).fit(X_train, y_train),
        "svm": SVC(probability=True).fit(X_train, y_train),
    }
    scores = {"lr": 0.8, "svm": 0.6}

    combiner = EnsembleCombiner(cv_folds=3)
    voting_model, weights = combiner.build_weighted_voting(estimators, scores, X_train, y_train)

    assert abs(sum(weights.values()) - 1.0) < 1e-9
    assert weights["lr"] > weights["svm"]
    preds = voting_model.predict(X_test)
    assert len(preds) == len(y_test)


def test_stacking_builds_and_predicts():
    X_train, X_test, y_train, y_test = _toy_data()
    estimators = {
        "lr": LogisticRegression(max_iter=1000).fit(X_train, y_train),
        "svm": SVC(probability=True).fit(X_train, y_train),
    }
    combiner = EnsembleCombiner(cv_folds=3)
    stacking_model = combiner.build_stacking(estimators, X_train, y_train)
    preds = stacking_model.predict(X_test)
    assert len(preds) == len(y_test)


def test_evaluate_returns_expected_keys():
    X_train, X_test, y_train, y_test = _toy_data()
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    combiner = EnsembleCombiner(cv_folds=3)
    result = combiner.evaluate(model, X_train, y_train, X_test, y_test)

    for key in ("accuracy", "cv_mean", "cv_std", "report", "confusion_matrix", "predictions"):
        assert key in result
    assert 0.0 <= result["accuracy"] <= 1.0


def test_stacking_accepts_custom_final_estimator():
    from sklearn.tree import DecisionTreeClassifier
    X_train, X_test, y_train, y_test = _toy_data()
    estimators = {
        "lr": LogisticRegression(max_iter=1000).fit(X_train, y_train),
        "svm": SVC(probability=True).fit(X_train, y_train),
    }
    combiner = EnsembleCombiner(cv_folds=3)
    stacking_model = combiner.build_stacking(
        estimators, X_train, y_train, final_estimator=DecisionTreeClassifier(max_depth=2)
    )
    from sklearn.tree import DecisionTreeClassifier as DTC
    assert isinstance(stacking_model.final_estimator_, DTC)
    preds = stacking_model.predict(X_test)
    assert len(preds) == len(y_test)


def test_stacking_top_k_selection():
    X_train, X_test, y_train, y_test = _toy_data()
    estimators = {
        "lr": LogisticRegression(max_iter=1000).fit(X_train, y_train),
        "svm": SVC(probability=True).fit(X_train, y_train),
        "weak": LogisticRegression(max_iter=1000, C=1e-10).fit(X_train, y_train),
    }
    scores = {"lr": 0.9, "svm": 0.85, "weak": 0.3}
    combiner = EnsembleCombiner(cv_folds=3)
    stacking_model = combiner.build_stacking(estimators, X_train, y_train, best_scores=scores, top_k=2)
    used_names = [name for name, _ in stacking_model.estimators]
    assert "weak" not in used_names
    assert len(used_names) == 2
