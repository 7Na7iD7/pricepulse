import numpy as np
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class EnsembleCombiner:
    def __init__(self, cv_folds=5, stacking_cv=3, random_state=42):
        self.cv_folds = stacking_cv
        self.eval_cv_folds = cv_folds
        self.random_state = random_state

    def build_weighted_voting(self, best_estimators, best_scores, X_train, y_train, top_k=None):
        names = list(best_estimators.keys())
        if top_k is not None and top_k < len(names):
            names = sorted(names, key=lambda n: best_scores[n], reverse=True)[:top_k]

        weights = np.array([best_scores[name] for name in names])
        weights = weights / weights.sum()

        voting = VotingClassifier(
            estimators=[(name, best_estimators[name]) for name in names],
            voting="soft",
            weights=weights.tolist(),
        )
        voting.fit(X_train, y_train)
        return voting, dict(zip(names, weights))

    def build_stacking(self, best_estimators, X_train, y_train, best_scores=None, top_k=None, final_estimator=None):
        estimators_dict = best_estimators
        if top_k is not None and best_scores is not None and top_k < len(best_estimators):
            top_names = sorted(best_estimators.keys(), key=lambda n: best_scores[n], reverse=True)[:top_k]
            estimators_dict = {name: best_estimators[name] for name in top_names}

        stacking = StackingClassifier(
            estimators=[(name, model) for name, model in estimators_dict.items()],
            final_estimator=final_estimator or LogisticRegression(max_iter=2000),
            cv=self.cv_folds,
            stack_method="predict_proba",
            n_jobs=-1,
        )
        stacking.fit(X_train, y_train)
        return stacking

    def evaluate(self, model, X_train, y_train, X_test, y_test):
        skf = StratifiedKFold(
            n_splits=self.eval_cv_folds, shuffle=True, random_state=self.random_state
        )
        cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="accuracy")
        predictions = model.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, predictions),
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "report": classification_report(y_test, predictions, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, predictions),
            "predictions": predictions,
        }
