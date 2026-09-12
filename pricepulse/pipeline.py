import logging
import sys

import matplotlib.pyplot as plt
import seaborn as sns

from config import AppConfig
from data.loader import DataLoader, DataLoadError
from features.engineering import FeatureEngineer
from models.tuner import OptunaTuner
from ensemble.combiner import EnsembleCombiner
from registry.model_registry import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pricepulse.pipeline")
logging.getLogger("shap").setLevel(logging.WARNING)


def plot_comparison(results, filename, title):
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] * 100 for n in names]
    cv_means = [results[n]["cv_mean"] * 100 for n in names]
    cv_stds = [results[n]["cv_std"] * 100 for n in names]

    x = range(len(names))
    plt.figure(figsize=(11, 6))
    plt.bar(x, accuracies, width=0.4, label="Test Accuracy", color="#4C72B0")
    plt.errorbar(x, cv_means, yerr=cv_stds, fmt="o", color="#DD8452", label="CV Accuracy")
    plt.xticks(list(x), names, rotation=20)
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.ylim(0, 100)
    plt.legend()
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 1, f"{acc:.2f}%", ha="center")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    logger.info(f"نمودار در {filename} ذخیره شد.")


def run():
    logger.info("=== شروع پایپ‌لاین PricePulse ===")
    config = AppConfig()

    try:
        loader = DataLoader(config.data)
        data = loader.split_and_scale()
        logger.info(f"داده بارگذاری شد. ویژگی‌ها: {len(data['feature_names'])}")
    except DataLoadError as exc:
        logger.error(str(exc))
        sys.exit(1)

    engineer = FeatureEngineer(config.pca)
    X_train_final = engineer.fit_pca(data["X_train"])
    X_test_final = engineer.transform_pca(data["X_test"])
    if config.pca.enabled:
        logger.info(f"واریانس حفظ‌شده پس از PCA: {engineer.explained_variance() * 100:.2f}%")

    logger.info("=== مرحله تیونینگ هوشمند با Optuna ===")
    tuner = OptunaTuner(config.tuning)
    best_estimators, best_scores, best_params = tuner.tune_all(
        X_train_final, data["y_train"], per_model_trials=config.tuning.per_model_trials
    )

    combiner = EnsembleCombiner(cv_folds=config.tuning.cv_folds, stacking_cv=3)
    ensemble_top_k = min(6, len(best_estimators))

    logger.info(f"=== ساخت Weighted Soft Voting (top-{ensemble_top_k} مدل) ===")
    voting_model, weights = combiner.build_weighted_voting(
        best_estimators, best_scores, X_train_final, data["y_train"], top_k=ensemble_top_k
    )
    logger.info(f"وزن‌ها: { {k: round(v, 3) for k, v in weights.items()} }")

    logger.info(f"=== ساخت Stacking Ensemble (top-{ensemble_top_k} مدل، مقایسه‌ی meta-learner) ===")
    stacking_lr = combiner.build_stacking(
        best_estimators, X_train_final, data["y_train"],
        best_scores=best_scores, top_k=ensemble_top_k,
    )
    from xgboost import XGBClassifier
    stacking_xgb = combiner.build_stacking(
        best_estimators, X_train_final, data["y_train"],
        best_scores=best_scores, top_k=ensemble_top_k,
        final_estimator=XGBClassifier(n_estimators=100, max_depth=3, random_state=42, eval_metric="mlogloss"),
    )
    eval_lr = combiner.evaluate(stacking_lr, X_train_final, data["y_train"], X_test_final, data["y_test"])
    eval_xgb = combiner.evaluate(stacking_xgb, X_train_final, data["y_train"], X_test_final, data["y_test"])
    logger.info(f"Stacking (meta=LogReg): CV={eval_lr['cv_mean']*100:.2f}% | Stacking (meta=XGBoost): CV={eval_xgb['cv_mean']*100:.2f}%")
    if eval_lr["cv_mean"] >= eval_xgb["cv_mean"]:
        stacking_model = stacking_lr
        logger.info("meta-learner انتخاب‌شده: Logistic Regression")
    else:
        stacking_model = stacking_xgb
        logger.info("meta-learner انتخاب‌شده: XGBoost")

    results = {}
    for name, model in best_estimators.items():
        results[name] = combiner.evaluate(
            model, X_train_final, data["y_train"], X_test_final, data["y_test"]
        )
    results["weighted_voting"] = combiner.evaluate(
        voting_model, X_train_final, data["y_train"], X_test_final, data["y_test"]
    )
    results["stacking"] = combiner.evaluate(
        stacking_model, X_train_final, data["y_train"], X_test_final, data["y_test"]
    )

    for name, r in results.items():
        logger.info(
            f"{name}: تست={r['accuracy']*100:.2f}% | CV={r['cv_mean']*100:.2f}%±{r['cv_std']*100:.2f}%"
        )

    plot_comparison(results, "pricepulse_comparison.png", "PricePulse — Model Comparison")

    best_name = max(results.items(), key=lambda kv: kv[1]["accuracy"])[0]
    best_model = (
        voting_model if best_name == "weighted_voting"
        else stacking_model if best_name == "stacking"
        else best_estimators[best_name]
    )
    logger.info(f"بهترین مدل نهایی: {best_name} با دقت {results[best_name]['accuracy']*100:.2f}%")

    registry = ModelRegistry(config.registry)
    registry.save(
        model=best_model,
        scaler=data["scaler"],
        pca=engineer.pca,
        feature_names=data["feature_names"],
        metrics=results[best_name],
        extra_metadata={"best_model_name": best_name, "ensemble_weights": weights},
    )

    if getattr(config, "explainability", None) and config.explainability.enabled:
        logger.info("=== تولید نمودار توضیح‌پذیری SHAP ===")
        try:
            from explainability.shap_explainer import ShapExplainer

            sample_size = config.explainability.background_sample_size
            explainer = ShapExplainer(best_model, data["feature_names"])
            explainer.fit(X_train_final[:sample_size])
            explain_sample = X_test_final[:config.explainability.explain_sample_size]
            explainer.explain(explain_sample)
            explainer.plot_summary(explain_sample, "pricepulse_shap_summary.png")
            logger.info("نمودار SHAP summary با موفقیت تولید شد.")
        except Exception as exc:
            logger.warning(f"تولید نمودار SHAP ناموفق بود (این خطا پایپ‌لاین را متوقف نمی‌کند): {exc}")

    logger.info("=== پایپ‌لاین با موفقیت به پایان رسید ===")


if __name__ == "__main__":
    run()
