import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from explainability.shap_explainer import ShapExplainer


def _toy_model():
    X, y = make_classification(
        n_samples=100, n_features=6, n_informative=4, n_classes=3,
        n_clusters_per_class=1, random_state=42,
    )
    model = LogisticRegression(max_iter=1000).fit(X, y)
    feature_names = [f"f{i}" for i in range(6)]
    return model, X, y, feature_names


def test_fit_and_explain_single():
    model, X, y, feature_names = _toy_model()
    explainer = ShapExplainer(model, feature_names)
    explainer.fit(X[:30])

    prediction = model.predict(X[0:1])[0]
    result = explainer.explain_single(X[0:1], prediction, top_n=3)

    assert len(result) == 3
    for item in result:
        assert "feature" in item and "contribution" in item
        assert item["feature"] in feature_names


def test_explain_returns_correct_shape():
    model, X, y, feature_names = _toy_model()
    explainer = ShapExplainer(model, feature_names)
    explainer.fit(X[:30])
    shap_values = explainer.explain(X[:5])

    assert shap_values.shape[0] == 5
    assert shap_values.shape[1] == len(feature_names)


def test_plot_summary_creates_file(tmp_path):
    model, X, y, feature_names = _toy_model()
    explainer = ShapExplainer(model, feature_names)
    explainer.fit(X[:30])
    explainer.explain(X[:10])

    output_path = tmp_path / "shap_summary.png"
    explainer.plot_summary(X[:10], str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
