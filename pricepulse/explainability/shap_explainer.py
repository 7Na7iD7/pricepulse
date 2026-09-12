import logging
import matplotlib.pyplot as plt
import numpy as np
import shap

logger = logging.getLogger("pricepulse.explainability")


class ShapExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def fit(self, X_background):
        self.explainer = shap.KernelExplainer(
            self.model.predict_proba, shap.sample(X_background, min(50, len(X_background)))
        )

    def explain(self, X_sample):
        self.shap_values = self.explainer.shap_values(X_sample, nsamples=100)
        return self.shap_values

    def plot_summary(self, X_sample, filename, class_names=None):
        values = self.shap_values
        if values.ndim == 3:
            # (n_samples, n_features, n_classes) -> میانگین قدرمطلق روی کلاس‌ها برای خلاصه‌ی سراسری
            values_for_plot = np.abs(values).mean(axis=2)
        else:
            values_for_plot = values

        shap.summary_plot(
            values_for_plot, X_sample, feature_names=self.feature_names,
            plot_type="bar", show=False,
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        logger.info(f"نمودار SHAP summary در {filename} ذخیره شد.")

    def explain_single(self, x_row, predicted_class, top_n=5):
        """
        توضیح یک پیش‌بینی تکی: تأثیر هر ویژگی روی کلاس پیش‌بینی‌شده.

        x_row: آرایه‌ی دو‌بعدی با یک ردیف (۱, تعداد ویژگی) پس از scale/pca.
        predicted_class: اندیس کلاسی که مدل پیش‌بینی کرده (برای انتخاب بردار SHAP مرتبط).
        بازمی‌گرداند: لیستی از دیکشنری {feature, contribution} مرتب‌شده بر اساس قدرمطلق تأثیر.
        """
        shap_values = self.explainer.shap_values(x_row, nsamples=100, silent=True)
        values_for_class = shap_values[0, :, predicted_class]
        order = np.argsort(-np.abs(values_for_class))[:top_n]
        return [
            {"feature": self.feature_names[i], "contribution": round(float(values_for_class[i]), 4)}
            for i in order
        ]
