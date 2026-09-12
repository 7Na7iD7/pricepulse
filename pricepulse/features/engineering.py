import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self, pca_config):
        self.pca_config = pca_config
        self.pca = None

    @staticmethod
    def add_domain_features(df):
        df = df.copy()
        df["pixel_density"] = (df["px_height"] * df["px_width"]) / (
            df["sc_h"] * df["sc_w"] + 1
        )
        df["screen_area"] = df["sc_h"] * df["sc_w"]
        df["ram_per_core"] = df["ram"] / (df["n_cores"] + 1)
        df["connectivity_score"] = (
            df["four_g"] + df["three_g"] + df["wifi"] + df["blue"] + df["dual_sim"]
        )
        df["camera_score"] = df["pc"] + df["fc"]
        df["battery_efficiency"] = df["battery_power"] / (df["mobile_wt"] + 1)
        return df

    def fit_pca(self, X_train_scaled):
        from sklearn.decomposition import PCA
        if not self.pca_config.enabled:
            return X_train_scaled
        self.pca = PCA(n_components=self.pca_config.n_components)
        return self.pca.fit_transform(X_train_scaled)

    def transform_pca(self, X_scaled):
        if not self.pca_config.enabled or self.pca is None:
            return X_scaled
        return self.pca.transform(X_scaled)

    def explained_variance(self):
        if self.pca is None:
            return None
        return self.pca.explained_variance_ratio_.sum()
