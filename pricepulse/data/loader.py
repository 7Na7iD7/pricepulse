import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoadError(Exception):
    pass


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()

    def load_raw(self):
        path = self.config.csv_path
        if not os.path.exists(path):
            raise DataLoadError(f"فایل دیتاست در مسیر '{path}' یافت نشد.")
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            raise DataLoadError(f"خطا در خواندن فایل CSV: {exc}")
        if self.config.target_column not in df.columns:
            raise DataLoadError(f"ستون هدف '{self.config.target_column}' وجود ندارد.")
        return df

    def split_and_scale(self):
        df = self.load_raw()
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=self.config.random_state,
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train.values,
            "y_test": y_test.values,
            "feature_names": list(X.columns),
            "scaler": self.scaler,
        }
