import os
import lightgbm as lgb
import joblib
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SalesPredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            # Определяем абсолютный путь к папке models внутри sales_forecasting
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # sales_forecasting/
            models_dir = os.path.join(base_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'lgb_model.pkl')

        self.model_path = model_path
        self.model = None

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = 'item_cnt_month',
        date_block_col: str = 'date_block_num',
        test_block_num: int = 34,
    ):
        logging.info("Starting training...")

        # Проверяем наличие необходимых колонок
        if date_block_col not in df.columns:
            raise ValueError(f"'{date_block_col}' column not found in dataframe")
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")

        # Фильтрация данных по условию train
        df_train = df[df[date_block_col] < test_block_num].copy()

        X = df_train.drop(columns=[target_col])
        y = df_train[target_col]

        # Делим на train и validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Обработка категориальных признаков для LightGBM
        categorical_cols = X.select_dtypes(include='object').columns.tolist()
        if categorical_cols:
            logging.info(f"Encoding categorical columns as category dtype: {categorical_cols}")
            for col in categorical_cols:
                X_train[col] = X_train[col].astype('category')
                X_val[col] = X_val[col].astype('category')

        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "learning_rate": 0.1,
            "num_leaves": 64,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_data_in_leaf": 20,
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        # Обучение модели с обработкой ошибок, если early_stopping_rounds не поддерживается
        try:
            self.model = lgb.train(
                params=params,
                train_set=train_data,
                valid_sets=[train_data, val_data],
                valid_names=['train', 'val'],
                num_boost_round=300,
                early_stopping_rounds=30,
                verbose_eval=50,
            )
        except TypeError as e:
            logging.warning(f"early_stopping_rounds not supported by LightGBM version, training without it: {e}")
            self.model = lgb.train(
                params=params,
                train_set=train_data,
                num_boost_round=300,
            )

        # Создаем папку под модель, если ее нет
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        # Сохраняем модель
        joblib.dump(self.model, self.model_path)
        logging.info(f"Model trained and saved to {self.model_path}")

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            logging.info(f"Loading model from {self.model_path} for prediction")
            self.model = joblib.load(self.model_path)

        # Обработка категориальных признаков перед предсказанием
        categorical_cols = X_test.select_dtypes(include='object').columns.tolist()
        if categorical_cols:
            for col in categorical_cols:
                X_test[col] = X_test[col].astype('category')

        return self.model.predict(X_test)
