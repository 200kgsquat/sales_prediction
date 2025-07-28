import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

from pathlib import Path
from typing import Optional

from sales_forecasting.utils.logger import get_logger

logger = get_logger(__name__)


class SalesPredictor:
    def __init__(self, model_path: Optional[str] = None, features_path: Optional[str] = None):
        current_file = Path(__file__).resolve()
        base_dir = current_file.parent.parent

        models_dir = base_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = str(model_path or models_dir / 'lgb_model.pkl')
        self.features_path = str(features_path or models_dir / 'feature_columns.pkl')

        self.model = None
        self.feature_columns = None
        self.enable_wandb = False

        # Try to import wandb
        try:
            import wandb
            self.wandb = wandb
            self.enable_wandb = True
        except ImportError:
            self.wandb = None
            logger.warning("Weights & Biases (wandb) not installed. Proceeding without experiment tracking.")

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = 'item_cnt_month',
        date_block_col: str = 'date_block_num',
        test_block_num: int = 34,
        wandb_project: str = "sales_forecasting",
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        logger.info("Starting training...")

        if date_block_col not in df.columns:
            raise ValueError(f"'{date_block_col}' not found in dataframe")
        if target_col not in df.columns:
            raise ValueError(f"'{target_col}' not found in dataframe")

        if self.enable_wandb:
            try:
                if getattr(self.wandb, "run", None) is not None:
                    self.wandb.finish()

                init_params = {"project": wandb_project}
                if wandb_entity:
                    init_params["entity"] = wandb_entity
                if wandb_run_name:
                    init_params["name"] = wandb_run_name
                init_params["config"] = {
                    "target_col": target_col,
                    "date_block_col": date_block_col,
                    "test_block_num": test_block_num,
                    "model_type": "LightGBM",
                }

                self.wandb.init(**init_params)
                wandb_config = init_params["config"]
            except Exception as e:
                logger.warning(f"W&B init failed ({e}), switching to offline mode.")
                self.enable_wandb = False
                self.wandb = None
        else:
            wandb_config = {}

        df_train = df[df[date_block_col] < test_block_num].copy()
        df_val = df[df[date_block_col] == (test_block_num - 1)].copy()

        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]
        X_val = df_val.drop(columns=[target_col])
        y_val = df_val[target_col]

        categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
        for col in categorical_cols:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')

        self.feature_columns = X_train.columns.tolist()
        joblib.dump(self.feature_columns, self.features_path)

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

        if self.enable_wandb:
            self.wandb.config.update(params)

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        def wandb_callback(env):
            if self.enable_wandb and env.evaluation_result_list:
                metrics = {f"{data}_{metric}": val for data, metric, val, _ in env.evaluation_result_list}
                metrics["iteration"] = env.iteration
                self.wandb.log(metrics)

        callbacks = [lgb.early_stopping(stopping_rounds=30)]
        if self.enable_wandb:
            callbacks.append(wandb_callback)

        self.model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=callbacks,
        )

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

        if self.enable_wandb:
            self.wandb.save(self.model_path)
            self.wandb.save(self.features_path)

            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importance()
            }).sort_values(by='importance', ascending=False)

            self.wandb.log({"feature_importance": self.wandb.Table(dataframe=importance_df)})
            self.wandb.finish()

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            logger.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)

        if self.feature_columns is None:
            logger.info(f"Loading feature columns from {self.features_path}")
            self.feature_columns = joblib.load(self.features_path)

        X_test = X_test.reindex(columns=self.feature_columns)

        categorical_cols = X_test.select_dtypes(include='object').columns.tolist()
        for col in categorical_cols:
            X_test[col] = X_test[col].astype('category')

        return self.model.predict(X_test)
