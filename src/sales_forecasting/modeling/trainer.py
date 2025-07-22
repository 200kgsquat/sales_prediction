import os
import joblib
import logging
import numpy as np  # Ensure numpy is imported for type hints
import pandas as pd
import lightgbm as lgb
import wandb
from wandb.errors import CommError


class SalesPredictor:
    def __init__(self, model_path=None, features_path=None):
        # Determine base directory and set default file paths
        current_dir = os.path.abspath(__file__)
        base_dir = os.path.dirname(os.path.dirname(current_dir))

        if model_path is None:
            models_dir = os.path.join(base_dir, 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, 'lgb_model.pkl')

        if features_path is None:
            features_dir = os.path.join(base_dir, 'models')
            os.makedirs(features_dir, exist_ok=True)
            features_path = os.path.join(features_dir, 'feature_columns.pkl')

        self.model_path = model_path
        self.features_path = features_path
        self.model = None
        self.feature_columns = None

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = 'item_cnt_month',
        date_block_col: str = 'date_block_num',
        test_block_num: int = 34,
        wandb_project: str = "sales_forecasting",
        wandb_entity: str = None,
        wandb_run_name: str = None,
    ):
        logging.info("Starting training...")

        # Finish existing run to avoid overlap
        if wandb.run is not None:
            wandb.finish()

        # Initialize W&B run, specifying entity if provided
        try:
            init_params = {"project": wandb_project}
            if wandb_entity:
                init_params['entity'] = wandb_entity
            if wandb_run_name:
                init_params['name'] = wandb_run_name
            init_params['config'] = {
                "target_col": target_col,
                "date_block_col": date_block_col,
                "test_block_num": test_block_num,
                "model_type": "LightGBM",
            }
            wandb.init(**init_params)
        except CommError as e:
            logging.warning(f"W&B init failed ({e}), switching to offline mode.")
            wandb.init(mode="offline", **init_params)

        # Validate input columns
        if date_block_col not in df.columns:
            raise ValueError(f"'{date_block_col}' not found in dataframe")
        if target_col not in df.columns:
            raise ValueError(f"'{target_col}' not found in dataframe")

        # Time-based split for training and validation
        df_train = df[df[date_block_col] < test_block_num].copy()
        df_val = df[df[date_block_col] == (test_block_num - 1)].copy()

        # Separate features and target
        X_train = df_train.drop(columns=[target_col])
        y_train = df_train[target_col]
        X_val = df_val.drop(columns=[target_col])
        y_val = df_val[target_col]

        # Convert object dtypes to category for LightGBM
        categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
        for col in categorical_cols:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')

        # Save feature column order
        self.feature_columns = X_train.columns.tolist()
        joblib.dump(self.feature_columns, self.features_path)

        # Define LightGBM hyperparameters
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

        # Log hyperparameters to W&B
        wandb.config.update(params)

        # Prepare LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)

        # Define W&B callback for logging metrics
        def wandb_callback(env):
            if env.evaluation_result_list:
                metrics = {f"{data}_{metric}": val for data, metric, val, _ in env.evaluation_result_list}
                metrics["iteration"] = env.iteration
                wandb.log(metrics)

        # Train with early stopping and W&B callback
        self.model = lgb.train(
            params=params,
            train_set=train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(stopping_rounds=30), wandb_callback],
        )

        # Save model artifact
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logging.info(f"Model saved to {self.model_path}")

        # Save feature columns artifact
        wandb.save(self.model_path)
        wandb.save(self.features_path)

        # Log feature importance table
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importance()
        }).sort_values(by='importance', ascending=False)
        wandb.log({"feature_importance": wandb.Table(dataframe=importance_df)})

        # Finish the W&B run
        wandb.finish()

    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        # Load model if not already loaded
        if self.model is None:
            logging.info(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)

        # Load feature columns if not already loaded
        if self.feature_columns is None:
            logging.info(f"Loading feature columns from {self.features_path}")
            self.feature_columns = joblib.load(self.features_path)

        # Align DataFrame columns
        X_test = X_test.reindex(columns=self.feature_columns)

        # Convert object types to category
        categorical_cols = X_test.select_dtypes(include='object').columns.tolist()
        for col in categorical_cols:
            X_test[col] = X_test[col].astype('category')

        # Predict and return
        return self.model.predict(X_test)
