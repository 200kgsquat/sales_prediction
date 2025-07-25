import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sales_forecasting.modeling.trainer import SalesPredictor
from sales_forecasting.utils.logger import get_logger

logger = get_logger(__name__)

current_script_dir = Path(__file__).parent.absolute()
project_root = current_script_dir.parent.parent  # .../sales_predict/src

sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


def run_inference_pipeline() -> pd.DataFrame:
    logger.info("Starting inference pipeline")
    try:
        paths = {
            "features_path": project_root / "datasets/processed/features.parquet",
            "test_features_path": project_root / "notebooks/test.parquet",
            "test_ids_path": project_root / "datasets/external/test.csv",
            "model_path": project_root / "models/lgb_model.pkl",
            "submission_path": project_root / "datasets/external/predictions.csv",
        }

        logger.info(f"Loading full features from {paths['features_path']}")
        df_all_features = pd.read_parquet(paths["features_path"])
        logger.info(f"Loaded features shape: {df_all_features.shape}")

        logger.info(f"Loading test features from {paths['test_features_path']}")
        test_df = pd.read_parquet(paths["test_features_path"])
        logger.info(f"Test features shape: {test_df.shape}")

        if "target" in test_df.columns:
            logger.info("Dropping target column from test features")
            test_df.drop(columns=["target"], inplace=True)

        logger.info(f"Loading model from {paths['model_path']}")
        model = SalesPredictor(model_path=str(paths["model_path"]))
        y_pred = model.predict(test_df)

        df = pd.DataFrame({
            "ID": np.arange(len(y_pred)),
            "item_cnt_month": y_pred.clip(0, 20)
        })

        logger.info(f"Saving predictions to {paths['submission_path']}")
        df.to_csv(paths["submission_path"], index=False)

        logger.info("Inference pipeline completed successfully")
        return df

    except Exception as e:
        logger.exception(f"Inference pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_inference_pipeline()
