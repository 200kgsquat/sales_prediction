import sys
import pandas as pd
import logging
from pathlib import Path
from sales_forecasting.modeling.trainer import SalesPredictor

current_script_dir = Path(__file__).parent.absolute()
project_root = current_script_dir.parent.parent  # Это путь до .../sales_predict/src

sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("inference.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

def run_inference_pipeline() -> pd.DataFrame:
    logger.info("Starting inference pipeline")
    try:
        # Обновленные пути: datasets лежат на уровень выше src (в sales_predict), model — в src/models
        paths = {
            "features_path": project_root.parent / "datasets/processed/features.parquet",
            "test_ids_path": project_root.parent / "datasets/external/test.csv",
            "model_path": project_root / "models/lgb_model.pkl",
            "submission_path": project_root.parent / "datasets/external/predictions.parquet",
        }

        # Load features
        logger.info(f"Loading features from {paths['features_path']}")
        df_all_features = pd.read_parquet(paths["features_path"])
        test_df = df_all_features[df_all_features["date_block_num"] == 34].copy()

        if "target" in test_df.columns:
            logger.info("Dropping target column from test set")
            test_df.drop(columns=["target"], inplace=True)

        # Load test IDs
        logger.info(f"Loading test IDs from {paths['test_ids_path']}")
        test_ids = pd.read_csv(paths["test_ids_path"])

        # Predict
        logger.info(f"Loading model from {paths['model_path']}")
        model = SalesPredictor(model_path=str(paths["model_path"]))
        y_pred = model.predict(test_df)

        # Format submission
        submission = test_ids.copy()
        submission["item_cnt_month"] = y_pred.clip(0, 20)

        # Save
        logger.info(f"Saving predictions to {paths['submission_path']}")
        submission.to_parquet(paths["submission_path"], index=False)

        logger.info("Inference pipeline completed successfully")
        return submission

    except Exception as e:
        logger.exception(f"Inference pipeline failed: {e}")
        sys.exit(1)
