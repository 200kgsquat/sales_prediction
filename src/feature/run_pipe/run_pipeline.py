import sys
import time
import os
import pandas as pd
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from feature.fe_pipeline.fe_pipeline import FeaturePipeline
from feature.validation_schema.validation_schema_1 import validate_dataset
from feature.validation_schema.validation_schema_2 import features_schema
from feature.validator.validator import Validator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

def main(input_path: str, output_path: str):
    logger.info("Starting feature pipeline")
    start_time = time.time()

    try:
        # Load raw sales dataset
        df = pd.read_csv(input_path)
        logger.info(f"Loaded input dataset: {df.shape}")

        # First validation: raw data
        logger.info("Validating raw sales dataset")
        df_validated = validate_dataset(df)
        logger.info("Raw data validation successful")
        logger.info("Duration: %.2f seconds", time.time() - start_time)

        # Load metadata files
        items = pd.read_csv("datasets/raw/items.csv")
        item_categories = pd.read_csv("datasets/raw/item_categories.csv")
        shops = pd.read_csv("datasets/raw/shops.csv")
        logger.info("Metadata files loaded")

        # Feature engineering
        logger.info("Running feature engineering")
        fe_pipeline = FeaturePipeline(items, item_categories, shops)
        df_features = fe_pipeline.transform(df_validated)

        # Second validation: after feature engineering
        logger.info("Validating dataset after feature engineering")
        validator = Validator(schemas=[features_schema])  # Передаём схему в валидатор
        validator.validate(df_features)
        logger.info("Feature-engineered data validation successful")

        # Save the output
        df_features.to_parquet(output_path, index=False)
        logger.info("Feature engineering complete. Output saved to: %s", output_path)
        logger.info("Total pipeline duration: %.2f seconds", time.time() - start_time)

    except FileNotFoundError as fnf_err:
        logger.error("File not found: %s", fnf_err)
    except ValueError as ve:
        logger.error("Validation error: %s", ve)
    except Exception as e:
        logger.exception("Unhandled exception during pipeline execution: %s", e)


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/cleaned_sales.csv"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "datasets/features.csv"
    main(input_path, output_path)
