import sys
import pandas as pd
import time
import logging
from pathlib import Path

current_script_dir = Path(__file__).parent.absolute()
src_dir = current_script_dir.parent  
project_root = src_dir.parent        

sys.path.insert(0, str(src_dir))
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

def setup_paths():
    """Configure all required paths."""
    try:
        base_path = project_root / 'datasets/raw'
        sales_path = base_path / 'sales_train.csv'
        items_path = base_path / 'items.csv'
        categories_path = base_path / 'item_categories.csv'
        shops_path = base_path / 'shops.csv'
        output_path = project_root / 'datasets/processed/features.parquet' 
        model_path = project_root / 'models/lgb_model.pkl'
        
        # Verify paths
        for path in [sales_path, items_path, categories_path, shops_path]:
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")
        
        return sales_path, items_path, categories_path, shops_path, output_path, model_path
    except Exception as e:
        logger.error(f"Path configuration failed: {str(e)}")
        sys.exit(1)

def run_pipeline(sales_path, items_path, categories_path, shops_path, output_path, model_path) -> int:
    """Run the complete data pipeline and train the model."""
    logger.info("Starting pipeline")
    start_time = time.time()
    
    try:
        # 1. ETL Process
        logger.info("Running ETL pipeline")
        from sales_forecasting.data.etl_pipeline import ETLPipeline
        etl = ETLPipeline(
            sales_path=str(sales_path),
            items_path=str(items_path),
            categories_path=str(categories_path),
            shops_path=str(shops_path)
        )
        etl.run()
        df_cleaned = etl.df_transformed
        logger.info(f"ETL completed. Shape: {df_cleaned.shape}")
        
        # 2. Validate cleaned data
        logger.info("Validating cleaned sales data")
        from sales_forecasting.validation.validator import Validator
        from sales_forecasting.validation.validation_schemas.validation_schema_1 import sale_schema

        validator = Validator()
        cleaned_sales_validated = validator.validate(df_cleaned, sale_schema, "CleanedSalesData")
        logger.info(f"Cleaned data validated. Shape: {cleaned_sales_validated.shape}")
        
        # 3. Feature Engineering
        logger.info("Running feature engineering")
        from sales_forecasting.feature.fe_pipeline import FeaturePipeline
        fe_pipeline = FeaturePipeline(etl.items, etl.categories, etl.shops)
        df_features = fe_pipeline.transform(cleaned_sales_validated)
        logger.info(f"Feature engineering completed. Shape: {df_features.shape}")
        
        # 4. Validate final features
        logger.info("Validating final features")
        from sales_forecasting.validation.validation_schemas.validation_schema_2 import features_schema

        df_features_validated = validator.validate(df_features, features_schema, "FinalFeatures")
        logger.info(f"Final features validated. Shape: {df_features_validated.shape}")
        
        # 5. Save features
        logger.info(f"Saving features to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_features_validated.to_parquet(output_path, index=False)
        
        # 6. Train model
        logger.info("Train model")
        from sales_forecasting.modeling.trainer import SalesPredictor

        trainer = SalesPredictor(model_path=str(model_path))
        trainer.train(df_features_validated)


    except Exception as e:
        logger.exception(f"Error in pipeline: {str(e)}")
        return 1

if __name__ == "__main__":
    paths = setup_paths()
    sys.exit(run_pipeline(*paths))
