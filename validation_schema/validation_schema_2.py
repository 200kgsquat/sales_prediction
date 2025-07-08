import logging
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
from pandera.errors import SchemaErrors

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

features_schema = DataFrameSchema({
    "date_block_num": Column(pa.Int, checks=Check.ge(0)),
    "shop_id": Column(pa.Int, checks=Check.ge(0)),
    "item_id": Column(pa.Int, checks=Check.ge(0)),

    "item_cnt_day": Column(pa.Float, checks=Check.ge(0)),
    "item_price": Column(pa.Float, checks=Check.ge(0)),

    "item_name": Column(pa.String),
    "item_category_id": Column(pa.Int, checks=Check.ge(0)),
    "item_category_name": Column(pa.String),
    "shop_name": Column(pa.String),

    "item_cnt_month": Column(pa.Float, checks=Check.ge(0)),

    "season": Column(pa.String),
    "month_sin": Column(pa.Float, checks=Check.in_range(-1, 1)),
    "month_cos": Column(pa.Float, checks=Check.in_range(-1, 1)),

    "item_cnt_month_lag_1": Column(pa.Float, nullable=True),
    "item_cnt_month_lag_2": Column(pa.Float, nullable=True),
    "item_cnt_month_lag_3": Column(pa.Float, nullable=True),
    "item_cnt_month_lag_12": Column(pa.Float, nullable=True),

    "rolling_median_3": Column(pa.Float, nullable=True),
    "rolling_median_6": Column(pa.Float, nullable=True),

    "item_avg_sales_month_lag1": Column(pa.Float),
    "shop_item_avg_lag1": Column(pa.Float),

    "log_item_cnt_month_lag_1": Column(pa.Float, nullable=True),

    "city": Column(pa.String),
    "type_of_shop": Column(pa.String, nullable=True),
    "price_increasing": Column(pa.Int, checks=[
        Check.ge(0),
        Check.isin([0, 1])
    ]),

    "item_min_price": Column(pa.Float, checks=Check.ge(0)),
    "item_max_price": Column(pa.Float, checks=Check.ge(0)),
    "price_range": Column(pa.Float, checks=Check.ge(0)),
},
checks=[
    Check(
        lambda df: ~df.duplicated(subset=["date_block_num", "shop_id", "item_id"]),
        error="Duplicate rows for (date_block_num, shop_id, item_id)"
    )
])

def validate_dataset(df: pd.DataFrame, dataset_name="incoming") -> bool:
    try:
        features_schema.validate(df, lazy=True)
        logger.info(f"[SUCCESS] {dataset_name} dataset passed validation.")
        return True
    except SchemaErrors as e:
        logger.error(f"[ERROR] {dataset_name} dataset failed validation.")
        logger.debug("Validation errors details:")
        logger.debug(e.failure_cases)
        return False

if __name__ == "__main__":
    path = "datasets/fe_df.csv"

    try:
        logger.info(f"Loading dataset from {path}")
        df = pd.read_csv(path)
        is_valid = validate_dataset(df, dataset_name="future_data")

        if not is_valid:
            logger.critical("Validation failed — pipeline halted.")
            raise ValueError("Validation failed — pipeline halted.")

        logger.info("Dataset is valid. Proceeding to the next stage.")

    except FileNotFoundError:
        logger.error(f"File not found: {path}")
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {e}")

