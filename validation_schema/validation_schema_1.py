import sys
import os
import pandas as pd
import pandera as pa
from pandera import DataFrameSchema, Column, Check
from pandera.errors import SchemaError
import pytest
import logging

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("validation.log", mode='a')
    ]
)
logger = logging.getLogger(__name__)

# --- 1. Schema Definition with Pandera ---
def _is_valid_date(v: str) -> bool:
    for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
        try:
            pd.to_datetime(v, format=fmt, errors='raise')
            return True
        except ValueError:
            continue
    return False

sale_schema = DataFrameSchema(
    {
        "date": Column(
            pa.String,
            Check(_is_valid_date, element_wise=True, error="Invalid date format: expected DD-MM-YYYY or YYYY-MM-DD"),
            nullable=False,
            coerce=True,
            description="Date in DD-MM-YYYY or YYYY-MM-DD format"
        ),
        "item_id": Column(
            pa.Int,
            Check(lambda x: x >= 0, error="item_id must be >= 0"),
            nullable=False,
            coerce=True,
            description="Item identifier, must be >= 0"
        ),
        "shop_id": Column(
            pa.Int,
            Check(lambda x: x >= 0, error="shop_id must be >= 0"),
            nullable=False,
            coerce=True,
            description="Shop identifier, must be >= 0"
        ),
        "item_price": Column(
            pa.Float,
            [
                Check(lambda x: x > 0, error="item_price must be > 0")
            ],
            nullable=False,
            coerce=True,
            description="Item price: >0"
        ),
        "item_cnt_day": Column(
            pa.Float,
            [
                Check(lambda x: x >= 0, error="item_cnt_day must be >= 0")
            ],
            nullable=False,
            coerce=True,
            description="Items sold: >=0"
        ),
    },
    checks=[
        Check(
            lambda df: ~df.duplicated(subset=["date", "item_id", "shop_id"]).any(),
            element_wise=False,
            error="duplicate rows for (date,item_id,shop_id)"
        )
    ],
    name="SaleRecordSchema"
)

# --- 2. Validation Function ---
def validate_dataset(df: pd.DataFrame, require_unique: bool = True) -> pd.DataFrame:
    schema_to_use = sale_schema
    if not require_unique:
        schema_to_use = sale_schema.remove_checks(
            lambda check: "duplicate rows" in check.error
        )
    try:
        logger.info("Validating dataset...")
        return schema_to_use.validate(df, lazy=True)
    except SchemaError as e:
        failure_records = e.failure_cases.to_dict(orient='records')
        logger.error("Validation failed with errors: %s", failure_records)
        raise ValueError(f"dataset validation failed: {failure_records}")

# --- 3. Main Execution & Test Handling ---
if __name__ == '__main__':
    if 'test' in sys.argv:
        sys.exit(pytest.main([__file__]))

    path = sys.argv[1] if len(sys.argv) > 1 else 'datasets/cleaned_sales.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'datasets/cleaned_sales.csv'

    logger.info("Script started. Input: %s | Output: %s", path, out)
    try:
        df = pd.read_csv(path)
        logger.info("CSV file loaded successfully.")
        df_valid = validate_dataset(df)
        df_valid.to_csv(out, index=False)
        logger.info("✅ Validated dataset saved to %s", out)
    except FileNotFoundError:
        logger.error("❌ File not found: %s", path)
    except ValueError as e:
        logger.error("❌ Validation error: %s", e)

# --- 4. Automated pytest tests ---
GOOD_PATH = 'datasets/cleaned_sales.csv'
BAD_PATH = 'datasets/broken_sales.csv'

def test_cleaned_sales_validates():
    if not os.path.exists(GOOD_PATH):
        pytest.skip(f"⚠️ File {GOOD_PATH} not found.")
    df = pd.read_csv(GOOD_PATH)
    df2 = validate_dataset(df)
    assert not df2[['date', 'item_id', 'shop_id', 'item_price', 'item_cnt_day']].isnull().any().any(), \
        "Expected no missing values in cleaned_sales.csv"

def test_broken_sales_fails():
    if not os.path.exists(BAD_PATH):
        pytest.skip(f"⚠️ File {BAD_PATH} not found.")
    df = pd.read_csv(BAD_PATH)
    with pytest.raises(ValueError):
        validate_dataset(df)