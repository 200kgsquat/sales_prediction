import sys
import pandas as pd
import pandera as pa
from pandera import DataFrameSchema, Column, Check
from pandera.errors import SchemaError
import pytest

# --- 1. Schema Definition with Pandera ---
# Custom date check: accepts DD-MM-YYYY or YYYY-MM-DD

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
                Check(lambda x: x > 0, error="item_price must be > 0"),
                Check(lambda x: x <= 100_000, error="item_price must be <= 100000")
            ],
            nullable=False,
            coerce=True,
            description="Item price: >0, <=100k"
        ),
        "item_cnt_day": Column(
            pa.Float,
            [
                Check(lambda x: x >= 0, error="item_cnt_day must be >= 0"),
                Check(lambda x: x <= 1_000, error="item_cnt_day must be <= 1000")
            ],
            nullable=False,
            coerce=True,
            description="Items sold: >=0, <=1k"
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
    """
    Validates the DataFrame against the sale_schema.
    If require_unique=False, skips the unique-rows constraint.
    Raises ValueError on validation errors.
    """
    schema_to_use = sale_schema
    if not require_unique:
        # remove the table-level unique check
        schema_to_use = sale_schema.remove_checks(
            lambda check: "duplicate rows" in check.error
        )
    try:
        # Validate and return coerced DataFrame
        return schema_to_use.validate(df, lazy=True)
    except SchemaError as e:
        # Raise ValueError for compatibility
        raise ValueError(f"dataset validation failed: {e.failure_cases.to_dict(orient='records')}")

# --- 3. Standalone execution & Test suite ---
if __name__ == '__main__':
    if 'test' in sys.argv:
        sys.exit(pytest.main([__file__]))

    path = sys.argv[1] if len(sys.argv) > 1 else 'datasets/cleaned_sales.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'validated_sales_train.csv'

    print('Script started')
    try:
        df = pd.read_csv(path)
        df_valid = validate_dataset(df)
        df_valid.to_csv(out, index=False)
        print(f"✅ Validated dataset saved to {out}")
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
    except ValueError as e:
        print(f"❌ Validation error: {e}")

# --- 4. Automated pytest tests ---

def _create_test_data():
    return pd.DataFrame({
        "date": ["01-01-2020", "02-01-2020"],
        "item_id": [0, 1],
        "shop_id": [0, 2],
        "item_price": [10.0, 20.0],
        "item_cnt_day": [0.0, 5.0]
    })


def test_valid_dataframe():
    df = _create_test_data()
    df2 = validate_dataset(df)
    assert df2.equals(df)


def test_missing_values():
    df = _create_test_data()
    df.loc[0, "item_price"] = None
    with pytest.raises(ValueError) as exc:
        validate_dataset(df)
    assert "nullable violation" in str(exc.value) or "dataset validation failed" in str(exc.value)


def test_negative_values():
    df = _create_test_data()
    df.loc[1, "item_cnt_day"] = -1
    with pytest.raises(ValueError):
        validate_dataset(df)


def test_date_format():
    df = _create_test_data()
    df.loc[0, "date"] = "2020.01.01"
    with pytest.raises(ValueError):
        validate_dataset(df)


def test_duplicate_rows():
    df = _create_test_data()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_dataset(df)
