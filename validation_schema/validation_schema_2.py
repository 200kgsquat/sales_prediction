import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
from pandera.errors import SchemaErrors

# Define validation schema with your actual column names
features_schema = DataFrameSchema({
    "date": Column(pa.DateTime),
    "date_block_num": Column(pa.Int, checks=Check.ge(0)),
    "shop_id": Column(pa.Int),
    "item_id": Column(pa.Int),

    "item_price": Column(pa.Float, checks=Check.ge(0)),
    "item_cnt_month": Column(pa.Float, checks=Check.ge(0)),
    "log_item_cnt_day": Column(pa.Float, nullable=True),

    "item_name": Column(pa.String),
    "item_category_id": Column(pa.Int),
    "item_category_name": Column(pa.String),
    "shop_name": Column(pa.String),
    "city": Column(pa.String),

    "month": Column(pa.Int, checks=Check.in_range(1, 12)),
    "year": Column(pa.Int),
    "season": Column(pa.String),  
    "month_sin": Column(pa.Float, checks=Check.in_range(-1, 1)),
    "month_cos": Column(pa.Float, checks=Check.in_range(-1, 1)),

    "lag_1": Column(pa.Float, nullable=True),
    "lag_2": Column(pa.Float, nullable=True),
    "lag_12": Column(pa.Float, nullable=True),

    "rolling_3_median": Column(pa.Float, nullable=True),
    "rolling_6_median": Column(pa.Float, nullable=True),

    "item_avg_sales_month": Column(pa.Float),
    "shop_item_avg": Column(pa.Float),

    "price_increasing": Column(pa.Int),

    "item_min_price": Column(pa.Float),
    "item_max_price": Column(pa.Float),
    "price_range": Column(pa.Float, checks=Check.ge(0)),

    "type_of_shop": Column(pa.String, nullable=True),
})

def validate_dataset(df: pd.DataFrame, dataset_name="incoming") -> bool:
    try:
        features_schema.validate(df, lazy=True)
        print(f"[✅ SUCCESS] {dataset_name} dataset passed validation.")
        return True
    except SchemaErrors as e:
        print(f"[❌ ERROR] {dataset_name} dataset failed validation.")
        print("🔍 Validation errors:")
        print(e.failure_cases)
        return False
# 🔹 Entry point for standalone script use

if __name__ == "__main__":
    path = "datasets/fe_df.csv"

    try:
        # Load dataset
        df = pd.read_csv(path)
        # Ensure 'date' column is parsed correctly
        df["date"] = pd.to_datetime(df["date"])

        # Run validation
        is_valid = validate_dataset(df, dataset_name="future_data")

        if not is_valid:
            raise ValueError("Validation failed — pipeline halted.")

        # Proceed with feature generation or modeling
        print("🎯 Dataset is valid. Proceeding to the next stage.")

    except FileNotFoundError:
        print(f"[⚠️] File not found: {path}")
    except Exception as e:
        print(f"[❗] Unexpected error occurred: {e}")
