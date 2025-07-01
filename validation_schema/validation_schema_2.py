import sys
import pandas as pd
import numpy as np
import pandera as pa
from pandera import DataFrameSchema, Column, Check
from pandera.errors import SchemaError

# --- 1. Validation Function with Pandera after Feature Engineering ---
def validate_after_fe(
    df: pd.DataFrame,
    required_cols=None,
    max_nan_ratio: float = 0.01
) -> pd.DataFrame:
    """
    Validates a DataFrame after feature engineering using Pandera.

    - Ensures required columns exist and contain no NaNs.
    - Allows up to `max_nan_ratio` NaNs in non-required columns.
    - Checks for duplicates on key columns.
    - Verifies integer types for key columns and numeric + positive for prices.
    - Flags infinite values in numeric columns.

    Raises:
        ValueError: on any validation failure, with details of failure cases.
    Returns:
        The validated (and coerced) DataFrame.
    """
    # Default required columns
    if required_cols is None:
        required_cols = ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day', 'item_price']

    # 1. Pre-check for missing required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 2. Build schema columns dynamically
    col_defs = {}
    for col in df.columns:
        # Determine base dtype
        if col in ['date_block_num', 'shop_id', 'item_id']:
            dtype = pa.Int
            nullable = False
            checks = [Check(lambda s: s.dropna().apply(float.is_integer).all(),
                            element_wise=False,
                            error=f"Column '{col}' must contain only integers")]  # series-level
        elif col == 'item_cnt_day':
            dtype = pa.Float
            nullable = False
            checks = [
                Check(lambda s: s >= 0, element_wise=True, error="item_cnt_day must be >= 0")
            ]
        elif col == 'item_price':
            dtype = pa.Float
            nullable = False
            checks = [
                Check(lambda s: s > 0, element_wise=True, error="item_price must be > 0")
            ]
        else:
            # For other columns, infer dtype and allow NaNs up to ratio
            if pd.api.types.is_integer_dtype(df[col].dropna()):
                dtype = pa.Int
            elif pd.api.types.is_float_dtype(df[col].dropna()):
                dtype = pa.Float
            else:
                dtype = pa.String
            nullable = True
            checks = [
                Check(
                    lambda s: s.isna().mean() <= max_nan_ratio,
                    element_wise=False,
                    error=f"Column '{col}' has too many missing values (> {max_nan_ratio * 100:.2f}% NaNs)"
                )
            ]
        col_defs[col] = Column(
            dtype,
            checks=checks,
            nullable=nullable,
            coerce=True
        )

    # 3. Table-level checks
    table_checks = []
    # Duplicates check
    table_checks.append(
        Check(
            lambda df: ~df.duplicated(subset=['date_block_num', 'shop_id', 'item_id']).any(),
            element_wise=False,
            error="duplicate rows for (date_block_num, shop_id, item_id)"
        )
    )
    # Infinite values check
    table_checks.append(
        Check(
            lambda df: not np.isinf(df.select_dtypes(include=[np.number]).values).any(),
            element_wise=False,
            error="infinite values detected in numeric columns"
        )
    )

    schema = DataFrameSchema(
        columns=col_defs,
        checks=table_checks,
        name="AfterFEValidation"
    )

    # 4. Validate
    try:
        validated = schema.validate(df, lazy=True)
        print("Validation passed!")
        return validated
    except SchemaError as e:
        # Collect failure cases
        cases = e.failure_cases.to_dict(orient='records')
        raise ValueError(f"Validation errors: {cases}")

# --- CLI Interface ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_fe.py path_to_dataset.csv [max_nan_ratio]")
        sys.exit(1)

    path = sys.argv[1]
    max_nan = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01

    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(1)

    try:
        validate_after_fe(data, max_nan_ratio=max_nan)
    except ValueError as e:
        print(e)
        sys.exit(1)
