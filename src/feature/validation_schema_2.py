import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

features_schema = DataFrameSchema(
    {
        "date_block_num": Column(pa.Int8, checks=Check.ge(0)),
        "shop_id": Column(pa.Int8, checks=Check.ge(0)),
        "item_id": Column(pa.Int16, checks=Check.ge(0)),

        "item_price": Column(pa.Float32, checks=Check.ge(0)),

        "item_name": Column(pa.String),
        "item_category_id": Column(pa.Int8, checks=Check.ge(0)),
        "item_category_name": Column(pa.String),
        "shop_name": Column(pa.String),

        "item_cnt_month": Column(pa.Float32, checks=Check.ge(0)),

        "season": Column(pa.String),
        "month_sin": Column(pa.Float32, checks=Check.in_range(-1, 1)),
        "month_cos": Column(pa.Float32, checks=Check.in_range(-1, 1)),

        "item_cnt_month_lag_1": Column(pa.Float32, nullable=True),
        "item_cnt_month_lag_2": Column(pa.Float32, nullable=True),
        "item_cnt_month_lag_3": Column(pa.Float32, nullable=True),
        "item_cnt_month_lag_12": Column(pa.Float32, nullable=True),

        "rolling_median_3": Column(pa.Float32, nullable=True),
        "rolling_median_6": Column(pa.Float32, nullable=True),

        "item_avg_sales_month_lag1": Column(pa.Float32),
        "shop_item_avg_lag1": Column(pa.Float32),

        "log_item_cnt_month_lag_1": Column(pa.Float32, nullable=True),

        "city": Column(pa.String),
        "type_of_shop": Column(pa.String, nullable=True),

        "price_increasing": Column(pa.Bool, checks=[
            Check.ge(0),
            Check.isin([0, 1])
        ]),

        "item_min_price": Column(pa.Float32, checks=Check.ge(0)),
        "item_max_price": Column(pa.Float32, checks=Check.ge(0)),
        "price_range": Column(pa.Float32, checks=Check.ge(0)),
    },
    checks=[
        Check(
            lambda df: ~df.duplicated(subset=["date_block_num", "shop_id", "item_id"]),
            error="Duplicate rows for (date_block_num, shop_id, item_id)"
        )
    ]
)
