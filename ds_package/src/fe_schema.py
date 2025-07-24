import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

features_schema = DataFrameSchema(
    {
        "date_block_num": Column(pa.Int8, checks=Check.ge(0)),
        "shop_id": Column(pa.Int8, checks=Check.ge(0)),
        "item_id": Column(pa.Int16, checks=Check.ge(0)),

        "item_price": Column(pa.Float32, checks=Check.ge(0)),
        "item_category_id": Column(pa.Int8, checks=Check.ge(0)),

        "item_cnt_month": Column(pa.Float32, checks=Check.ge(0)),

        "month_cos": Column(pa.Float32, checks=Check.in_range(-1, 1)),

        "item_cnt_month_lag_1": Column(pa.Float32, nullable=True),
        "item_cnt_month_lag_2": Column(pa.Float32, nullable=True),

        "rolling_median_3": Column(pa.Float32, nullable=True),
        "rolling_median_6": Column(pa.Float32, nullable=True),

        "item_avg_sales_month_lag1": Column(pa.Float32),
        "shop_item_avg_lag1": Column(pa.Float32),

        "log_item_cnt_month_lag_1": Column(pa.Float32, nullable=True),

        "price_increasing": Column(pa.Bool, checks=[
            Check.ge(0),
            Check.isin([0, 1])
        ]),
        "item_price_lag_1": Column(pa.Float32, nullable=True),
        "price_change": Column(pa.Float32, nullable=True),
    },
    checks=[
        Check(
            lambda df: ~df.duplicated(subset=["date_block_num", "shop_id", "item_id"]),
            error="Duplicate rows for (date_block_num, shop_id, item_id)"
        )
    ]
)
