import pandera.pandas as pa
from pandera.pandas import DataFrameSchema, Column, Check

sale_schema = DataFrameSchema(
    {
        "date": Column(
            pa.DateTime,
            nullable=False,
            coerce=True,
            description="Date as datetime64"
        ),
        "item_id": Column(
            pa.Int,
            Check.in_range(0, 23_000),
            nullable=False,
            coerce=True,
            description="Item identifier, must be >= 0 and reasonable upper bound"
        ),
        "shop_id": Column(
            pa.Int,
            Check.in_range(0, 70),
            nullable=False,
            coerce=True,
            description="Shop identifier, must be >= 0 and reasonable upper bound"
        ),
        "item_price": Column(
            pa.Float,
            nullable=True,
            coerce=True,
            checks=[
                Check.in_range(0, 400_000, error="item_price must be in [0, 400000]")
            ],
            description="Item price"
        ),
        "item_cnt_day": Column(
            pa.Float,
            nullable=True,
            coerce=True,
            checks=[
                Check.in_range(0, 3_000, error="item_cnt_day must be in [0, 3000]")
            ],
            description="Items sold"
        ),
        "date_block_num": Column(
            pa.Int,
            Check.in_range(0, 34, error="date_block_num must be in [0, 34]"),
            nullable=False,
            coerce=True,
            description="Sequential month number"
        ),
    },
    name="SaleRecordSchema",
    strict=True
)
