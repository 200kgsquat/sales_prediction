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
            Check(lambda s: (s >= 0).all(), error="item_id must be >= 0"),
            nullable=False,
            coerce=True,
            description="Item identifier, must be >= 0"
        ),
        "shop_id": Column(
            pa.Int,
            Check(lambda s: (s >= 0).all(), error="shop_id must be >= 0"),
            nullable=False,
            coerce=True,
            description="Shop identifier, must be >= 0"
        ),
        "item_price": Column(
            pa.Float,
            nullable=True,
            coerce=True,
            checks=[
                Check(lambda s: (s >= 0).all(), error="item_price must be >= 0"),
                Check(lambda s: (s <= 400000).all(), error="item_price must be <= 400000")
            ],
            description="Item price"
        ),
        "item_cnt_day": Column(
            pa.Float,
            nullable=True,
            coerce=True,
            checks=[
                Check(lambda s: (s >= 0).all(), error="item_cnt_day must be >= 0"),
                Check(lambda s: (s <= 3000).all(), error="item_cnt_day must be <= 3000")
            ],
            description="Items sold"
        ),
        "date_block_num": Column(
            pa.Int,
            Check(lambda s: (s >= 0).all(), error="date_block_num must be >= 0"),
            nullable=False,
            coerce=True,
            description="Sequential month number"
        ),
    },
    name="SaleRecordSchema",
    strict=True
)
