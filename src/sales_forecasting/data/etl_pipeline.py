from pathlib import Path
from typing import Optional
import pandas as pd
from sales_forecasting.utils.logger import get_logger

logger = get_logger(__name__)


class ETLPipeline:
    def __init__(
        self,
        sales_path: str,
        items_path: str,
        categories_path: str,
        shops_path: str,
        output_path: Optional[str] = None,
    ):
        self.sales_path = Path(sales_path)
        self.items_path = Path(items_path)
        self.categories_path = Path(categories_path)
        self.shops_path = Path(shops_path)
        self.output_path = Path(output_path) if output_path else None

        self.sales = None
        self.items = None
        self.categories = None
        self.shops = None
        self.df_transformed = None

        logger.info("ETLPipeline initialized")

    def _load_data(self):
        logger.info("Loading input data")
        for path in [self.sales_path, self.items_path, self.categories_path, self.shops_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing input file: {path}")

        self.sales = pd.read_csv(self.sales_path)
        self.items = pd.read_csv(self.items_path)
        self.categories = pd.read_csv(self.categories_path)
        self.shops = pd.read_csv(self.shops_path)

        initial_len = len(self.sales)
        self.sales = self.sales[
            (self.sales['item_price'] >= 0) &
            (self.sales['item_cnt_day'] >= 0)
        ]
        logger.info(f"Sales loaded: {initial_len} rows -> {len(self.sales)} after filtering negatives")

    def _validate_and_clean_sales(self):
        logger.info("Validating and cleaning sales data")

        self.sales['date'] = pd.to_datetime(self.sales['date'], dayfirst=True, errors='coerce')
        if self.sales['date'].isna().any():
            bad_rows = self.sales[self.sales['date'].isna()]
            logger.error(f"Invalid dates found:\n{bad_rows[['date']].head()}")
            raise ValueError(f"Invalid dates: {len(bad_rows)} rows")

        for col in ['item_id', 'shop_id', 'date_block_num']:
            self.sales[col] = pd.to_numeric(self.sales[col], errors='coerce', downcast='integer')
            if self.sales[col].isna().any():
                bad_rows = self.sales[self.sales[col].isna()]
                logger.error(f"Invalid {col} values:\n{bad_rows[[col]].head()}")
                raise ValueError(f"Invalid values in {col}")

        before = len(self.sales)
        self.sales.drop_duplicates(subset=['date', 'item_id', 'shop_id'], inplace=True)
        logger.info(f"Dropped {before - len(self.sales)} duplicate rows")

        before = len(self.sales)
        self.sales = self.sales[
            (self.sales['item_price'] >= 0) &
            (self.sales['item_cnt_day'] >= 0)
        ]
        logger.info(f"Dropped {before - len(self.sales)} rows with negative values")

        self.sales = self.sales[self.sales['date'].dt.year.between(2013, 2015)]
        logger.info(f"Data restricted to years 2013–2015: {self.sales.shape}")

    def _clean_and_transform(self):
        logger.info("Aggregating and transforming data")

        grouped = self.sales.groupby(
            ['date', 'shop_id', 'item_id', 'date_block_num'],
            as_index=False
        ).agg({
            'item_cnt_day': 'sum',
            'item_price': 'mean'
        })

        neg_rows = grouped[grouped['item_cnt_day'] < 0]
        if not neg_rows.empty:
            logger.warning(f"Removing {len(neg_rows)} negative item_cnt_day rows post-aggregation")
            grouped = grouped[grouped['item_cnt_day'] >= 0]

        self.df_transformed = grouped
        logger.info(f"Transformation complete: {grouped.shape}")

    def run(self):
        logger.info("Running full ETL pipeline")
        try:
            self._load_data()
            self._validate_and_clean_sales()
            self._clean_and_transform()

            if self.df_transformed['item_cnt_day'].lt(0).any():
                count = self.df_transformed['item_cnt_day'].lt(0).sum()
                logger.warning(f"Final cleanup: removing {count} rows with negative item_cnt_day")
                self.df_transformed = self.df_transformed[self.df_transformed['item_cnt_day'] >= 0]

            logger.info("ETL pipeline completed successfully")

        except Exception as e:
            logger.exception(f"ETL pipeline failed: {e}")
            raise
