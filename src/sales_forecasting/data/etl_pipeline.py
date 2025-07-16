import logging
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class ETLPipeline:
    """ETL pipeline for sales data processing."""

    def __init__(
        self,
        sales_path: str,
        items_path: str,
        categories_path: str,
        shops_path: str,
        output_path: Optional[str] = None
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
        """Load all input data files."""
        logger.info("Loading input data")

        for path in [self.sales_path, self.items_path, self.categories_path, self.shops_path]:
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")

        self.sales = pd.read_csv(self.sales_path)
        self.items = pd.read_csv(self.items_path)
        self.categories = pd.read_csv(self.categories_path)
        self.shops = pd.read_csv(self.shops_path)

        logger.info("Data loaded")

    def _clean_and_transform_sales_data(self):
        """Validate, clean and transform sales data."""
        logger.info("Cleaning and transforming sales data")

        # Fix duplicate shop_ids
        shop_id_mapping = {
            0: 57,
            1: 58,
            10: 11,
            39: 40
        }
        self.sales['shop_id'] = self.sales['shop_id'].replace(shop_id_mapping)

        # Convert date
        self.sales['date'] = pd.to_datetime(
            self.sales['date'],
            dayfirst=True,
            errors='coerce'
        )

        if self.sales['date'].isna().any():
            bad_dates = self.sales[self.sales['date'].isna()]
            logger.error(f"Invalid dates found: {bad_dates[['date']].head()}")
            raise ValueError(f"Invalid dates found: {bad_dates.shape[0]} rows")

        # Numeric conversion
        for col in ['item_id', 'shop_id', 'date_block_num']:
            self.sales[col] = pd.to_numeric(self.sales[col], errors='coerce', downcast='integer')
            if self.sales[col].isna().any():
                bad_values = self.sales[self.sales[col].isna()][[col]].head()
                logger.error(f"Invalid values in {col}: {bad_values}")
                raise ValueError(f"Invalid values in column {col}")

        # Remove duplicates
        initial_count = len(self.sales)
        self.sales = self.sales.drop_duplicates(subset=['date', 'item_id', 'shop_id'])
        dup_count = initial_count - len(self.sales)
        if dup_count > 0:
            logger.warning(f"Removed {dup_count} duplicate rows")

        # Remove negative values
        before_filter = len(self.sales)
        self.sales = self.sales[
            (self.sales['item_price'] >= 0) &
            (self.sales['item_cnt_day'] >= 0)
        ]
        filtered_count = before_filter - len(self.sales)
        if filtered_count > 0:
            logger.warning(f"Removed {filtered_count} rows with negative prices or item counts")

        # Filter by date range
        self.sales = self.sales[self.sales['date'].dt.year.between(2013, 2015)]

        # Fill NaNs
        self.sales['item_cnt_day'] = self.sales['item_cnt_day'].fillna(0)
        self.sales['item_price'] = self.sales['item_price'].fillna(0)

        # Clip outliers
        self.sales['item_cnt_day'] = self.sales['item_cnt_day'].clip(
            *self.sales['item_cnt_day'].quantile([0.01, 0.99])
        )
        self.sales['item_price'] = self.sales['item_price'].clip(
            *self.sales['item_price'].quantile([0.01, 0.99])
        )

        # Transform (aggregation)
        self.df_transformed = self.sales.groupby(
            ['date', 'shop_id', 'item_id', 'date_block_num'],
            as_index=False
        ).agg({
            'item_cnt_day': 'sum',
            'item_price': 'mean'
        })

        logger.info(f"Sales data cleaned and transformed. Shape: {self.df_transformed.shape}")

    def run(self):
        """Run complete ETL pipeline."""
        logger.info("Starting ETL pipeline")
        try:
            self._load_data()
            self._clean_and_transform_sales_data()
            logger.info("ETL pipeline completed successfully")
        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            raise
