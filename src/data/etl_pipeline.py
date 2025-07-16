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

        # Data storage
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

        self.sales = self.sales[
            (self.sales['item_price'] >= 0) & 
            (self.sales['item_cnt_day'] >= 0)
        ]

        logger.info("Data loaded and initial filtering done")

    def _validate_and_clean_sales(self):
        """Validate and clean sales data."""
        logger.info("Validating and cleaning sales data")

        # 1. Date conversion
        self.sales['date'] = pd.to_datetime(
            self.sales['date'],
            dayfirst=True,
            errors='coerce'
        )

        if self.sales['date'].isna().any():
            bad_dates = self.sales[self.sales['date'].isna()]
            logger.error(f"Invalid dates found: {bad_dates[['date']].head()}")
            raise ValueError(f"Invalid dates found: {bad_dates.shape[0]} rows")

        # 2. Numeric conversion
        for col in ['item_id', 'shop_id', 'date_block_num']:
            self.sales[col] = pd.to_numeric(self.sales[col], errors='coerce', downcast='integer')
            if self.sales[col].isna().any():
                bad_values = self.sales[self.sales[col].isna()][[col]].head()
                logger.error(f"Invalid values in {col}: {bad_values}")
                raise ValueError(f"Invalid values in column {col}")

        # 3. Drop duplicates
        initial_count = len(self.sales)
        self.sales = self.sales.drop_duplicates(subset=['date', 'item_id', 'shop_id'])
        dup_count = initial_count - len(self.sales)
        if dup_count > 0:
            logger.warning(f"Removed {dup_count} duplicate rows")

        before_filter = len(self.sales)
        self.sales = self.sales[
            (self.sales['item_price'] >= 0) & 
            (self.sales['item_cnt_day'] >= 0)
        ]
        filtered_count = before_filter - len(self.sales)
        if filtered_count > 0:
            logger.warning(f"Removed {filtered_count} rows with negative prices or item counts")

        # 5. Filter by date range
        self.sales = self.sales[self.sales['date'].dt.year.between(2013, 2015)]

        logger.info(f"Sales data validated and cleaned. New shape: {self.sales.shape}")

    def _clean_and_transform(self):
        """Clean and transform data."""
        logger.info("Cleaning and transforming data")

        #transform
        self.sales = self.sales.groupby(
            ['date', 'shop_id', 'item_id', 'date_block_num'],
            as_index=False
        ).agg({
            'item_cnt_day': 'sum',
            'item_price': 'mean'
        })

        neg_count = (self.sales['item_cnt_day'] < 0).sum()
        if neg_count > 0:
            logger.warning(f"Removing {neg_count} rows with negative item_cnt_day after aggregation")
            self.sales = self.sales[self.sales['item_cnt_day'] >= 0]

        
        self.df_transformed = self.sales

        logger.info(f"Data transformation complete. Shape: {self.df_transformed.shape}")

    def run(self):
        """Run complete ETL pipeline."""
        logger.info("Starting ETL pipeline")
        try:
            self._load_data()
            self._validate_and_clean_sales()
            self._clean_and_transform()
            neg_mask = self.df_transformed['item_cnt_day'] < 0
            if neg_mask.any():
                logger.warning(f"Removing {neg_mask.sum()} negative item_cnt_day rows before finalizing")
                self.df_transformed = self.df_transformed[~neg_mask]

            logger.info("ETL pipeline completed successfully")
        except Exception as e:
            logger.error(f"ETL pipeline failed: {str(e)}")
            raise
