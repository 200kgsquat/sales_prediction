import pandas as pd
import numpy as np
from sales_forecasting.utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePipeline:
    def __init__(self, items: pd.DataFrame, item_categories: pd.DataFrame, shops: pd.DataFrame):
        # Create copies of reference data to avoid modifying the originals
        self.items = items.copy()
        self.item_categories = item_categories.copy()
        self.shops = shops.copy()
        # Clean the reference data
        self._preprocess_reference_data()
        logger.info("FeaturePipeline initialized")

    def _preprocess_reference_data(self):
        # Normalize and strip whitespace from text columns
        self.shops['shop_name'] = self.shops['shop_name'].str.lower().str.strip()
        self.items['item_name'] = self.items['item_name'].str.lower().str.strip()
        self.item_categories['item_category_name'] = self.item_categories['item_category_name'].str.lower().str.strip()

    @staticmethod
    def downcast(df: pd.DataFrame) -> pd.DataFrame:
        # Downcast numeric columns to reduce memory usage
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
        return df

    def _add_lags(self, df: pd.DataFrame, group_cols: list, target: str, lags: list) -> pd.DataFrame:
        # Generate lag features for a given target column
        df = df.sort_values(group_cols + ['date_block_num'])
        for lag in lags:
            lag_col = f"{target}_lag_{lag}"
            df[lag_col] = df.groupby(group_cols)[target].shift(lag).astype('float32')
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting feature engineering. Input shape: {df.shape}")

        # Handle missing values
        df['item_cnt_day'] = df['item_cnt_day'].fillna(0)
        df['item_price'] = df['item_price'].fillna(0)

        # Remove outliers by clipping extreme values
        df['item_cnt_day'] = df['item_cnt_day'].clip(*df['item_cnt_day'].quantile([0.01, 0.99]))
        df['item_price'] = df['item_price'].clip(*df['item_price'].quantile([0.01, 0.99]))

        # Aggregate to monthly level
        monthly_sales = df.groupby(['date_block_num', 'shop_id', 'item_id']).agg(
            item_cnt_month=('item_cnt_day', 'sum'),
            item_price=('item_price', 'mean')
        ).reset_index()

        all_blocks = sorted(monthly_sales['date_block_num'].unique())
        batches = []

        # Generate grid for each block, using only past shop-item pairs (to avoid leakage)
        for block in all_blocks:
            if block == 0:
                continue  # No history exists in the first block

            # Get known shop-item pairs from all previous blocks
            past_sales = monthly_sales[monthly_sales['date_block_num'] < block]
            past_pairs = past_sales[['shop_id', 'item_id']].drop_duplicates()

            grid = past_pairs.copy()
            grid['date_block_num'] = block

            # Merge current month's data (may be missing for some pairs)
            current_sales = monthly_sales[monthly_sales['date_block_num'] == block]
            grid = grid.merge(current_sales, on=['date_block_num', 'shop_id', 'item_id'], how='left')

            # Fill missing target values with 0
            grid['item_cnt_month'] = grid['item_cnt_month'].fillna(0)
            grid['item_price'] = grid['item_price'].fillna(0)

            # Merge item and category metadata
            grid = grid.merge(self.items, on='item_id', how='left')
            grid = grid.merge(self.item_categories, on='item_category_id', how='left')

            # Add cyclical month feature
            month = (block % 12) + 1
            grid['month_cos'] = np.cos(2 * np.pi * month / 12)

            batches.append(grid)

        # Combine all blocks
        full_df = pd.concat(batches, ignore_index=True)

        # Add lag features for item count and price
        full_df = self._add_lags(full_df, ['shop_id', 'item_id'], 'item_cnt_month', [1, 2])
        full_df = self._add_lags(full_df, ['shop_id', 'item_id'], 'item_price', [1])

        # Add rolling window statistics (median)
        for window in [3, 6]:
            col = f'rolling_median_{window}'
            full_df[col] = (
                full_df.groupby(['shop_id', 'item_id'])['item_cnt_month']
                .transform(lambda x: x.shift(1).rolling(window).median())
            )

        # Global item average monthly sales (with 1 month lag)
        full_df['item_avg_sales_month_lag1'] = (
            full_df.groupby('item_id')['item_cnt_month']
            .transform(lambda x: x.shift(1).mean())
        )

        # Raw lag-1 values for each shop-item
        full_df['shop_item_avg_lag1'] = (
            full_df.groupby(['shop_id', 'item_id'])['item_cnt_month']
            .transform(lambda x: x.shift(1))
        )

        # Log-transformed lag
        full_df['log_item_cnt_month_lag_1'] = np.log1p(full_df['item_cnt_month_lag_1'])

        # Price change and price trend indicator
        full_df['price_change'] = full_df['item_price'] - full_df['item_price_lag_1']
        full_df['price_increasing'] = (full_df['price_change'] > 0)

        # Drop text features (not needed for modeling)
        full_df = full_df.drop(columns=['item_name', 'item_category_name'], errors='ignore')

        # Fill missing lag values
        lag_cols = [
            'item_cnt_month_lag_1', 'item_cnt_month_lag_2',
            'rolling_median_3', 'rolling_median_6',
            'item_avg_sales_month_lag1', 'shop_item_avg_lag1', 'log_item_cnt_month_lag_1',
            'item_price_lag_1', 'price_change'
        ]
        for col in lag_cols:
            if col in full_df.columns:
                full_df[col] = full_df[col].fillna(0)

        # Optimize memory usage
        full_df = self.downcast(full_df)
        logger.info(f"Feature engineering complete. Final shape: {full_df.shape}")
        return full_df
