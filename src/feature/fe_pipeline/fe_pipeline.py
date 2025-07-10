import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeaturePipeline:
    def __init__(self, items: pd.DataFrame, item_categories: pd.DataFrame, shops: pd.DataFrame):
        self.items = items
        self.item_categories = item_categories
        self.shops = shops
        logger.info("Metadata DataFrames set inside FeaturePipeline")

    @staticmethod
    def get_season(block: int) -> str:
        month = (block % 12) + 1
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    #@staticmethod
    #def downcast(df: pd.DataFrame) -> pd.DataFrame:
    #    float_cols = df.select_dtypes(include='float64').columns
    #    int_cols = df.select_dtypes(include='int64').columns
    #    df[float_cols] = df[float_cols].astype('float32')
    #    df[int_cols] = df[int_cols].astype('int32')
    #    return df

    @staticmethod
    def add_lags(data: pd.DataFrame, group_cols: list, target: str, lags: list) -> pd.DataFrame:
        data = data.sort_values(group_cols + ['date_block_num']).copy()
        for lag in lags:
            col = f"{target}_lag_{lag}"
            data[col] = data.groupby(group_cols)[target].shift(lag).astype('float64')
        return data
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting feature engineering on input data: {df.shape}")

        # Aggregate monthly sales and prices
        monthly_sales = df.groupby(['date_block_num', 'shop_id', 'item_id']).agg({
            'item_cnt_day': 'sum',
            'item_price': 'mean'
        }).reset_index()
        monthly_sales.rename(columns={
            'item_cnt_day': 'item_cnt_month',
            'item_price': 'item_price'
        }, inplace=True)

        # Get unique values
        all_dates = sorted(df['date_block_num'].unique())
        all_shops = self.shops['shop_id'].unique()
        all_items = self.items['item_id'].unique()

        batch_list = []
        for block_num in all_dates:
            logger.info(f"Processing batch for date_block_num = {block_num}")

            grid = pd.MultiIndex.from_product(
                [[block_num], all_shops, all_items],
                names=['date_block_num', 'shop_id', 'item_id']
            ).to_frame(index=False)

            sales_batch = monthly_sales[monthly_sales['date_block_num'] == block_num]

            df_batch = grid.merge(sales_batch, on=['date_block_num', 'shop_id', 'item_id'], how='left')

            # Fill missing sales and price with 0
            df_batch['item_cnt_month'] = df_batch['item_cnt_month'].fillna(0).astype('float64')
            # Important: item_price can be zero if no sales, but keep as float64
            df_batch['item_price'] = df_batch['item_price'].fillna(0).astype('float64')

            # Merge metadata
            df_batch = df_batch.merge(self.items, on='item_id', how='left') \
                               .merge(self.item_categories, on='item_category_id', how='left') \
                               .merge(self.shops, on='shop_id', how='left')

            # Season
            df_batch['season'] = self.get_season(block_num)

            month = (block_num % 12) + 1
            df_batch['month_sin'] = np.sin(2 * np.pi * month / 12).astype('float64')
            df_batch['month_cos'] = np.cos(2 * np.pi * month / 12).astype('float64')

            # Extract city and type_of_shop
            df_batch['city'] = df_batch['shop_name'].str.extract(r'(^\S+)')[0].fillna('')
            df_batch['type_of_shop'] = df_batch['shop_name'].str.extract(r'^\S+\s+(\S+)')[0].fillna('')

            # df_batch = self.downcast(df_batch)  
            batch_list.append(df_batch)

            df_full = pd.concat(batch_list, ignore_index=True)
            logger.info(f"All batches combined: shape = {df_full.shape}")

        logger.info("Removing duplicates in chunks by date_block_num")
        chunked_batches = []
        unique_blocks = df_full['date_block_num'].unique()

        for block in unique_blocks:
            logger.info(f"Removing duplicates in block {block}")
            chunk = df_full[df_full['date_block_num'] == block]
            chunk = chunk.drop_duplicates(subset=['date_block_num', 'shop_id', 'item_id', 'item_cnt_month'])
            chunked_batches.append(chunk)

        df_full = pd.concat(chunked_batches, ignore_index=True)

        # Add lags for sales and price
        df_full = self.add_lags(df_full, ['shop_id', 'item_id'], 'item_cnt_month', [1, 2, 3, 12])
        df_full = self.add_lags(df_full, ['shop_id', 'item_id'], 'item_price', [1])

        # Price increase flag
        df_full['price_increasing'] = (df_full['item_price'] > df_full['item_price_lag_1']).astype('int64')

        # Rolling medians on sales lagged by 1 month
        df_full = df_full.sort_values(['shop_id', 'item_id', 'date_block_num'])
        grouped = df_full.groupby(['shop_id', 'item_id'])

        df_full['rolling_median_3'] = grouped['item_cnt_month'].shift(1).rolling(3, min_periods=1).median().reset_index(drop=True)
        df_full['rolling_median_6'] = grouped['item_cnt_month'].shift(1).rolling(6, min_periods=1).median().reset_index(drop=True)
        df_full[['rolling_median_3', 'rolling_median_6']] = df_full[['rolling_median_3', 'rolling_median_6']].fillna(0)

        # Median of lag-1 sales by item and shop-item
        df_full['item_avg_sales_month_lag1'] = df_full.groupby(['item_id', 'date_block_num'])['item_cnt_month_lag_1'].transform('median').fillna(0)
        df_full['shop_item_avg_lag1'] = df_full.groupby(['shop_id', 'item_id'])['item_cnt_month_lag_1'].transform('median').fillna(0)

        # Log transform of lag-1 sales
        df_full['log_item_cnt_month_lag_1'] = np.log1p(df_full['item_cnt_month_lag_1'].fillna(0))

        logger.info("Feature engineering completed")
        return df_full
