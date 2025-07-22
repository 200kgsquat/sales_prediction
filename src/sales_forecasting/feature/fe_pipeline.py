import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class FeaturePipeline:
    def __init__(self, items: pd.DataFrame, item_categories: pd.DataFrame, shops: pd.DataFrame):
        self.items = items.copy()
        self.item_categories = item_categories.copy()
        self.shops = shops.copy()
        self._preprocess_reference_data()
        logger.info("FeaturePipeline initialized")

    def _preprocess_reference_data(self):
        self.shops['shop_name'] = self.shops['shop_name'].str.lower().str.strip()
        self.items['item_name'] = self.items['item_name'].str.lower().str.strip()
        self.item_categories['item_category_name'] = self.item_categories['item_category_name'].str.lower().str.strip()

    @staticmethod
    def downcast(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
        return df

    def _add_lags(self, df: pd.DataFrame, group_cols: list, target: str, lags: list) -> pd.DataFrame:
        df = df.sort_values(group_cols + ['date_block_num'])
        for lag in lags:
            lag_col = f"{target}_lag_{lag}"
            df[lag_col] = df.groupby(group_cols)[target].shift(lag).astype('float32')
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting feature engineering. Input shape: {df.shape}")

        df['item_cnt_day'] = df['item_cnt_day'].fillna(0)
        df['item_price'] = df['item_price'].fillna(0)

        df['item_cnt_day'] = df['item_cnt_day'].clip(*df['item_cnt_day'].quantile([0.01, 0.99]))
        df['item_price'] = df['item_price'].clip(*df['item_price'].quantile([0.01, 0.99]))

        monthly_sales = df.groupby(['date_block_num', 'shop_id', 'item_id']).agg(
            item_cnt_month=('item_cnt_day', 'sum'),
            item_price=('item_price', 'mean')
        ).reset_index()

        all_blocks = sorted(monthly_sales['date_block_num'].unique())
        batches, known_pairs = [], set()

        for block in all_blocks:
            current_sales = monthly_sales[monthly_sales['date_block_num'] == block]
            known_pairs.update(tuple(x) for x in current_sales[['shop_id', 'item_id']].to_numpy())

            grid = pd.DataFrame(list(known_pairs), columns=['shop_id', 'item_id'])
            grid['date_block_num'] = block

            grid = grid.merge(current_sales, on=['date_block_num', 'shop_id', 'item_id'], how='left')
            grid['item_cnt_month'] = grid['item_cnt_month'].fillna(0)
            grid['item_price'] = grid['item_price'].fillna(0)

            grid = grid.merge(self.items, on='item_id', how='left')
            grid = grid.merge(self.item_categories, on='item_category_id', how='left')

            month = (block % 12) + 1
            grid['month_cos'] = np.cos(2 * np.pi * month / 12)

            batches.append(grid)

        full_df = pd.concat(batches, ignore_index=True)

        full_df = self._add_lags(full_df, ['shop_id', 'item_id'], 'item_cnt_month', [1, 2])
        full_df = self._add_lags(full_df, ['shop_id', 'item_id'], 'item_price', [1])

        for window in [3, 6]:
            col = f'rolling_median_{window}'
            full_df[col] = (
                full_df.groupby(['shop_id', 'item_id'])['item_cnt_month']
                .transform(lambda x: x.shift(1).rolling(window).median())
            )

        full_df['item_avg_sales_month_lag1'] = (
            full_df.groupby('item_id')['item_cnt_month']
            .transform(lambda x: x.shift(1).mean())
        )
        full_df['shop_item_avg_lag1'] = (
            full_df.groupby(['shop_id', 'item_id'])['item_cnt_month']
            .transform(lambda x: x.shift(1))
        )
        full_df['log_item_cnt_month_lag_1'] = np.log1p(full_df['item_cnt_month_lag_1'])

        full_df['price_change'] = full_df['item_price'] - full_df['item_price_lag_1']
        full_df['price_increasing'] = (full_df['price_change'] > 0)

        # Удаляем ненужные текстовые признаки
        full_df = full_df.drop(columns=['item_name', 'item_category_name'], errors='ignore')

        lag_cols = [
            'item_cnt_month_lag_1', 'item_cnt_month_lag_2',
            'rolling_median_3', 'rolling_median_6',
            'item_avg_sales_month_lag1', 'shop_item_avg_lag1', 'log_item_cnt_month_lag_1',
            'item_price_lag_1', 'price_change'
        ]
        for col in lag_cols:
            if col in full_df.columns:
                full_df[col] = full_df[col].fillna(0)

        full_df = self.downcast(full_df)
        logger.info(f"Feature engineering complete. Final shape: {full_df.shape}")
        return full_df
