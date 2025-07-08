import logging
import numpy as np
import pandas as pd

from validation_schema import validate_dataset  # import validate_dataset from your schema module

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

def aggregate_monthly(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sales into item_cnt_month by (date_block_num, shop_id, item_id).
    Removes duplicates after aggregation.
    """
    sales['date'] = pd.to_datetime(sales['date'], dayfirst=True, errors='coerce')
    sales['month'] = sales['date'].dt.month
    sales['year'] = sales['date'].dt.year
    sales['date_block_num'] = (sales['year'] - sales['year'].min()) * 12 + (sales['month'] - 1)

    grouped = (
        sales
        .groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)
        .agg(item_cnt_month=('item_cnt_day', 'sum'))
        .drop_duplicates(subset=['date_block_num', 'shop_id', 'item_id'])
    )
    
    return grouped

def build_test_features(
    test_df: pd.DataFrame,
    expanded_df: pd.DataFrame,
    items: pd.DataFrame,
    item_categories: pd.DataFrame,
    shops: pd.DataFrame,
    block_num: int
) -> pd.DataFrame:
    """
    Using the test_df (with columns item_id, shop_id, and date_block_num=block_num),
    build all additional features according to your logic.
    """
    test_df = test_df.copy()
    test_df['date_block_num'] = block_num

    # merge with reference tables
    df = (
        test_df
        .merge(items, on='item_id', how='left')
        .merge(item_categories, on='item_category_id', how='left')
        .merge(shops, on='shop_id', how='left')
    )

    # season based on date_block_num
    def get_season_from_block_num(bn: int) -> str:
        m = (bn % 12) + 1
        if m in [12, 1, 2]:
            return "Зима"
        if m in [3, 4, 5]:
            return "Весна"
        if m in [6, 7, 8]:
            return "Лето"
        return "Осень"

    df['season'] = df['date_block_num'].map(get_season_from_block_num)
    df['month_sin'] = np.sin(2 * np.pi * df['date_block_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date_block_num'] / 12)

    # helper function to get monthly sales series for a given month
    def month_series(m: int) -> pd.Series:
        tmp = expanded_df.loc[
            expanded_df['date_block_num'] == m,
            ['shop_id','item_id','item_cnt_month']
        ]
        return tmp.set_index(['shop_id','item_id'])['item_cnt_month']

    keys = list(zip(df['shop_id'], df['item_id']))
    # lags for sales
    lags = {
        'item_cnt_month_lag_1': block_num - 1,
        'item_cnt_month_lag_2': block_num - 2,
        'item_cnt_month_lag_3': block_num - 3,
        'item_cnt_month_lag_12': block_num - 12,
    }
    for col, bn in lags.items():
        ser = month_series(bn)
        df[col] = pd.Series(
            [ser.get(k, 0) for k in keys],
            index=df.index
        )

    # rolling median calculation
    def rolling_median(window: int) -> np.ndarray:
        months = range(block_num - window, block_num)
        frames = [month_series(m) for m in months]
        df_roll = pd.concat(frames, axis=1).fillna(0)
        return df_roll.median(axis=1).values

    df['rolling_median_3'] = rolling_median(3)
    df['rolling_median_6'] = rolling_median(6)

    # average sales (simply copy lag-1 per your logic)
    df['item_avg_sales_month_lag1'] = df['item_cnt_month_lag_1']
    df['shop_item_avg_lag1']       = df['item_cnt_month_lag_1']
    df['log_item_cnt_month_lag_1'] = np.log1p(df['item_cnt_month_lag_1'])

    # parse city and shop type from shop_name
    df['city']         = df['shop_name'].str.split().str[0]
    df['type_of_shop'] = df['shop_name'].str.split().str[1]

    # price info from expanded_df reference
    grp_price = expanded_df.groupby('item_id')['item_price']
    df['item_price']       = df['item_id'].map(grp_price.last())
    df['item_min_price']   = df['item_id'].map(grp_price.min())
    df['item_max_price']   = df['item_id'].map(grp_price.max())
    df['price_range']      = df['item_max_price'] - df['item_min_price']
    df['price_increasing'] = df['item_id'].map(expanded_df.groupby('item_id')['price_increasing'].last()).fillna(0)

    # final cleanup - drop unnecessary columns if present
    for c in ['item_cnt_day','item_cnt_month','date']:
        df.drop(columns=c, errors='ignore', inplace=True)

    return df

def fe_pipeline(
    sales_daily_path: str,
    items_path: str,
    categories_path: str,
    shops_path: str,
    test_template_path: str,
    output_path: str,
    test_block_num: int = 34
):
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting FE pipeline")

    # 1. load data
    sales = pd.read_csv(sales_daily_path)
    items = pd.read_csv(items_path)
    item_categories = pd.read_csv(categories_path)
    shops = pd.read_csv(shops_path)
    test_template = pd.read_csv(test_template_path)  # contains shop_id, item_id

    # 2. aggregate sales history monthly
    expanded_df = aggregate_monthly(sales)

    # 3. build features for the test period
    fe_df = build_test_features(
        test_template,
        expanded_df=expanded_df,
        items=items,
        item_categories=item_categories,
        shops=shops,
        block_num=test_block_num
    )

    # 4. validation
    if not validate_dataset(fe_df, dataset_name="feature_engineered_data"):
        logger.critical("Validation failed — pipeline halted.")
        raise ValueError("Validation failed — pipeline halted.")

    # 5. save results
    fe_df.to_csv(output_path, index=False)
    logger.info(f"FE data successfully saved to {output_path}")
    return fe_df

if __name__ == "__main__":
    fe_pipeline(
        sales_daily_path="datasets/sales_train.csv",
        items_path="datasets/items.csv",
        categories_path="datasets/item_categories.csv",
        shops_path="datasets/shops.csv",
        test_template_path="datasets/test_template.csv",
        output_path="datasets/fe_df.csv",
        test_block_num=34
    )
