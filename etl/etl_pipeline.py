import argparse
import logging
import pandas as pd
import os


def load_data(sales_path, items_path, categories_path, shops_path, test_path=None):
    logging.info("Loading datasets")
    # Normalize paths to handle non-ASCII characters
    sales_path = os.path.normpath(sales_path)
    items_path = os.path.normpath(items_path)
    categories_path = os.path.normpath(categories_path)
    shops_path = os.path.normpath(shops_path)
    test_path = os.path.normpath(test_path) if test_path else None

    # Check file existence
    for path in [sales_path, items_path, categories_path, shops_path]:
        if not os.path.exists(path):
            logging.error(f"File not found: {path}")
            raise FileNotFoundError(f"File not found: {path}")

    sales = pd.read_csv(sales_path, parse_dates=["date"], dayfirst=True)

    # Создаем date_block_num, если отсутствует
    if "date_block_num" not in sales.columns:
        logging.warning("date_block_num not found in sales data — creating it from date column")
        sales["date_block_num"] = (sales["date"].dt.year - 2013) * 12 + sales["date"].dt.month - 1

    items = pd.read_csv(items_path)
    cats = pd.read_csv(categories_path)
    shops = pd.read_csv(shops_path)
    test = pd.read_csv(test_path) if test_path and os.path.exists(test_path) else None
    logging.info(f"Columns after load_data: {sales.columns.tolist()}")
    return sales, items, cats, shops, test


def remove_negative_values(sales):
    logging.info("Removing rows with negative item_cnt_day or item_price")
    before = len(sales)
    sales = sales[(sales["item_cnt_day"] >= 0) & (sales["item_price"] >= 0)]
    after = len(sales)
    logging.info(f"Removed {before - after} rows with negative values")
    if "date_block_num" not in sales.columns:
        logging.error("date_block_num missing after remove_negative_values")
        raise ValueError("date_block_num missing after remove_negative_values")
    logging.info(f"Columns after remove_negative_values: {sales.columns.tolist()}")
    return sales


def merge_duplicate_shops(sales):
    logging.info("Merging duplicate shops by reassigning shop_id")
    shop_mapping = {0: 57, 1: 58, 10: 11, 39: 40}
    sales["shop_id"] = sales["shop_id"].replace(shop_mapping)
    sales = sales.groupby(
        ["date_block_num", "shop_id", "item_id", "date"], as_index=False
    ).agg(
        {
            "item_cnt_day": "sum",
            "item_price": "mean"
        }
    )
    logging.info(f"Columns after grouping: {sales.columns.tolist()}")
    logging.info("Duplicate shops merged and aggregated")
    if "date_block_num" not in sales.columns:
        logging.error("date_block_num missing after merge_duplicate_shops")
        raise ValueError("date_block_num missing after merge_duplicate_shops")
    logging.info(f"Columns after merge_duplicate_shops: {sales.columns.tolist()}")
    return sales


def filter_invalid_dates(sales):
    logging.info("Filtering dates outside 2013-2015 range")
    sales = sales[(sales["date"].dt.year >= 2013) & (sales["date"].dt.year <= 2015)]
    if "date_block_num" not in sales.columns:
        logging.error("date_block_num missing after filter_invalid_dates")
        raise ValueError("date_block_num missing after filter_invalid_dates")
    logging.info(f"Columns after filter_invalid_dates: {sales.columns.tolist()}")
    return sales


def handle_missing(sales, report_path=None):
    missing_cnt = sales["item_cnt_day"].isna().sum()
    missing_prices = sales["item_price"].isna().sum()
    logging.info(f"Found {missing_cnt} missing values in 'item_cnt_day'")
    logging.info(f"Found {missing_prices} missing values in 'item_price'")
    total_missing = missing_cnt + missing_prices
    if total_missing == 0:
        logging.info("No missing values detected.")
    else:
        logging.warning("Missing values detected — no imputation applied.")
        if report_path:
            logging.info(f"Saving rows with missing values to {report_path}")
            missing_rows = sales[sales["item_cnt_day"].isna() | sales["item_price"].isna()]
            missing_rows.to_csv(report_path, index=False)
    if "date_block_num" not in sales.columns:
        logging.error("date_block_num missing after handle_missing")
        raise ValueError("date_block_num missing after handle_missing")
    logging.info(f"Columns after handle_missing: {sales.columns.tolist()}")
    return sales


def clip_outliers(sales):
    logging.info("Clipping outliers at 1st–99th percentiles")
    low_cnt, high_cnt = sales["item_cnt_day"].quantile([0.01, 0.99])
    low_pr, high_pr = sales["item_price"].quantile([0.01, 0.99])
    sales["item_cnt_day"] = sales["item_cnt_day"].clip(lower=low_cnt, upper=high_cnt)
    sales["item_price"] = sales["item_price"].clip(lower=low_pr, upper=high_pr)
    if "date_block_num" not in sales.columns:
        logging.error("date_block_num missing after clip_outliers")
        raise ValueError("date_block_num missing after clip_outliers")
    logging.info(f"Columns after clip_outliers: {sales.columns.tolist()}")
    return sales


def validate_final(df):
    logging.info("Validating output DataFrame")
    if df.empty:
        raise RuntimeError("Resulting dataset is empty!")
    expected = {"shop_id", "item_id", "date", "item_cnt_day", "item_price", "date_block_num"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns after ETL: {missing}")
    logging.info(f"Final columns: {df.columns.tolist()}")


def save_output(df, path):
    logging.info(f"Saving cleaned data to {path}")
    path = os.path.normpath(path)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".pkl") or path.endswith(".pickle"):
        df.to_pickle(path)
    else:
        df.to_csv(path, index=False)
        logging.warning("Saved as CSV — consider using .parquet or .pkl for efficiency")


def run_pipeline(args):
    sales, items, cats, shops, test = load_data(
        args.sales, args.items, args.categories, args.shops, args.test
    )

    sales = remove_negative_values(sales)
    sales = merge_duplicate_shops(sales)
    sales = filter_invalid_dates(sales)
    sales = handle_missing(sales)
    sales = clip_outliers(sales)

    validate_final(sales)
    save_output(sales, args.output)
    logging.info("ETL pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL for future sales prediction")
    parser.add_argument("--sales", required=True, help="Path to sales_train.csv")
    parser.add_argument("--items", required=True, help="Path to items.csv")
    parser.add_argument("--categories", required=True, help="Path to item_categories.csv")
    parser.add_argument("--shops", required=True, help="Path to shops.csv")
    parser.add_argument("--test", help="Optional path to test.csv")
    parser.add_argument("--output", required=True, help="Path for cleaned output CSV")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    run_pipeline(parser.parse_args())