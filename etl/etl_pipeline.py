import argparse
import logging
import pandas as pd


def load_data(sales_path, items_path, categories_path, shops_path, test_path=None):
    logging.info("Loading datasets")
    sales = pd.read_csv(sales_path, parse_dates=["date"], dayfirst=True)
    items = pd.read_csv(items_path)
    cats = pd.read_csv(categories_path)
    shops = pd.read_csv(shops_path)
    test = pd.read_csv(test_path) if test_path else None
    return sales, items, cats, shops, test


def remove_invalid_rows(sales):
    logging.info("Removing rows with negative item_cnt_day or item_price")
    return sales[(sales["item_price"] >= 0)]


def filter_invalid_dates(sales):
    logging.info("Filtering dates outside 2013-2015 range")
    return sales[(sales["date"].dt.year >= 2013) & (sales["date"].dt.year <= 2015)]


def remove_duplicates(sales):
    logging.info("Aggregating sales per day (shop, item, date)")
    return sales.groupby(
        ["shop_id", "item_id", "date"], as_index=False
    ).agg(
        item_cnt_day=("item_cnt_day", "sum"),
        item_price=("item_price", "mean")
    )


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

    return sales


def clip_outliers(sales):
    logging.info("Clipping outliers at 1st–99th percentiles")
    low_cnt, high_cnt = sales["item_cnt_day"].quantile([0.01, 0.99])
    low_pr, high_pr = sales["item_price"].quantile([0.01, 0.99])
    sales["item_cnt_day"] = sales["item_cnt_day"].clip(lower=low_cnt, upper=high_cnt)
    sales["item_price"] = sales["item_price"].clip(lower=low_pr, upper=high_pr)
    return sales


def validate_final(df):
    logging.info("Validating output DataFrame")
    if df.empty:
        raise RuntimeError("Resulting dataset is empty!")
    expected = {"shop_id", "item_id", "month", "item_cnt_day", "item_price"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns after ETL: {missing}")


def save_output(df, path):
    logging.info(f"Saving cleaned data to {path}")
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

    sales = remove_invalid_rows(sales)
    sales = filter_invalid_dates(sales)
    sales = remove_duplicates(sales)
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