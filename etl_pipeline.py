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
    logging.info("Dropping duplicates & aggregating sales per day")
    sales = sales.drop_duplicates(subset=["shop_id", "item_id", "date"])
    return sales.groupby(
        ["shop_id", "item_id", "date"], as_index=False
    ).agg(
        item_cnt_day=("item_cnt_day", "sum"),
        item_price=("item_price", "mean")
    )


def handle_missing(sales):
    logging.info("Filling missing item_cnt_day using linear interpolation")
    sales["item_cnt_day"] = sales.groupby(["shop_id", "item_id"])["item_cnt_day"]\
                                 .transform(lambda x: x.interpolate(method="linear").fillna(0))

    logging.info("Filling missing item_price using forward/backward fill per item")
    sales["item_price"] = sales.groupby("item_id")["item_price"]\
                               .transform(lambda x: x.ffill().bfill().fillna(0))
    return sales


def clip_outliers(sales):
    logging.info("Clipping outliers at 1st–99th percentiles")
    low_cnt, high_cnt = sales["item_cnt_day"].quantile([0.01, 0.99])
    low_pr, high_pr = sales["item_price"].quantile([0.01, 0.99])
    sales["item_cnt_day"] = sales["item_cnt_day"].clip(lower=low_cnt, upper=high_cnt)
    sales["item_price"] = sales["item_price"].clip(lower=low_pr, upper=high_pr)
    return sales


def merge_metadata(sales, items, cats, shops):
    logging.info("Merging item/shop metadata")
    sales = sales.merge(
        items[["item_id", "item_category_id"]], on="item_id", how="left"
    )
    sales = sales.merge(
        cats[["item_category_id", "item_category_name"]],
        on="item_category_id", how="left"
    )
    sales = sales.merge(
        shops[["shop_id", "shop_name"]], on="shop_id", how="left"
    )
    return sales


def compute_date_block_num(sales):
    logging.info("Computing date_block_num from date")
    sales["date_block_num"] = (
        (sales["date"].dt.year - 2013) * 12 + (sales["date"].dt.month - 1)
    )
    return sales


def expand_monthly_grid(sales):
    logging.info("Expanding only real (shop_id, item_id, month) combinations")
    sales["month"] = sales["date"].dt.to_period("M")
    monthly = sales.groupby(["shop_id", "item_id", "month"], as_index=False).agg({
        "item_cnt_day": "sum",
        "item_price": "mean"
    })
    monthly["item_cnt_day"] = monthly["item_cnt_day"].fillna(0)
    monthly["item_price"] = monthly["item_price"].fillna(0)
    return monthly


def add_lag_features(df):
    logging.info("Generating lag features")
    df = df.sort_values(["shop_id", "item_id", "month"])
    grp = df.groupby(["shop_id", "item_id"])["item_cnt_day"]
    df["sales_prev_month"] = grp.shift(1).fillna(0)
    df["sales_prev_2_months"] = grp.shift(2).fillna(0)
    return df


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
    df.to_csv(path, index=False)


def run_pipeline(args):
    sales, items, cats, shops, test = load_data(
        args.sales, args.items, args.categories, args.shops, args.test
    )

    sales = remove_invalid_rows(sales)
    sales = filter_invalid_dates(sales)
    sales = remove_duplicates(sales)
    sales = handle_missing(sales)
    sales = clip_outliers(sales)
    sales = merge_metadata(sales, items, cats, shops)
    sales = compute_date_block_num(sales)
    sales = expand_monthly_grid(sales)
    sales = merge_metadata(sales, items, cats, shops)
    sales = add_lag_features(sales)

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