Data Quality Check (DQC) Steps:
Data Loading

Imported datasets: sales_train, items, item_categories, shops, test, and sample_submission.

Initial Exploration

Examined structure of all datasets using .head() to understand columns and sample records.

Shop ID Standardization

Consolidated duplicate shop IDs in sales_train and test datasets:

Shop 0 → 57

Shop 1 → 58

Shop 10 → 11

Shop 39 → 40

Data Merging

Combined datasets into a unified DataFrame (df):

Joined sales_train with items → item_categories → shops using item_id, item_category_id, and shop_id.

Data Transformation

Converted date from string to datetime format (DD.MM.YYYY → datetime64).

Sorted records chronologically and reset index.

Missing Value Check

Identified no missing values across all columns in df.

Duplicate Handling

Detected duplicate rows.

Aggregated item_cnt_day (daily sales) for identical items in duplicates.

Re-examined data: 0 duplicates remained after aggregation.

Temporal Scope Validation

Confirmed date range: January 2013 to October 2015.

Outlier Detection

Visualized distributions:

item_cnt_day (daily sales quantity)

item_price (item price)

Addressed anomalies:

Removed negative prices (item_price > 0).

Capped extreme prices at 50,000 RUB (upper bound).

Returns Analysis

Created returns feature (absolute value of negative item_cnt_day).

Explored returns pattern:

Insight: Higher return rates for low-priced items.

Key Insights:
Data covers 34 months of sales history.

Price vs Returns show inverse correlation (cheaper items returned more frequently).

Critical outliers addressed:

Negative/zero prices removed.

Extreme prices capped.

Duplicate sales aggregated.

Final Output: Cleaned dataset ready for time-series feature engineering and modeling.