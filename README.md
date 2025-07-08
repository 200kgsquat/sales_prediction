**Project Overview**

This repository contains a feature engineering pipeline for a retail sales prediction project. The goal is to process raw daily sales data, validate it, generate monthly aggregated features, validate the feature set, and prepare it for modeling and prediction.

---

## Directory Structure

```text
├── datasets/               # Raw and intermediate datasets
│   ├── sales_train.csv     # Raw daily sales data
│   ├── cleaned_sales.csv   # Validated sales data
│   ├── items.csv           # Item metadata
│   ├── item_categories.csv # Item category metadata
│   ├── shops.csv           # Shop metadata
│   ├── test_template.csv   # (shop_id, item_id) for test period
│   ├── test.csv            # Final test set for prediction
│   ├── sample_submission.csv
│   └── fe_df.csv           # Output of feature engineering
│
├── validation/             # Data validation scripts and tests
│   ├── validate_raw.py     # Pandera schema and validation for raw sales
│   └── test_validate_raw.py# pytest tests for raw validation
│   ├── validate_features.py# Pandera schema for feature set validation
│
├── pipeline/               # Feature engineering pipeline scripts
│   └── fe_pipeline.py      # Aggregation and feature-building functions
│
├── notebooks/              # Exploratory notebook and ad-hoc scripts
│   └── feature_expansion.py # Script for expanding monthly grid and initial FE
│
├── checkpoint.pkl          # Intermediate pickled DataFrame
├── downcasted.pkl          # Pickled DF with optimized dtypes
└── README.md               # Project documentation
```

---

## Getting Started

### Prerequisites

* Python 3.8 or higher
* Recommended to use a virtual environment (e.g., `venv` or `conda`)

Install required packages:

```bash
pip install -r requirements.txt
```

**Key dependencies:**

* `pandas`, `numpy` — core data manipulation
* `pandera` — DataFrame schema validation
* `pytest` — Automated testing
* `logging` — Pipeline logging

---

## Data Validation

### 1. Raw Sales Validation

Script: `validation/validate_raw.py`

* Defines `SaleRecordSchema` using Pandera to enforce:

  * Valid date formats (DD-MM-YYYY or YYYY-MM-DD)
  * Non-negative IDs and prices
  * No duplicate `(date, item_id, shop_id)` rows

Run validation:

```bash
python validation/validate_raw.py datasets/sales_train.csv datasets/cleaned_sales.csv
```

Automated tests with pytest:

```bash
pytest validation/test_validate_raw.py
```

### 2. Feature Set Validation

Script: `validation/validate_features.py`

* Defines `features_schema` using Pandera to enforce types, ranges, and uniqueness on the engineered feature set.

Run validation:

```bash
python validation/validate_features.py
```

---

## Feature Engineering Pipeline

Script: `pipeline/fe_pipeline.py`

1. **Load data** from CSVs
2. **Aggregate** daily sales into monthly counts (`aggregate_monthly`)
3. **Build features** (`build_test_features`):

   * Merge item/shop metadata
   * Compute seasonal, cyclic, lag, rolling, and price features
   * Encode city and shop type
4. **Validate** the generated features using `validate_dataset`
5. **Save** the final feature set to `datasets/fe_df.csv`

Run the pipeline:

```bash
python pipeline/fe_pipeline.py \
  datasets/sales_train.csv \
  datasets/items.csv \
  datasets/item_categories.csv \
  datasets/shops.csv \
  datasets/test_template.csv \
  datasets/fe_df.csv \
  --test_block_num 34
```

---

## Exploratory Expansion Script

File: `notebooks/feature_expansion.py`

* Demonstrates an alternative interactive approach:

  * Merges daily sales with metadata
  * Expands full monthly grid with `pd.MultiIndex`
  * Fills missing prices and sales
  * Engineers features step-by-step (lags, rolling stats, cyclic encoding, extras)
  * Downcasts dtypes to optimize memory
  * Saves checkpoints (`checkpoint.pkl`, `downcasted.pkl`)

To run:

```bash
python notebooks/feature_expansion.py
```

---

## Logging and Monitoring

* Logging is configured via `logging` module in each script.
* Raw validation logs to `validation.log`.
* Pipeline logs to STDOUT and records info/critical errors.

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/name`
3. Commit your changes: \`git commit -m "Add new feature"
4. Push to the branch: `git push origin feature/name`
5. Submit a pull request

Please ensure all tests pass before merging.

---

## License

MIT License. See `LICENSE` file for details.
