**Project Overview**

This repository contains a feature engineering pipeline for a retail sales prediction project. The goal is to process raw daily sales data, validate it, generate monthly aggregated features, validate the feature set, and prepare it for modeling and prediction.

---

## Directory Structure

```text
├── data/                              # All project data
│   ├── raw/                           # Raw input files
│   │   ├── sales_train.csv
│   │   ├── items.csv
│   │   ├── item_categories.csv
│   │   └── shops.csv
│   ├── interim/                       # Cleaned and intermediate data
│   │   ├── cleaned_sales.csv          # After ETL
│   │   ├── checkpoint.pkl
│   │   └── downcasted.pkl
│   ├── processed/                     # Final datasets
│   │   └── fe_df.csv
│   └── external/                      # Test and submission files
│       ├── test.csv
│       └── sample_submission.csv
│
├── src/                               # Source code
│   ├── etl/
│   │   └── etl_pipeline.py            # raw → cleaned
│   ├── fe_pipeline/
│   │   └── fe_pipeline.py             # Feature engineering logic
│   └── validation/
│       ├── schemas/                  
│       │   ├── validation_schema_1.py
│       │   ├── validation_schema_2.py
│       │   └── __init__.py
│       └── validator/
|           └── validator.py
│      
│
├── notebooks/
|   ├── DQC_and_ETL.ipynb
|   ├── EDA.ipynb
│   ├── feature_engineering.ipynb
│   └── modeling.ipynb
│
├── README.md
└── requirements.txt
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
