# Retail Sales Forecasting Pipeline

## Project Overview

`kaggle_comp_future_sales_forecasting` is a Python package and toolkit for retail sales prediction, covering data preprocessing, validation, feature engineering, model training, and inference. It was developed as part of a Kaggle competition.

---

## Directory Structure

```text
sales_prediction/                   # Project root
├── datasets/                       # Data directories
│   ├── raw/                        # Raw input files
│   │   ├── sales_train.csv
│   │   ├── items.csv
│   │   ├── item_categories.csv
│   │   └── shops.csv
│   ├── interim/                    # Intermediate processed data
│   │   ├── cleaned_sales.csv
│   │   ├── checkpoint.pkl
│   │   └── downcasted.pkl
│   ├── processed/                  # Final feature outputs
│   │   └── fe_df.parquet
│   └── external/                   # Test and submission data
│       ├── test.csv
│       └── sample_submission.csv
│       └── predictions.parquet
│
├── src/                            # Source code
│   ├── kaggle_comp_future_sales_forecasting/  # Package modules
│   │   ├── __init__.py             # Exposes key classes/functions
│   │   ├── data/                   # ETL components
│   │   │   └── etl_pipeline.py
│   │   ├── feature/                # Feature engineering
│   │   │   └── fe_pipeline.py
│   │   ├── modeling/               # Model training and inference
│   │   │   └── trainer.py
│   │   ├── validation/             # Data validation
│   │   │   ├── validator.py
│   │   │   └── validation_schemas/
│   │   │       ├── raw_schema.py
│   │   │       └── fe_schema.py
│   │   └── utils/                  # Utility modules
│   │       └── logger.py
│   └── scripts/                    # Command-line runners
│       ├── run_pipeline.py        # Full pipeline execution
│       └── run_inference.py       # Inference runner
│
├── notebooks/                      # Jupyter notebooks
│   ├── DQC_and_ETL.ipynb           # Data quality checks & ETL
│   ├── EDA.ipynb                   # Exploratory data analysis
│   └── feature_engineering.ipynb   # Feature development
│
├── LICENSE                         # MIT License
├── README.md                       # This file
└── requirements.txt                # Package dependencies
```

---

## Key Components

1. **Data Processing** (`kaggle_comp_future_sales_forecasting.data.etl_pipeline.ETLPipeline`)

   * Cleans and preprocesses raw sales data
   * Handles missing values, type conversions, downcasting
   * Saves interim outputs to `datasets/interim/`

2. **Feature Engineering** (`kaggle_comp_future_sales_forecasting.feature.fe_pipeline.FeaturePipeline`)

   * Aggregates daily sales into monthly features
   * Generates:

     * Lag features (1–12 months)
     * Rolling statistics (3–12 month windows)
     * Price-based features (average, min, max)
     * Temporal features (month, year)
   * Saves final features to `datasets/processed/`

3. **Validation** (`kaggle_comp_future_sales_forecasting.validation.validator.Validator`)

   * Uses Pandera schemas for:

     * Raw data validation (`raw_schema.py`)
     * Feature data validation (`fe_schema.py`)
   * Ensures consistency, non-negative values, required columns

4. **Model Training & Inference** (`kaggle_comp_future_sales_forecasting.modeling.trainer.SalesPredictor`)

   * Trains LightGBM or XGBoost models on engineered features
   * Saves trained model to `models/lgb_model.pkl`
   * Inference runner loads model and generates submission files

5. **Scripts** (`src/scripts/`)

   * `run_pipeline.py`: orchestrates ETL → validation → feature engineering → validation → model training
   * `run_inference.py`: loads saved features and model to produce predictions

---

## Installation

### From PyPI

```bash
pip install kaggle_comp_future_sales_forecasting
```

### From Source

```bash
git clone https://github.com/200kgsquat/sales_prediction.git
cd sales_prediction
pip install .
```

---

## Usage

```python
from kaggle_comp_future_sales_forecasting import (
    ETLPipeline,
    FeaturePipeline,
    Validator,
    SalesPredictor,
    sale_schema,
    features_schema
)

# 1. Run full pipeline and train model
paths = ETLPipeline.setup_paths()
ETLPipeline.run_pipeline(*paths)

# 2. Run inference
from kaggle_comp_future_sales_forecasting.scripts.run_inference import run_inference_pipeline
submission_df = run_inference_pipeline()
```

---

## Dependencies

See `requirements.txt`. Key requirements:

* Python >= 3.9
* pandas
* numpy
* scikit-learn (>=1.2,<1.8)
* xgboost (>=1.6.2)
* pandera (>=0.10.0)
* pyarrow (<=20.0.0)
* tqdm
* joblib
* typing-inspect

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please include tests and update documentation accordingly.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
