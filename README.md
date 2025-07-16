# Retail Sales Forecasting - Feature Engineering Pipeline

## Project Overview
This repository contains a feature engineering pipeline for retail sales prediction. The pipeline processes raw daily sales data, validates input, generates monthly aggregated features, validates the output, and prepares data for modeling.

## Updated Directory Structure
├── datasets/ # Data directories
│ ├── raw/ # Raw input files
│ │ ├── sales_train.csv
│ │ ├── items.csv
│ │ ├── item_categories.csv
│ │ └── shops.csv
│ ├── interim/ # Processed intermediate data
│ │ ├── cleaned_sales.csv
│ │ ├── checkpoint.pkl
│ │ └── downcasted.pkl
│ ├── processed/ # Final output
│ │ └── fe_df.csv
│ └── external/ # Test data
│ ├── test.csv
│ └── sample_submission.csv
│
├── src/
│ ├── sales_forecasting/ # Core pipeline code
│ │ ├── data/ # ETL components
│ │ │ └── etl_pipeline.py
│ │ ├── feature/ # Feature engineering
│ │ │ └── fe_pipeline.py
│ │ └── validation/ # Validation logic
│ │ ├── schemas/
│ │ │ ├── validation_schema_1.py
│ │ │ ├── validation_schema_2.py
│ │ │ └── init.py
│ │ └── validator.py
│ │
│ └── scripts/ # Execution scripts
│ └── run_pipeline.py # Main pipeline runner
│
├── notebooks/ # Jupyter notebooks
│ ├── DQC_and_ETL.ipynb # Data quality checks
│ ├── EDA.ipynb # Exploratory analysis
│ └── feature_engineering.ipynb # Feature development
│
├── README.md # This file
└── requirements.txt # Python dependencies


## Key Components

### 1. Data Processing (`src/data/etl_pipeline.py`)
- Cleans and preprocesses raw sales data
- Handles missing values and data type conversions
- Outputs cleaned data to `data/interim/`

### 2. Feature Engineering (`src/feature/fe_pipeline.py`)
- Aggregates daily sales to monthly level
- Generates features:
  - Lag features (1-12 month lags)
  - Rolling statistics (3-12 month windows)
  - Price features (avg, min, max)
  - Shop/item metadata merges
  - Temporal features (month, year)
- Outputs final feature set to `data/processed/fe_df.csv`

### 3. Data Validation (`src/validation/validation_schemas/validation.py`)
- Schema validation using Pandera
- Raw data validation:
  - Date format consistency
  - Non-negative prices/quantities
  - ID validity checks
- Feature set validation:
  - Range checks for numerical features
  - Category validations
  - Missing value checks

### 4. Pipeline Runner (`src/scripts/run_pipeline.py`)
- Orchestrates full workflow:
  1. Run ETL pipeline
  2. Validate input for feature engineering
  3. Execute feature engineering
  4. Validate outputs
- Example usage:
  ```bash
  python src/scripts/run_pipeline.py

Getting Started
Prerequisites
Python 3.8+

Dependencies: pip install -r requirements.txt

Key Dependencies
pandas
numpy
pandera
scikit-learn
python-dateutil
Running Pipeline
bash
# Run full pipeline:
python src/scripts/run_pipeline.py

# Run individual components:
python src/sales_forecasting/data/etl_pipeline.py
python src/sales_forecasting/feature/fe_pipeline.py
Pipeline Consistency	Input/output shape validation	fe_pipeline.py
Contributing
Create feature branch: git checkout -b feature/new-feature

Add tests for new functionality