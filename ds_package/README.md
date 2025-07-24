
# kaggle_comp_future_sales_forecasting

A project for future sales prediction, data preprocessing, and model training.

---

## Description

This package provides tools and pipelines for:

- Data preprocessing and validation  
- Training machine learning models (XGBoost, scikit-learn, etc.)  
- Forecasting future sales based on historical data  
- Organizing ETL processes and feature engineering  

The project was created as part of a Kaggle competition on sales forecasting.

---

## Installation

Install from PyPI:

```bash
pip install kaggle_comp_future_sales_forecasting

Or install from source:

git clone https://github.com/200kgsquat/sales_prediction.git
cd sales_prediction/ds_package
pip install .
Usage

from etl_pipeline import ETLPipeline
from trainer import Trainer

# Run data preprocessing
etl = ETLPipeline()
etl.run()

# Train the model
trainer = Trainer()
trainer.train()
For more details, see the documentation in the docs/ folder or inside the code.

Dependencies
Python >= 3.9

scikit-learn >= 1.2, < 1.8

pandas >= 1.5, < 2.3

numpy >= 1.21, < 2.3

pandera >= 0.10.0

tqdm == 4.67.1

joblib == 1.5.1

xgboost >= 1.6.2

pyarrow <= 20.0.0

typing-inspect >= 0.6.0

License
MIT License. See LICENSE for details.

Contact
Author: Pashayyur@gmail.com.com
Repository: https://github.com/200kgsquat/sales_prediction