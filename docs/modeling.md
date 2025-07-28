# 🛒 Sales Forecasting with Feature Engineering and ML

This project performs monthly item sales forecasting using advanced feature engineering and machine learning models (Lasso, LightGBM), evaluated against the [Kaggle "Predict Future Sales"](https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales) dataset.

---

## 📁 Dataset

- `features.parquet`: Pre-engineered training data
- `test.parquet`: Test set matching the format required for Kaggle submission
- Source columns include:
  - `date_block_num`, `shop_id`, `item_id`
  - Engineered lags, categorical encodings, seasonality, etc.

---

## 📊 Baseline Model — Lag Feature

A naive baseline using the previous month's sales:

```python
y_pred_lag1 = test['item_cnt_month_lag_1']
Metric: RMSE on test split (month 33)

Submission prepared by aligning with the test set and clipping predictions to [0, 20].

⚙️ Feature Preprocessing
🔁 Categorical Encoding
city, type_of_shop: Encoded using TargetEncoder and LabelEncoder

season: One-hot encoded

from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
🧹 Column Cleaning
Drops columns containing "name" substrings

Ensures feature consistency across training and test/validation datasets

Adds missing one-hot columns in validation as 0

📐 Model 1 — Lasso Regression (Public score: 1.70522, Private score: 1.73285)
🔧 Pipeline
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(max_iter=10000))
])
Tuned using GridSearchCV over alpha

TimeSeries cross-validation (TimeSeriesSplit)

Feature matrix matched exactly between X_train, X_test, and validation

Clipped final predictions to [0, 20]

🔢 Output
Submission saved as lasso_sub.csv

⚡ Model 2 — LightGBM (Private score: 1.34047, Public score: 1.34852)
⚙️ Setup
Handled label encoding for categorical variables

Dropped low-variance columns (constant values)

Clipped final predictions to [0, 20]

🧪 Training
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'max_depth': 6,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42,
    'verbosity': -1  
}
Early stopping with validation set (date_block_num == 33)

Submission saved as lgb_sub.csv

🔍 SHAP-Based Feature Selection
Used shap.TreeExplainer for model interpretability

Selected Top 20 most important features using mean absolute SHAP values

Retrained model on selected features

selected_features = shap_summary.head(TOP_N)['feature'].tolist()
📤 Submissions
subm_lag1.csv: Baseline lag model

lasso_sub.csv: Lasso regression predictions

lgb_sub.csv: LightGBM predictions with SHAP-selected features

🧠 Key Learnings
Lag features provide a powerful signal

Tree-based models + SHAP give strong performance and interpretability (Private score: 1.32675, Public Score: 1.33971)

Feature alignment between datasets is critical for generalization

