import pandas as pd
import time
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the training and test data
train_df = pd.read_csv("house_prices_data/train.csv")
test_df = pd.read_csv("house_prices_data/test.csv")

# Separate features and target variable
X_train_all = train_df.drop("SalePrice", axis=1).select_dtypes(include='number').fillna(0)
y_train = train_df["SalePrice"]
X_test_all = test_df.select_dtypes(include='number').fillna(0)

# Identify common numerical columns
common_cols = list(set(X_train_all.columns) & set(X_test_all.columns))

# Select only the common columns for training and testing
X_train = X_train_all[common_cols]
X_test = X_test_all[common_cols]

# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# LightGBM Model
lgb_train_data = lgb.Dataset(X_train_split, y_train_split)
lgb_val_data = lgb.Dataset(X_val_split, y_val_split, reference=lgb_train_data)
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1 # Suppress LightGBM output
}

start_time = time.time()
lgb_model = lgb.train(lgb_params, lgb_train_data, num_boost_round=100, valid_sets=[lgb_train_data, lgb_val_data])
lgb_train_time = time.time() - start_time

start_time = time.time()
lgb_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
lgb_pred_time = time.time() - start_time

# XGBoost Model
xgb_train_data = xgb.DMatrix(X_train_split, label=y_train_split)
xgb_val_data = xgb.DMatrix(X_val_split, label=y_val_split)
xgb_test_data = xgb.DMatrix(X_test)
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'booster': 'gbtree',
    'n_estimators': 100,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'verbosity': 0 # Suppress XGBoost output
}

start_time = time.time()
xgb_model = xgb.train(xgb_params, xgb_train_data, num_boost_round=100, evals=[(xgb_train_data, 'train'), (xgb_val_data, 'val')], verbose_eval=False)
xgb_train_time = time.time() - start_time

start_time = time.time()
xgb_pred = xgb_model.predict(xgb_test_data, iteration_range=(0, xgb_model.best_iteration))
xgb_pred_time = time.time() - start_time

# Evaluate models on validation set
lgb_val_pred = lgb_model.predict(X_val_split, num_iteration=lgb_model.best_iteration)
xgb_val_pred = xgb_model.predict(xgb_val_data, iteration_range=(0, xgb_model.best_iteration))

lgb_rmse = mean_squared_error(y_val_split, lgb_val_pred, squared=False)
xgb_rmse = mean_squared_error(y_val_split, xgb_val_pred, squared=False)

# Print results
print("LightGBM Training Time: {:.4f} seconds".format(lgb_train_time))
print("LightGBM Prediction Time: {:.4f} seconds".format(lgb_pred_time))
print("LightGBM RMSE: {:.4f}".format(lgb_rmse))
print("XGBoost Training Time: {:.4f} seconds".format(xgb_train_time))
print("XGBoost Prediction Time: {:.4f} seconds".format(xgb_pred_time))
print("XGBoost RMSE: {:.4f}".format(xgb_rmse))