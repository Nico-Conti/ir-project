import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
data = pd.read_csv("house_prices_data/train.csv")

# Target variable
y = data['SalePrice']

# Drop target and ID columns from features
X = data.drop(columns=['SalePrice', 'Id'], errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify categorical features
categorical_features = X_train.select_dtypes(include=['object']).columns

# Convert categorical features to 'category' dtype in both training and testing sets
for col in categorical_features:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Handle missing values consistently (using the same strategy for both train and test)
# For numerical features, a simple approach is to fill with the mean of the training data
numerical_features = X_train.select_dtypes(include=np.number).columns
for col in numerical_features:
    mean_val = X_train[col].mean()
    X_train[col].fillna(mean_val, inplace=True)
    X_test[col].fillna(mean_val, inplace=True)

# For categorical features, filling with the most frequent category from the training data
for col in categorical_features:
    most_frequent = X_train[col].mode()[0]
    X_train[col].fillna(most_frequent, inplace=True)
    X_test[col].fillna(most_frequent, inplace=True)

