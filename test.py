from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import lightgbm as lgb

print("Loading data...")
# load or create your dataset

df_train = pd.read_csv("regression.train", header=None, sep="\t")
df_test = pd.read_csv("regression.test", header=None, sep="\t")

y_train = df_train[0]
y_test = df_test[0]
X_train = df_train.drop(0, axis=1)
X_test = df_test.drop(0, axis=1)

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

lgb_train = lgb.Dataset(
    X_train,
    y_train,
    feature_name=[f"f{i + 1}" for i in range(X_train.shape[-1])],
    categorical_feature=[21],
)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {"num_leaves": 5, "metric": ["l1", "l2"], "verbose": -1}

evals_result = {}  # to record eval results for plotting
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_train, lgb_test],
    callbacks=[lgb.log_evaluation(10), lgb.record_evaluation(evals_result)],
)

def render_metric(metric_name):
    lgb.plot_metric(evals_result, metric=metric_name, figsize=(10, 5))
    plt.show()

render_metric(params["metric"][0])

def render_plot_importance(importance_type, max_features=10, ignore_zero=True, precision=3):
    lgb.plot_importance(
        gbm,
        importance_type=importance_type,
        max_num_features=max_features,
        ignore_zero=ignore_zero,
        figsize=(12, 8),
        precision=precision,
    )
    plt.show()

render_plot_importance(importance_type="split")