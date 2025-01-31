import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import time
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set parameters for both models
params_baseline = {
    'boosting_type': 'gbdt',
    'enable_bundle': False,        # Disable EFB
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 1.0,
    'random_state': 42,
    'verbose': -1
}

params_efb_goss = {
    'boosting_type': 'goss',       # Enable GOSS
    'enable_bundle': True,         # Enable EFB
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 1.0,
    'random_state': 42,
    'verbose': -1
}

# Create models
model_baseline = lgb.LGBMClassifier(**params_baseline)
model_efb_goss = lgb.LGBMClassifier(**params_efb_goss)

print(model_baseline.get_params())

print(model_efb_goss.get_params())


# Dictionary to store evaluation results
evals_result_baseline = {}
evals_result_efb_goss = {}

# Train baseline model with timing
print("Training baseline model...")
start_time_baseline = time.time()
model_baseline.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric='auc',
                  callbacks=[lgb.record_evaluation(evals_result_baseline)])
training_time_baseline = time.time() - start_time_baseline

# Train EFB+GOSS model with timing
print("\nTraining EFB+GOSS model...")
start_time_efb_goss = time.time()
model_efb_goss.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric='auc',
                  callbacks=[lgb.record_evaluation(evals_result_efb_goss)])
training_time_efb_goss = time.time() - start_time_efb_goss

# Extract AUC values
auc_baseline = evals_result_baseline['valid_0']['auc']
auc_efb_goss = evals_result_efb_goss['valid_0']['auc']

# Generate time points for each iteration
n_estimators = params_baseline['n_estimators']
time_per_iter_baseline = training_time_baseline / n_estimators
time_per_iter_efb_goss = training_time_efb_goss / n_estimators

time_points_baseline = [time_per_iter_baseline * (i+1) for i in range(n_estimators)]
time_points_efb_goss = [time_per_iter_efb_goss * (i+1) for i in range(n_estimators)]

# Ensure both models have the same total training time
max_training_time = max(training_time_baseline, training_time_efb_goss)

# Pad the model that finishes earlier
if training_time_baseline < training_time_efb_goss:
    last_auc_baseline = auc_baseline[-1]
    extra_steps = int((max_training_time - training_time_baseline) / time_per_iter_baseline)
    time_points_baseline += [max_training_time] * extra_steps
    auc_baseline += [last_auc_baseline] * extra_steps
elif training_time_efb_goss < training_time_baseline:
    last_auc_efb_goss = auc_efb_goss[-1]
    extra_steps = int((max_training_time - training_time_efb_goss) / time_per_iter_efb_goss)
    time_points_efb_goss += [max_training_time] * extra_steps
    auc_efb_goss += [last_auc_efb_goss] * extra_steps

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(time_points_baseline, auc_baseline, label='lgb_baseline', color='orange')
plt.plot(time_points_efb_goss, auc_efb_goss, label='LightGBM', color='blue')
plt.xlabel('Training Time (seconds)')
plt.ylabel('Test Set AUC')
plt.title('AUC Progression During Training')
plt.legend()
plt.grid(True)
plt.show()
# plt.savefig("lightgbm.png")


# Evaluate final models
def evaluate_model(name, model):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print(f"Training Time: {training_time_baseline if name == 'Baseline' else training_time_efb_goss:.2f}s")

evaluate_model("Baseline", model_baseline)
evaluate_model("EFB+GOSS", model_efb_goss)
