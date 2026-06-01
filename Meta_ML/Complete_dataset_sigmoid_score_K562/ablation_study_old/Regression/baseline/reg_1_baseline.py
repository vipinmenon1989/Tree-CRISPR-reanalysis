import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('CRISPR_ml_features_final.csv')
y = df['Sigmoid_Score']

# 1. Drop strictly non-predictive metadata
cols_to_drop = ['ID', 'Sigmoid_Score', 'class', 'Gene', 'sgRNA Sequence', 'Chromosome', 'Strand', 'PAM', 'extended_sequence', 'closest_TSS_coord']
X_base = df.drop(columns=cols_to_drop, errors='ignore')

# 2. Dynamically drop any remaining string/object columns to protect spatial features
X_base = X_base.select_dtypes(exclude=['object'])

# SCRIPT 1 Feature Logic
epi_cols = [col for col in X_base.columns if 'K562' in col or 'H3K' in col or 'guide_' in col]
X = X_base.drop(columns=epi_cols)
print(f"Feature space size (Baseline): {X.shape[1]}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize XGBRegressor
xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)

# Hyperparameter Grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8],
    'colsample_bytree': [0.3, 0.5, 0.7],
    'reg_alpha': [0, 0.1, 1, 10],
    'reg_lambda': [1, 5, 10]
}

# CV setup
cv = KFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(
    estimator=xgb_reg,
    param_distributions=param_grid,
    n_iter=25,
    scoring='neg_mean_squared_error',
    cv=cv,
    random_state=42,
    n_jobs=-1
)

print("Fitting RandomizedSearchCV on training data...")
random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_
print("Fitting completed.")

# Save the model
best_model.save_model("reg_1_baseline_model.json")

# Predictions
y_pred = best_model.predict(X_test)

# Calculate metrics
spearman_corr, _ = spearmanr(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Spearman Correlation: {spearman_corr:.6f}")
print(f"Mean Absolute Error: {mae:.6f}")
print(f"Mean Squared Error: {mse:.6f}")
print(f"R-squared: {r2:.6f}")

# Save metrics to text file
with open("reg_1_baseline_metrics.txt", "w") as f:
    f.write("=== Baseline Regressor Metrics ===\n")
    f.write(f"Spearman Correlation: {spearman_corr:.6f}\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.6f}\n")
    f.write(f"Mean Squared Error (MSE): {mse:.6f}\n")
    f.write(f"R-squared: {r2:.6f}\n\n")
    f.write("=== Best Parameters ===\n")
    for param, val in random_search.best_params_.items():
        f.write(f"{param}: {val}\n")

print("Done!")
