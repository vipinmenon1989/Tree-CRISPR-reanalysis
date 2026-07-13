import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

# 1. Load the classified dataset
file_path = "gene_aggregated_classified.txt"
df = pd.read_csv(file_path, sep="\t")

# 2. Drop intermediate class (-1) to lock in strict binary classification
df_model = df[df["class"] != -1].reset_index(drop=True)

# 3. Isolate features and target
feature_cols = [col for col in df_model.columns if col.startswith("mean_guide_")]
X = df_model[feature_cols]
y = df_model["class"]

# 4. Calculate the precise scale_pos_weight factor to correct the 1.47:1 asymmetry
class_counts = y.value_counts()
neg_count = class_counts.get(0)
pos_count = class_counts.get(1)
calculated_weight = neg_count / pos_count

# 5. Define the hyperparameter search space
param_dist = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

# 6. Initialize base XGBoost Classifier
xgb_base = xgb.XGBClassifier(
    scale_pos_weight=calculated_weight,
    eval_metric="logloss",
    random_state=42,
    use_label_encoder=False,
)

# 7. Configure RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_dist,
    n_iter=25,
    scoring="roc_auc",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

# 8. Train and extract the optimized production model
print("Executing hyperparameter search across 5 stratified folds...")
random_search.fit(X, y)
best_model = random_search.best_estimator_
print(f"Optimal Out-of-Fold ROC-AUC: {random_search.best_score_:.4f}\n")

# =====================================================================
#                          SHAP ANALYSIS
# =====================================================================
print("Initializing SHAP TreeExplainer...")

# TreeExplainer calculates exact SHAP values efficiently for tree ensembles
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X)

# 9. Extract Global Quantitative Feature Rankings
# Compute the mean absolute SHAP value across all samples for each feature
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
shap_importance = pd.DataFrame(
    {"Feature": X.columns, "Mean_Absolute_SHAP": mean_abs_shap}
).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)

print("==================================================")
print("          TOP 10 MOST CONTRIBUTING FEATURES       ")
print("==================================================")
print(shap_importance.head(10).to_string(index=False))
print("==================================================")

# 10. Generate and Export Visualizations
print("\nExporting diagnostic plots...")

# Plot A: Global Feature Importance Bar Plot (Magnitude Only)
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("Global Feature Importance Matrix (Mean |SHAP Value|)", pad=15)
plt.savefig("shap_global_importance_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot B: Summary Beeswarm Plot (Shows Magnitude AND Directionality)
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Diagnostic Beeswarm Plot (Directional Impact)", pad=15)
plt.savefig("shap_diagnostic_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()

print("Execution complete. Visualizations saved locally.")
