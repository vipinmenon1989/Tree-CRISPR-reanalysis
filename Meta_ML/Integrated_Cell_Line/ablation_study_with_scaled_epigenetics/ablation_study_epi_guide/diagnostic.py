import os
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# Load datasets
df_k562 = pd.read_csv("/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Complete_dataset_sigmoid_score_K562/CRISPR_ml_features_final.csv")
df_a549 = pd.read_csv("CRISPR_ml_features_independent_A549.csv")

# Clean A549 columns exactly like the execution run
df_a549 = df_a549.rename(columns=lambda x: x.replace('A549', 'K562'))
df_a549 = df_a549.rename(columns=lambda x: x.replace('guide_ATAC_seq_bin_', 'guide_K562_ATAC_seq_bin_'))

# Load Model
model = xgb.XGBClassifier()
model.load_model("guide_epi_model.json")
features = model.get_booster().feature_names

print("=== DIAGNOSTIC 1: FEATURE DISTRIBUTION CHECK ===")
# Check if the mean and variance of top features match between cell lines
for f in features[:3] + [feat for feat in features if 'bin' in feat][:3]:
    k_mean = df_k562[f].mean() if f in df_k562.columns else df_k562[f.replace('K562', 'K562')].mean() # fallbacks if raw training named differently
    # matching the mapped name in a549
    a_mean = df_a549[f].mean() if f in df_a549.columns else np.nan
    print(f"Feature: {f:<45} | K562 Mean: {k_mean:.3f} | A549 Mean: {a_mean:.3f}")

print("\n=== DIAGNOSTIC 2: PREDICTION BIAS CHECK ===")
X_test_a549 = df_a549[features].select_dtypes(include=[np.number])
y_proba = model.predict_proba(X_test_a549)[:, 1]

print(f"A549 Predicted Class 1 Probability - Mean: {y_proba.mean():.4f} | Min: {y_proba.min():.4f} | Max: {y_proba.max():.4f}")
print(f"A549 Ground Truth Class 1 Ratio:     {((df_a549['Sigmoid_Score'] > 0.25).astype(int)).mean():.4f}")
