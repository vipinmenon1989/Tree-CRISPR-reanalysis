import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score, roc_curve, precision_recall_curve
)

# 1. Load Data
df = pd.read_csv("Holdout_with_SSC_Scores.csv")

# 2. Metrics Calculation
y_true = df['class']
y_scores = df['SSC']

# Binary prediction based on 0.0 threshold (standard for many scoring tools)
y_pred = (y_scores >= 0.0).astype(int)

roc_auc = roc_auc_score(y_true, y_scores)
pr_auc = average_precision_score(y_true, y_scores)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"ROC-AUC: {roc_auc:.4f}, AUPR: {pr_auc:.4f}, F1: {f1:.4f}")

# ======================================================================
# 3. Save Metrics to Disk
# ======================================================================
metrics_dict = {
    'ROC_AUC': [roc_auc],
    'AUPR': [pr_auc],
    'Precision': [prec],
    'Recall': [rec],
    'F1': [f1]
}
metrics_df = pd.DataFrame(metrics_dict)
metrics_out_path = "SSC_Evaluation_Metrics.csv"
metrics_df.to_csv(metrics_out_path, index=False)
print(f"[✔] Metrics successfully saved to -> {metrics_out_path}")

# ======================================================================
# 4. Plotting Curves
# ======================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
ax1.plot(fpr, tpr, label=f'SSC (AUC={roc_auc:.2f})', color='crimson')
ax1.plot([0, 1], [0, 1], 'k--')
ax1.set_title('ROC Curve')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend()

# PR Curve
precision, recall, _ = precision_recall_curve(y_true, y_scores)
ax2.plot(recall, precision, label=f'SSC (AUPR={pr_auc:.2f})', color='dodgerblue')
ax2.set_title('Precision-Recall Curve')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.legend()

plt.tight_layout()
plt.savefig("SSC_Evaluation_Curves.png")
print("[✔] Evaluation curves saved to -> SSC_Evaluation_Curves.png")
