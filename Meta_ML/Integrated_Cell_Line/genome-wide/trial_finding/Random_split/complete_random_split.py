import os
import sys

# ======================================================================
# FORCE HEADLESS HPC RENDER BACKEND (MUST EXECUTE BEFORE PYPLOT INGEST)
# ======================================================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# ======================================================================

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score, roc_curve, precision_recall_curve
)

def main():
    # 1. Cluster Workspace Configuration
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Integrated_Cell_Line/genome-wide/"
    train_file = os.path.join(working_dir, "CRISPR_ml_features_final_K562.csv")
    
    # Separate outputs specifically for the random split diagnostic run
    metrics_output_path = os.path.join(working_dir, "random_split_model_metrics.txt")
    curves_png_path = os.path.join(working_dir, "random_split_model_curves.png")
    curves_pdf_path = os.path.join(working_dir, "random_split_model_curves.pdf")
    shap_png_path = os.path.join(working_dir, "random_split_model_shap.png")
    shap_pdf_path = os.path.join(working_dir, "random_split_model_shap.pdf")
    
    if not os.path.exists(train_file):
        print(f"CRITICAL ERROR: Training file missing from {working_dir}")
        sys.exit(1)
        
    print("[*] Ingesting K562 matrix for random row split diagnostic...")
    df_master = pd.read_csv(train_file, sep=',')
    df_master.columns = [col.lower() for col in df_master.columns]

    # 2. Multi-Modality Feature Activation Layer
    explicit_metadata_drops = [
        'unique_sgrna_id', 'id', 'gene', 'sgrna sequence', 
        'cell_line_origin', 'sigmoid_score', 'class'
    ]
    cols_to_drop = [col for col in explicit_metadata_drops if col in df_master.columns]
    
    X_master = df_master.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])
    y_master = df_master['class'].values

    print(f"--> Feature Matrix Isolated: {X_master.shape[0]} rows x {X_master.shape[1]} integrated features.")
    
    # Validation logging
    has_tss = 'distance_to_tss' in X_master.columns
    bin_count = sum(1 for c in X_master.columns if 'bin' in c)
    print(f"    -> distance_to_tss status: {'RETAINED' if has_tss else 'MISSING'}")
    print(f"    -> Epigenetic chromatin bins tracked: {bin_count}")

    # 3. Enforce Standard Pure Random Row-Level Split (NO GROUPING CONSTRAINTS)
    # This permits potentially identical sequence contexts to fall on both sides of the wall
    X_train, X_val, y_train, y_val = train_test_split(
        X_master, y_master, test_size=0.20, stratify=y_master, random_state=42
    )
    
    print(f"--> WARNING: Running Pure Random Split. Train Pool = {X_train.shape[0]} rows | Validation Pool = {X_val.shape[0]} rows")

    # 4. XGBoost Regularization Engine Configuration
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1
    )
    
    # Retaining the same search constraints to make the comparison mathematically direct
    param_grid = {
        'n_estimators': [30, 50, 75],
        'max_depth': [2, 3], 
        'learning_rate': [0.01, 0.02, 0.03],
        'subsample': [0.6, 0.7],
        'colsample_bytree': [0.2, 0.3, 0.4],
        'reg_alpha': [5, 10, 15],
        'reg_lambda': [10, 15, 20]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("[*] Tuning hyperparameters via random-stratified inner cross-validation...")
    random_search = RandomizedSearchCV(
        estimator=xgb_clf, 
        param_distributions=param_grid, 
        n_iter=30, 
        scoring='roc_auc', 
        cv=cv, 
        n_jobs=-1, 
        random_state=42
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    # 5. Evaluate Performance on the Validation Subset
    print("[*] Running inference on the randomly isolated validation subset...")
    y_pred = best_model.predict(X_val)
    y_proba = best_model.predict_proba(X_val)[:, 1]
    
    roc_auc = roc_auc_score(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)
    
    metrics = {
        "Validation ROC-AUC": roc_auc,
        "Validation PR-AUC (Average Precision)": pr_auc,
        "Validation Precision": precision_score(y_val, y_pred, zero_division=0),
        "Validation Recall": recall_score(y_val, y_pred, zero_division=0),
        "Validation F1 Score": f1_score(y_val, y_pred, zero_division=0)
    }

    # 6. Save Performance Metrics Log
    print(f"--> Writing performance log to: {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        f.write("=== Full Multi-Modality Model Metrics (Pure Random Split Diagnostic) ===\n")
        f.write(f"Internal Training Pool Size: {X_train.shape[0]} rows\n")
        f.write(f"Internal Validation Pool Size: {X_val.shape[0]} rows\n")
        f.write(f"Total Combined Features Retained: {X_train.shape[1]}\n")
        f.write("-" * 65 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("-" * 65 + "\n")
        f.write("\n=== Optimized Training Hyperparameters ===\n")
        for k, v in random_search.best_params_.items():
            f.write(f"{k}: {v}\n")

    # 7. Generate Performance Curves (ROC & PRC)
    print("[*] Rendering evaluation curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: ROC Curve
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    ax1.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Random Split Model (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=11, fontweight='bold')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # Plot 2: PRC Curve
    precision, recall, _ = precision_recall_curve(y_val, y_proba)
    baseline_pr = np.sum(y_val == 1) / len(y_val) if len(y_val) > 0 else 0.5
    ax2.plot(recall, precision, color='dodgerblue', lw=2.5, label=f'Random Split Model (PR-AUC = {pr_auc:.3f})')
    ax2.axhline(y=baseline_pr, color='gray', lw=1.5, linestyle='--', label=f'Baseline Class Ratio (PR = {baseline_pr:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
    ax2.set_title('Precision-Recall Curve (PRC)', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc="lower left", frameon=True, facecolor='white', edgecolor='none')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(curves_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(curves_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

    # 8. Compute and Plot SHAP Values
    print("[*] Generating TreeSHAP Feature Importance Summary...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_train)
    
    fig_shap = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, max_display=20, show=False)
    
    plt.title("Random Split Model Feature Importance (TreeSHAP)", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    plt.savefig(shap_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(shap_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig_shap)
    
    print(f"[-->] Random split diagnostic run complete. Logs written to: {metrics_output_path}")

if __name__ == "__main__":
    main()
