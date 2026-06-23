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
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score, roc_curve, precision_recall_curve
)

def main():
    # 1. Cluster Workspace Configuration
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Integrated_Cell_Line/genome-wide/ML/"
    train_file = os.path.join(working_dir, "CRISPRi_ML_Train_80.txt")
    
    # Updated Clean Output Configurations
    metrics_output_path = os.path.join(working_dir, "sequence_plus_guide_epi_metrics.txt")
    curves_png_path = os.path.join(working_dir, "sequence_plus_guide_epi_curves.png")
    curves_pdf_path = os.path.join(working_dir, "sequence_plus_guide_epi_curves.pdf")
    shap_png_path = os.path.join(working_dir, "sequence_plus_guide_epi_shap.png")
    shap_pdf_path = os.path.join(working_dir, "sequence_plus_guide_epi_shap.pdf")
    
    if not os.path.exists(train_file):
        print(f"CRITICAL ERROR: Training file missing from {working_dir}")
        sys.exit(1)
        
    print("[*] Ingesting large master matrix layout (~1400 rows)...")
    df_master = pd.read_csv(train_file, sep='\t')
    df_master.columns = [col.lower() for col in df_master.columns]

    # 2. Case-Insensitive Feature Isolation Layer (Selective Epigenetic Ablation)
    explicit_metadata_drops = [
        'unique_sgrna_id', 'id', 'gene', 'sgrna sequence', 
        'cell_line_origin', 'distance_to_tss', 'sigmoid_score', 'class'
    ]
    
    # STRATEGY: Drop gene promoter bins, but explicitly KEEP columns starting with 'guide_'
    gene_level_epigenetic_bins = [
        col for col in df_master.columns 
        if 'bin' in col and not col.startswith('guide_')
    ]
    
    metadata_drops = [col for col in explicit_metadata_drops if col in df_master.columns]
    total_drops = list(set(metadata_drops + gene_level_epigenetic_bins))
    
    # Slice the clean dataset (Numeric features only)
    X_master = df_master.drop(columns=total_drops, errors='ignore').select_dtypes(include=[np.number])
    y_master = df_master['class'].values
    genes_master = df_master['gene'].values

    # Quick diagnostic tracking check
    retained_guide_bins = [col for col in X_master.columns if col.startswith('guide_')]
    print(f"--> Extraction Layer Diagnostics Complete.")
    print(f"    Total Matrix Width: {X_master.shape[1]} features.")
    print(f"    Retained Local Guide-Epi channels: {len(retained_guide_bins)}")

    # 3. Enforce Strict Group-Aware Internal 80/20 Data Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(X_master, y_master, groups=genes_master))
    
    X_train, X_val = X_master.iloc[train_idx], X_master.iloc[val_idx]
    y_train, y_val = y_master[train_idx], y_master[val_idx]
    genes_train = genes_master[train_idx]
    
    print(f"--> Group Split: Train Pool = {X_train.shape[0]} rows | Validation Pool = {X_val.shape[0]} rows")

    # 4. XGBoost Optimization Engine Configuration
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1
    )
    
    # Expanded grid optimized for complex sequence + epigenetic combinations
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [2, 3, 4], 
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'reg_alpha': [0.1, 1, 5],
        'reg_lambda': [1, 5, 10]
    }
    
    cv = GroupKFold(n_splits=5)
    print("[*] Launching parallel GroupKFold Randomized Search (40 Iterations)...")
    random_search = RandomizedSearchCV(
        estimator=xgb_clf, 
        param_distributions=param_grid, 
        n_iter=40, 
        scoring='roc_auc', 
        cv=cv, 
        n_jobs=-1, 
        random_state=42
    )
    random_search.fit(X_train, y_train, groups=genes_train)
    best_model = random_search.best_estimator_

    # 5. Evaluate Performance on the Internal Unseen Validation Group
    print("[*] Running inference on validation split...")
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

    # 6. Save Configuration Metrics File
    print(f"--> Writing performance matrix to: {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        f.write("=== Sequence + Guide Epigenetics Model Metrics (Internal 80/20 Split) ===\n")
        f.write(f"Internal Training Matrix Size: {X_train.shape[0]} rows\n")
        f.write(f"Internal Validation Matrix Size: {X_val.shape[0]} rows\n")
        f.write(f"Total Combined Features Utilized: {X_train.shape[1]}\n")
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
    ax1.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Seq + Guide Epi Model (AUC = {roc_auc:.3f})')
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
    ax2.plot(recall, precision, color='dodgerblue', lw=2.5, label=f'Seq + Guide Epi Model (PR-AUC = {pr_auc:.3f})')
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
    
    plt.title("Sequence + Guide Epigenetic Importance (TreeSHAP Beeswarm)", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    plt.savefig(shap_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(shap_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig_shap)
    
    print(f"[-->] Metrics report output to: {metrics_output_path}")
    print(f"[-->] Evaluation curves exported to: {curves_png_path} and .pdf")
    print(f"[-->] SHAP importance plots exported to: {shap_png_path} and .pdf")
    print("[*] Pipeline complete.")

if __name__ == "__main__":
    main()
