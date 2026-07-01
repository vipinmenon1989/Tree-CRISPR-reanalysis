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
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score, roc_curve, precision_recall_curve
)

def main():
    # 1. Cluster Workspace Configuration (Isolated Equation Space for Unseen Genes)
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Integrated_Cell_Line/Isolated_equation/unseen_gene/"
    
    # Reading from the designated gene holdout base script path
    train_file = os.path.join(working_dir, "CRISPRi_ML_Train_80_unseen_genes.txt")
    
    # Path configurations distinctly named to reflect the unseen gene data layout
    model_output_path = os.path.join(working_dir, "unseen_gene_multimodality_model.json")
    metrics_output_path = os.path.join(working_dir, "unseen_gene_multimodality_metrics.txt")
    curves_png_path = os.path.join(working_dir, "unseen_gene_multimodality_curves.png")
    curves_pdf_path = os.path.join(working_dir, "unseen_gene_multimodality_curves.pdf")
    shap_png_path = os.path.join(working_dir, "unseen_gene_multimodality_shap.png")
    shap_pdf_path = os.path.join(working_dir, "unseen_gene_multimodality_shap.pdf")
    
    if not os.path.exists(train_file):
        print(f"CRITICAL ERROR: Training file missing from {working_dir}")
        sys.exit(1)
        
    print("[*] Ingesting complete master data matrix...")
    df_master = pd.read_csv(train_file, sep='\t')
    df_master.columns = [col.lower() for col in df_master.columns]

    # Target definition from stable sigmoid score thresholding
    df_master['class'] = (df_master['sigmoid_score'] > 0.25).astype(int)

    # Verify presence of individual gene tracking array before dropping metadata
    if 'gene' not in df_master.columns:
        print("CRITICAL ERROR: Column 'gene' missing from input dataset header!")
        sys.exit(1)
    
    # Extract structural grouping metadata array to isolate entire loci
    genes_master = df_master['gene'].values

    # ==================================================================
    # 2. FULL MULTI-MODALITY FEATURE ISOLATION (NO ABLATION)
    # ==================================================================
    explicit_metadata_drops = [
        'unique_sgrna_id', 'id', 'gene', 'sgrna sequence', 
        'cell_line_origin', 'distance_to_tss', 'sigmoid_score', 'class'
    ]
    cols_to_drop = [col for col in explicit_metadata_drops if col in df_master.columns]
    
    X_master = df_master.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])
    y_master = df_master['class'].values

    print(f"--> Full Feature Matrix Isolated: {X_master.shape[0]} rows x {X_master.shape[1]} active features.")

    # ==================================================================
    # 3. ENFORCE STRICT GROUP-AWARE UNSEEN GENE 80/20 DATA SPLIT
    # ==================================================================
    # This ensures that entire gene loci are completely banished from the training validation loop
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(X_master, y_master, groups=genes_master))
    
    X_train, X_val = X_master.iloc[train_idx], X_master.iloc[val_idx]
    y_train, y_val = y_master[train_idx], y_master[val_idx]
    genes_train = genes_master[train_idx]
    
    print(f"--> Unseen Gene Split Layout: Train = {X_train.shape[0]} rows | Validation = {X_val.shape[0]} rows")

    # 4. XGBoost Optimization Engine Configuration
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1
    )
    
    # Defensive hyperparameter tuning matrix
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [2, 3, 4, 5], 
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'min_child_weight': [1, 3, 5], 
        'gamma': [0, 0.1, 0.2],      
        'reg_alpha': [0.1, 1, 5],       
        'reg_lambda': [1, 5, 10]      
    }
    
    # ==================================================================
    # 5. STRATIFY INTERNAL TRAINING VIA STRICT GROUPKFOLD (GENE IDENTIFIER)
    # ==================================================================
    cv = GroupKFold(n_splits=5)
    print("[*] Tuning hyperparameters via gene-isolated inner GroupKFold CV...")
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

    # 6. Evaluate Performance on Out-Of-Sample Gene Partition
    print("[*] Running forward pass inference on the unseen gene validation partition...")
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

    # ==================================================================
    # 7. SERIALIZE OPTIMIZED ARCHITECTURE AND EXPORT RECORDS
    # ==================================================================
    print(f"--> Saving final optimized unseen gene booster model file to: {model_output_path}")
    best_model.save_model(model_output_path)

    print(f"--> Exporting comprehensive evaluation log to: {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        f.write("=== Full Multi-Modality Master Model Metrics (Unseen Gene Split) ===\n")
        f.write(f"Locus-Grouped Training Pool Size: {X_train.shape[0]} rows\n")
        f.write(f"Locus-Grouped Validation Pool Size: {X_val.shape[0]} rows\n")
        f.write(f"Total Integrated Features Active: {X_train.shape[1]}\n")
        f.write("-" * 65 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("-" * 65 + "\n")
        f.write("\n=== Best Selected Architecture Parameters ===\n")
        for k, v in random_search.best_params_.items():
            f.write(f"{k}: {v}\n")

    # 8. Generate Performance Curves (ROC & PRC)
    print("[*] Rendering evaluation curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    ax1.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Full Model (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=11, fontweight='bold')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    precision, recall, _ = precision_recall_curve(y_val, y_proba)
    baseline_pr = np.sum(y_val == 1) / len(y_val)
    ax2.plot(recall, precision, color='dodgerblue', lw=2.5, label=f'Full Model (PR-AUC = {pr_auc:.3f})')
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

    # 9. Compute and Plot SHAP Values
    print("[*] Generating TreeSHAP Feature Importance Summary...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_train)
    
    fig_shap = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, max_display=20, show=False)
    
    plt.title("Full Integrated Feature Importance (TreeSHAP Beeswarm)", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    plt.savefig(shap_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(shap_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig_shap)
    
    print(f"[-->] Unseen gene script configuration completed successfully inside {working_dir}")

if __name__ == "__main__":
    main()
