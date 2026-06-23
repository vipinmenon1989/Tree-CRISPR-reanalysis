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
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score, roc_curve, precision_recall_curve
)

def main():
    # 1. Cluster Workspace Configuration
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Integrated_Cell_Line/Isolated_equation/cell_line/K562/"    
    train_file = os.path.join(working_dir, "CRISPR_ml_features_final_K562.csv")
    
    # Path configuration for model checkpoint and evaluation outputs (full multi-modality random split)
    model_output_path = os.path.join(working_dir, "random_split_multimodality_model_720.json")
    metrics_output_path = os.path.join(working_dir, "random_split_multimodality_metrics_720.txt")
    curves_png_path = os.path.join(working_dir, "random_split_multimodality_curves_720.png")
    curves_pdf_path = os.path.join(working_dir, "random_split_multimodality_curves_720.pdf")
    shap_png_path = os.path.join(working_dir, "random_split_multimodality_shap_720.png")
    shap_pdf_path = os.path.join(working_dir, "random_split_multimodality_shap_720.pdf")
    
    if not os.path.exists(train_file):
        print(f"CRITICAL ERROR: Training file missing from {working_dir}")
        sys.exit(1)
        
    print("[*] Ingesting 720-row master data matrix...")
    df_master = pd.read_csv(train_file, sep=None, engine='python')
    
    # Aggressive whitespace stripping and lowercasing immediately
    df_master.columns = [str(col).strip().lower() for col in df_master.columns]

    # Verify admin and tracking trackers post-sanitization
    if 'sigmoid_score' not in df_master.columns:
        print("CRITICAL ERROR: 'sigmoid_score' column not found!")
        sys.exit(1)

    # Target binarization from stable sigmoid score thresholding
    print("[*] Binarizing labels via isolated sigmoid threshold...")
    df_master['class'] = (df_master['sigmoid_score'] > 0.25).astype(int)

    # ==================================================================
    # 2. FULL MULTI-MODALITY FEATURE ISOLATION (NO ABLATION)
    # ==================================================================
    print("[*] Filtering feature matrix to isolate ALL active features...")
    
    explicit_metadata_drops = [
        'unique_sgrna_id', 'id', 'gene', 'sgrna sequence', 
        'cell_line_origin', 'distance_to_tss', 'sigmoid_score', 'class'
    ]
    cols_to_drop = [col for col in explicit_metadata_drops if col in df_master.columns]
    
    # Isolate all numeric tracking weights, keeping all k-mers, counts, and 10-bin epigenetic markers active
    X_master = df_master.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])
    y_master = df_master['class'].values

    print(f"--> Full Feature Matrix Isolated: {X_master.shape[0]} rows x {X_master.shape[1]} active features.")
    
    # Sanity verification for active epigenetic tracks
    active_bins = [col for col in X_master.columns if 'bin' in col]
    print(f"    -> Successfully verified {len(active_bins)} active epigenetic chromatin/methylation tracks.")

    # ==================================================================
    # 3. ENFORCE STRATIFIED RANDOMIZED 80/20 DATA SPLIT (NO GROUPS)
    # ==================================================================
    X_train, X_val, y_train, y_val = train_test_split(
        X_master, 
        y_master, 
        test_size=0.20, 
        stratify=y_master, 
        random_state=42
    )
    
    print(f"--> Randomized Split Layout: Train = {X_train.shape[0]} rows | Validation = {X_val.shape[0]} rows")

    # 4. XGBoost Optimization Engine Configuration
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1
    )
    
    # Defensive parameter grid calibrated to guard against overfitting 751 tracks over small row limits
    param_grid = {
        'n_estimators': [30, 50, 80, 100],
        'max_depth': [2, 3], 
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [3, 5, 7], 
        'gamma': [0.1, 0.2, 0.4],      
        'reg_alpha': [1, 5, 10],       
        'reg_lambda': [10, 15, 20]      
    }
    
    # ==================================================================
    # 5. STRATIFIED SHUFFLED K-FOLD TUNING SYSTEM
    # ==================================================================
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("[*] Tuning hyperparameters via randomized inner StratifiedKFold CV...")
    random_search = RandomizedSearchCV(
        estimator=xgb_clf, 
        param_distributions=param_grid, 
        n_iter=40, 
        scoring='roc_auc', 
        cv=cv, 
        n_jobs=-1, 
        random_state=42
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    # 6. Evaluate Performance on Random Out-of-Sample Partition
    print("[*] Running inference on the randomized validation partition...")
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
    # 7. EXPORT MODEL ARCHITECTURE AND PERFORMANCE LOGS
    # ==================================================================
    print(f"--> Serializing best booster checkpoint to: {model_output_path}")
    best_model.save_model(model_output_path)

    print(f"--> Exporting comprehensive evaluation log to: {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        f.write("=== Full Multi-Modality Model Metrics (Randomized Split - 720 Rows) ===\n")
        f.write(f"Randomized Training Pool Size: {X_train.shape[0]} rows\n")
        f.write(f"Randomized Validation Pool Size: {X_val.shape[0]} rows\n")
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
    
    plt.title("Full Multi-Modality Feature Importance (TreeSHAP Beeswarm)", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    plt.savefig(shap_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(shap_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig_shap)
    
    print(f"[-->] Full multimodality randomized script completed inside {working_dir}")

if __name__ == "__main__":
    main()
