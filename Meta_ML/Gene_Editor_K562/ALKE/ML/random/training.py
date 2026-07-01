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
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score, roc_curve, precision_recall_curve
)

def main():
    # 1. Cluster Workspace Configuration
    working_dir = "./"
    train_file = os.path.join(working_dir, "CRISPRi_ML_Train_80_random.txt")
    
    model_output_path = os.path.join(working_dir, "full_multimodality_model.json")
    metrics_output_path = os.path.join(working_dir, "random_split_multimodality_metrics.txt")
    curves_png_path = os.path.join(working_dir, "random_split_multimodality_curves.png")
    curves_pdf_path = os.path.join(working_dir, "random_split_multimodality_curves.pdf")
    shap_png_path = os.path.join(working_dir, "random_split_multimodality_shap.png")
    shap_pdf_path = os.path.join(working_dir, "random_split_multimodality_shap.pdf")
    
    if not os.path.exists(train_file):
        print(f"CRITICAL ERROR: Training file missing from {working_dir}")
        sys.exit(1)
        
    print("[*] Ingesting master data matrix...")
    df_master = pd.read_csv(train_file, sep='\t')
    df_master.columns = [col.lower() for col in df_master.columns]

    # Verify target variable existence
    if 'class' not in df_master.columns:
        if 'mean_sigmoid_score' in df_master.columns:
            print("[*] 'class' column missing. Deriving from 'mean_sigmoid_score' threshold > 0.25...")
            df_master['class'] = (df_master['mean_sigmoid_score'] > 0.25).astype(int)
        else:
            print("CRITICAL ERROR: Neither 'class' nor 'mean_sigmoid_score' found in headers.")
            sys.exit(1)

    # ==================================================================
    # 2. FEATURE ISOLATION & METADATA DROPS
    # ==================================================================
    explicit_metadata_drops = [
        'id', 'gene', 'mean_sigmoid_score', 'sigmoid_score',
        'unique_sgrna_id', 'sgrna sequence', 'cell_line_origin', 
        'distance_to_tss', 'class'
    ]
    cols_to_drop = [col for col in explicit_metadata_drops if col in df_master.columns]
    
    X_master = df_master.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])
    y_master = df_master['class'].values

    print(f"--> Feature Matrix Isolated: {X_master.shape[0]} rows x {X_master.shape[1]} active features.")

    # ==================================================================
    # 3. ISOLATE STERILE HOLDOUT TEST POOL (20%) FROM MASTER POPULATION
    # ==================================================================
    X_train_cv_pool, X_holdout_test, y_train_cv_pool, y_holdout_test = train_test_split(
        X_master, 
        y_master, 
        test_size=0.20, 
        stratify=y_master, 
        random_state=42
    )
    print(f"--> Data Footprint Layout: Training/CV Pool = {X_train_cv_pool.shape[0]} rows | Pure Holdout Test = {X_holdout_test.shape[0]} rows")

    # ==================================================================
    # 4. ROBUST PARAMS (CONSTRAINED HIGH-BIAS SETUP TO PREVENT MEMORIZATION)
    # ==================================================================
    ratio = float(np.sum(y_train_cv_pool == 0)) / np.sum(y_train_cv_pool == 1)
    
    best_params = {
        'objective': 'binary:logistic',
        'scale_pos_weight': ratio,
        'random_state': 42,
        'eval_metric': 'auc',
        'n_estimators': 400,
        'max_depth': 3,                  # Loosened slightly to 3 to capture subtle signal interactions
        'learning_rate': 0.02,           # Consistently slow updates
        'subsample': 0.7,                
        'colsample_bytree': 0.6,         
        'min_child_weight': 5,           
        'gamma': 0.2,                    
        'reg_alpha': 2.0,                # Marginally relaxed to give real signals room to emerge
        'reg_lambda': 5.0,               
        'n_jobs': -1
    }

    # ==================================================================
    # 5. EXPLICIT 5-FOLD STRATIFIED CROSS-VALIDATION WITHIN CV POOL
    # ==================================================================
    print("\n[*] Initializing Explicit 5-Fold Stratified Cross-Validation Engine on Training Pool...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Track cross-validated performance metrics inside the training subset
    oof_probabilities = np.zeros(len(X_train_cv_pool))
    oof_predictions = np.zeros(len(X_train_cv_pool))
    fold_metrics = []
    trained_models = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_cv_pool, y_train_cv_pool), 1):
        X_train_f, X_val_f = X_train_cv_pool.iloc[train_idx], X_train_cv_pool.iloc[val_idx]
        y_train_f, y_val_f = y_train_cv_pool[train_idx], y_train_cv_pool[val_idx]
        
        # Clone learner configuration per fold
        model_fold = xgb.XGBClassifier(**best_params, early_stopping_rounds=15)
        
        # Fit strictly on this fold split; monitor metrics against out-of-fold slice cleanly
        model_fold.fit(
            X_train_f, y_train_f,
            eval_set=[(X_val_f, y_val_f)],
            verbose=False
        )
        
        # Pull evaluation metrics
        probs_f = model_fold.predict_proba(X_val_f)[:, 1]
        preds_f = model_fold.predict(X_val_f)
        
        oof_probabilities[val_idx] = probs_f
        oof_predictions[val_idx] = preds_f
        trained_models.append(model_fold)
        
        fold_auc = roc_auc_score(y_val_f, probs_f)
        fold_pr  = average_precision_score(y_val_f, probs_f)
        fold_metrics.append((fold_auc, fold_pr))
        print(f"    [Fold {fold}/5 Complete] Internal Volatiles -> ROC-AUC: {fold_auc:.4f} | PR-AUC: {fold_pr:.4f}")

    # Calculate overall cross-validation matrix metrics
    cv_roc_auc = roc_auc_score(y_train_cv_pool, oof_probabilities)
    cv_pr_auc  = average_precision_score(y_train_cv_pool, oof_probabilities)

    # ==================================================================
    # 6. UNBIASED FINAL INFERENCE OVER CLEAN HOLDOUT VALIDATION POOL (20%)
    # ==================================================================
    print("\n[*] Evaluating clean, un-leaked Holdout Test Partition (20%)...")
    
    # Average the probability outputs of all 5 fold models to create a robust ensemble prediction
    test_probs_ensemble = np.mean([model.predict_proba(X_holdout_test)[:, 1] for model in trained_models], axis=0)
    test_preds_ensemble = (test_probs_ensemble > 0.5).astype(int)
    
    final_test_roc_auc = roc_auc_score(y_holdout_test, test_probs_ensemble)
    final_test_pr_auc  = average_precision_score(y_holdout_test, test_probs_ensemble)
    
    final_metrics = {
        "Holdout Test ROC-AUC": final_test_roc_auc,
        "Holdout Test PR-AUC": final_test_pr_auc,
        "Holdout Test Precision": precision_score(y_holdout_test, test_preds_ensemble, zero_division=0),
        "Holdout Test Recall": recall_score(y_holdout_test, test_preds_ensemble, zero_division=0),
        "Holdout Test F1 Score": f1_score(y_holdout_test, test_preds_ensemble, zero_division=0)
    }

    # Extract the highest-performing model variant from our loop to use for downstream SHAP diagnostics
    best_model = trained_models[np.argmax([f[0] for f in fold_metrics])]

    # ==================================================================
    # 7. SERIALIZE AND WRITE METRICS TRACKING LOG
    # ==================================================================
    print(f"--> Saving top-performing fold booster to: {model_output_path}")
    best_model.save_model(model_output_path)

    print(f"--> Exporting comprehensive nested CV & Holdout evaluation log to: {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        f.write("=== Nested 5-Fold CV & Sterile Holdout Metrics ===\n")
        f.write(f"Total Master Row Population: {X_master.shape[0]} rows\n")
        f.write(f"Internal CV Training Pool: {X_train_cv_pool.shape[0]} rows\n")
        f.write(f"Isolated Holdout Test Pool: {X_holdout_test.shape[0]} rows\n")
        f.write(f"Total Integrated Features Active: {X_master.shape[1]}\n")
        f.write("-" * 65 + "\n")
        f.write("--- Internal 5-Fold Training Pool CV Logs ---\n")
        for fold, (f_auc, f_pr) in enumerate(fold_metrics, 1):
            f.write(f" Fold {fold} | ROC-AUC: {f_auc:.4f} | PR-AUC: {f_pr:.4f}\n")
        f.write(f" Combined CV Loop ROC-AUC: {cv_roc_auc:.4f}\n")
        f.write(f" Combined CV Loop PR-AUC:  {cv_pr_auc:.4f}\n")
        f.write("-" * 65 + "\n")
        f.write("--- Sterile 20% Holdout Test Final Logs ---\n")
        for k, v in final_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("-" * 65 + "\n")
        f.write("\n=== Rigid Architecture Parameters Applied ===\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")

    # ==================================================================
    # 8. GENERATE TRUE OUT-OF-SAMPLE PERFORMANCE CURVES (ON HOLDOUT TEST)
    # ==================================================================
    print("[*] Rendering evaluation curves from holdout validation context...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    fpr, tpr, _ = roc_curve(y_holdout_test, test_probs_ensemble)
    ax1.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Holdout Ensemble (AUC = {final_test_roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=11, fontweight='bold')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    precision, recall, _ = precision_recall_curve(y_holdout_test, test_probs_ensemble)
    baseline_pr = np.sum(y_holdout_test == 1) / len(y_holdout_test)
    ax2.plot(recall, precision, color='dodgerblue', lw=2.5, label=f'Holdout Ensemble (PR-AUC = {final_test_pr_auc:.3f})')
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

    # ==================================================================
    # 9. COMPUTE TREESHAP EXPLANATIONS ON MASTER HOLDOUT REPRESENTATIVE
    # ==================================================================
    print("[*] Generating TreeSHAP Feature Importance Summary...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_master)
    
    fig_shap = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_master, max_display=20, show=False)
    
    plt.title("Cross-Validated Feature Importance (TreeSHAP Beeswarm)", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    plt.savefig(shap_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(shap_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig_shap)
    
    print(f"[-->] Training execution successfully complete for working directory: {working_dir}")

if __name__ == "__main__":
    main()
