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
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score, roc_curve, precision_recall_curve
)

def main():
    # 1. Cluster Workspace Configuration
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Complete_dataset_sigmoid_score_K562/Isolated_equation/Resistance_data/"    
    train_file = os.path.join(working_dir, "CRISPR_ml_features_pure_resistance_filtered.csv")
    
    # Path configuration for unique unseen guide outputs
    model_output_path = os.path.join(working_dir, "unseen_guide_multiomic_model.json")
    metrics_output_path = os.path.join(working_dir, "unseen_guide_multiomic_metrics.txt")
    curves_png_path = os.path.join(working_dir, "unseen_guide_multiomic_curves.png")
    curves_pdf_path = os.path.join(working_dir, "unseen_guide_multiomic_curves.pdf")
    shap_png_path = os.path.join(working_dir, "unseen_guide_multiomic_shap.png")
    shap_pdf_path = os.path.join(working_dir, "unseen_guide_multiomic_shap.pdf")
    
    if not os.path.exists(train_file):
        print(f"CRITICAL ERROR: Training file missing from {working_dir}")
        sys.exit(1)
        
    print("[*] Ingesting 901-row pure resistance data matrix...")
    df_master = pd.read_csv(train_file, sep=None, engine='python')
    
    # Aggressive whitespace stripping and lowercasing immediately
    df_master.columns = [str(col).strip().lower() for col in df_master.columns]

    # Verify tracking markers post-sanitization
    if 'class' not in df_master.columns:
        print("CRITICAL ERROR: Pre-computed 'class' column not found!")
        sys.exit(1)

    # ==================================================================
    # 2. FEATURE ISOLATION ENGINE: SEQUENCE + ALL 140 EPIGENETIC BINS
    # ==================================================================
    print("[*] Filtering feature matrix to build multi-omic feature space...")
    
    thermo_and_structural_set = {
        'mfe', 'tm', 'entropy', 'gc_count', 'gc_content', 'gc_low', 'gc_high', 'distance_to_tss'
    }
    
    active_features = []
    found_types = set()
    
    for col in df_master.columns:
        # Retain all 140 epigenetic chromatin and global methylation tracks
        if 'bin' in col or 'h3k' in col or 'atac' in col or 'methylation' in col:
            active_features.append(col)
            found_types.add('epigenetic_chromatin_bins')
            
        # Retain counts, positional tracks, and dinucleotide sequences
        elif col.startswith(('count_', 'pos', 'di')):
            active_features.append(col)
            found_types.add(col.split('_')[0] if '_' in col else col[:3])
            
        # Retain explicitly designated thermodynamics and location metadata
        elif col in thermo_and_structural_set:
            active_features.append(col)
            found_types.add('thermo_structural')

    if not active_features:
        print("CRITICAL ERROR: Zero functional features isolated!")
        sys.exit(1)

    # Re-assign matrix with the complete un-ablated multi-omic feature space
    X_master = df_master[active_features].select_dtypes(include=[np.number])
    y_master = df_master['class'].values

    print(f"--> Integrated Multi-Omic Matrix Isolated: {X_master.shape[0]} rows x {X_master.shape[1]} features.")
    print(f"    -> Active feature spaces: {list(found_types)}")

    # ==================================================================
    # 3. ENFORCE STRATIFIED UNSEEN GUIDE 80/20 DATA SPLIT
    # ==================================================================
    # Standard stratified shuffle split isolates independent guide lines completely at random
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(sss.split(X_master, y_master))
    
    X_train, X_val = X_master.iloc[train_idx], X_master.iloc[val_idx]
    y_train, y_val = y_master[train_idx], y_master[val_idx]
    
    print(f"--> Unseen Guide Split Layout: Train = {X_train.shape[0]} rows | Validation = {X_val.shape[0]} rows")

    # ==================================================================
    # 4. XGBOOST OPTIMIZATION ENGINE CONFIGURATION (ASYMMETRIC GRADIENTS)
    # ==================================================================
    # Minority Class (0) / Majority Class (1) to heavily penalize rare baseline errors
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    print(f"--> Calculated training split scale_pos_weight factor: {ratio:.4f}")

    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='aucpr',  # Directly monitors area under Precision-Recall curve
        n_jobs=-1
    )
    
    # Broad grid calibrated to balance structural accessibility with sequence code
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
    # 5. STRATIFY INTERNAL TUNING VIA STRATIFIED K-FOLD (PR-AUC SEARCH)
    # ==================================================================
    # Standard StratifiedKFold removes the locus constraint from the hyperparameter selection loop
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("[*] Tuning hyperparameters via inner StratifiedKFold CV optimized for PR-AUC...")
    random_search = RandomizedSearchCV(
        estimator=xgb_clf, 
        param_distributions=param_grid, 
        n_iter=40, 
        scoring='average_precision',  # Direct PR-AUC tracking optimization target
        cv=cv, 
        n_jobs=-1, 
        random_state=42
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    # 6. Evaluate Performance on Out-Of-Sample Guide Partition
    print("[*] Running inference on the unseen guide validation partition...")
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
        f.write("=== Multi-Omic Sequence + Epigenetic Model Metrics (Unseen Guide Split) ===\n")
        f.write(f"Stratified Training Pool Size: {X_train.shape[0]} rows\n")
        f.write(f"Stratified Validation Pool Size: {X_val.shape[0]} rows\n")
        f.write(f"Total Combined Features Active: {X_train.shape[1]}\n")
        f.write(f"Applied scale_pos_weight Factor: {ratio:.4f}\n")
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
    ax1.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Multi-Omic Model (AUC = {roc_auc:.3f})')
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
    ax2.plot(recall, precision, color='dodgerblue', lw=2.5, label=f'Multi-Omic Model (PR-AUC = {pr_auc:.3f})')
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
    
    plt.title("Integrated Multi-Omic Feature Importance (TreeSHAP Beeswarm)", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    
    plt.savefig(shap_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(shap_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig_shap)
    
    print(f"[-->] Unseen guide multi-omic script complete. Performance profiles serialized inside: {working_dir}")

if __name__ == "__main__":
    main()
