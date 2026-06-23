import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

# ======================================================================
# FORCE HEADLESS HPC RENDER BACKEND (MUST EXECUTE BEFORE PYPLOT INGEST)
# ======================================================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# ======================================================================

import shap
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score, roc_curve, precision_recall_curve, confusion_matrix, auc
)

def calculate_pr_gain(y_true, y_proba):
    """
    Computes Precision-Recall Gain curves and integrates Area Under Curve (AUPRG).
    Harmonic bounds are structurally confined to the [0, 1] positive unit square.
    """
    pi = np.sum(y_true == 1) / len(y_true)
    if pi == 0 or pi == 1:
        return np.array([0.0]), np.array([0.0]), 0.0
        
    thresholds = np.sort(y_proba)
    prec_gains = []
    rec_gains = []
    
    prec_gains.append(0.0)
    rec_gains.append(1.0)
    
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            continue
            
        if tp == 0:
            continue
            
        prec_g = 1.0 - (pi / (1.0 - pi)) * (fp / tp)
        rec_g = 1.0 - (pi / (1.0 - pi)) * (fn / tp)
        
        prec_gains.append(max(0.0, prec_g))
        rec_gains.append(max(0.0, rec_g))
        
    prec_gains.append(1.0)
    rec_gains.append(0.0)
    
    idx_sort = np.argsort(rec_gains)
    rg_sorted = np.array(rec_gains)[idx_sort]
    pg_sorted = np.array(prec_gains)[idx_sort]
    
    return rg_sorted, pg_sorted, auc(rg_sorted, pg_sorted)

def main():
    # 1. Cluster Workspace Configuration
    working_dir = "./"    
    train_file = os.path.join(working_dir, "CRISPR_ml_features_pure_resistance_perfect_align.csv")
    
    metrics_output_path = os.path.join(working_dir, "ensemble_unseen_guide_prgain_metrics.txt")
    curves_png_path = os.path.join(working_dir, "ensemble_unseen_guide_prgain_curves.png")
    shap_png_path = os.path.join(working_dir, "ensemble_unseen_guide_shap_beeswarm.png")
    
    if not os.path.exists(train_file):
        print(f"CRITICAL ERROR: Input file missing from {working_dir}")
        sys.exit(1)
        
    print("[*] Ingesting 901-row pure resistance data matrix...")
    df_master = pd.read_csv(train_file)
    df_master.columns = [str(col).strip().lower() for col in df_master.columns]

    if 'sgrna sequence' not in df_master.columns:
        print("CRITICAL ERROR: 'sgrna sequence' column missing for guide-level grouping!")
        sys.exit(1)

    meta_cols = {'id', 'sgrna sequence', 'gene', 'class', 'sigmoid_score'}
    features = [col for col in df_master.columns if col not in meta_cols]
    
    X_master = df_master[features].select_dtypes(include=[np.number])
    y_master = df_master['class'].values

    # ==================================================================
    # 2. ENFORCE STRICTOR UNSEEN GUIDE SPLIT BASED ON UNIQUE SEQUENCES
    # ==================================================================
    print("[*] Grouping unique guide sequences to ensure sterile evaluation...")
    unique_guides = df_master['sgrna sequence'].unique()
    
    np.random.seed(42)
    shuffled_guide_idx = np.random.permutation(len(unique_guides))
    split_cutoff = int(len(unique_guides) * 0.80)
    
    train_guides = set(unique_guides[shuffled_guide_idx[:split_cutoff]])
    val_guides = set(unique_guides[shuffled_guide_idx[split_cutoff:]])
    
    train_mask = df_master['sgrna sequence'].isin(train_guides).values
    val_mask = df_master['sgrna sequence'].isin(val_guides).values
    
    X_train_full, X_val = X_master[train_mask], X_master[val_mask]
    y_train_full, y_val = y_master[train_mask], y_master[val_mask]
    
    pos_idx = np.where(y_train_full == 1)[0]
    neg_idx = np.where(y_train_full == 0)[0]
    
    X_pos = X_train_full.iloc[pos_idx].reset_index(drop=True)
    X_neg = X_train_full.iloc[neg_idx].reset_index(drop=True)

    # ==================================================================
    # 3. CONSTRUCT DISJOINT BALANCED CHUNKS
    # ==================================================================
    shuffled_pos_idx = np.random.permutation(len(X_pos))
    chunks = np.array_split(shuffled_pos_idx, 4)
    ensemble_models = []
    
    # Store sub-ensemble background training arrays for unified SHAP calculation
    sub_train_pools = []
    
    param_grid = {
        'n_estimators': [30, 50, 80],
        'max_depth': [2, 3], 
        'learning_rate': [0.02, 0.05],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.6, 0.7],
        'min_child_weight': [3, 5]
    }

    # ==================================================================
    # 4. TRAIN BALANCED SUB-ENSEMLES
    # ==================================================================
    for idx, chunk in enumerate(chunks):
        X_pos_chunk = X_pos.iloc[chunk]
        X_sub = pd.concat([X_neg, X_pos_chunk], axis=0).reset_index(drop=True)
        y_sub = np.concatenate([np.zeros(len(X_neg)), np.ones(len(X_pos_chunk))])
        
        sub_train_pools.append(X_sub)
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        xgb_base = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42, n_jobs=-1)
        
        search = RandomizedSearchCV(
            estimator=xgb_base, param_distributions=param_grid, 
            n_iter=10, scoring='average_precision', cv=cv, n_jobs=-1, random_state=42
        )
        search.fit(X_sub, y_sub)
        ensemble_models.append(search.best_estimator_)

    # ==================================================================
    # 5. BLENDED INFERENCE & METRIC HARVESTING
    # ==================================================================
    proba_accum = np.zeros(len(X_val))
    for model in ensemble_models:
        proba_accum += model.predict_proba(X_val)[:, 1]
        
    y_proba_ensemble = proba_accum / len(ensemble_models)
    y_pred_ensemble = (y_proba_ensemble > 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, y_proba_ensemble)
    pr_auc = average_precision_score(y_val, y_proba_ensemble)
    baseline_floor = np.sum(y_val == 1) / len(y_val)
    
    rg, pg, auprg = calculate_pr_gain(y_val, y_proba_ensemble)

    # ==================================================================
    # 6. ENSEMBLE TREESHAP INTEGRATION ENGINE
    # ==================================================================
    print("[*] Generating unified TreeSHAP values across sub-ensembles...")
    
    # We evaluate feature importance over the validation space to see what drives generalizable logic
    shap_values_list = []
    
    for model in ensemble_models:
        explainer = shap.TreeExplainer(model)
        # Generate raw shap values for the validation matrix
        sv = explainer.shap_values(X_val)
        shap_values_list.append(sv)
        
    # Average the SHAP arrays across all 4 disjoint models
    mean_shap_values = np.mean(shap_values_list, axis=0)
    
    # Construct a clean SHAP Explanation container compatible with the standard beeswarm plotting API
    explanation = shap.Explanation(
        values=mean_shap_values,
        base_values=np.mean([shap.TreeExplainer(m).expected_value for m in ensemble_models]),
        data=X_val.values,
        feature_names=X_val.columns.tolist()
    )

    # Output the presentation-grade beeswarm plot
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("Ensemble Balanced Model Feature Importance (TreeSHAP Beeswarm)", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(shap_png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"--> Unified TreeSHAP profiles successfully rendered to: {shap_png_path}")

    # ==================================================================
    # 7. LOG ENHANCED COMBINED RESULTS
    # ==================================================================
    with open(metrics_output_path, 'w') as f:
        f.write("=== Unseen Guide (Sequence Split) Balanced Bagging Performance ===\n")
        f.write(f"Sterile Sequence Validation Pool Size: {X_val.shape[0]} rows\n")
        f.write(f"Baseline Un-balanced Class 1 Ratio Floor: {baseline_floor:.4f}\n")
        f.write("-" * 65 + "\n")
        f.write(f"Ensemble Resolved Validation ROC-AUC:   {roc_auc:.4f}  (Baseline: 0.5000)\n")
        f.write(f"Ensemble Resolved Validation PR-AUC:    {pr_auc:.4f}  (Baseline: {baseline_floor:.4f})\n")
        f.write(f"Ensemble Resolved Validation AUPRG:     {auprg:.4f}  (Baseline: 0.0000)\n")
        f.write("-" * 65 + "\n")
        f.write(f"Ensemble Resolved Precision Score:     {precision_score(y_val, y_pred_ensemble, zero_division=0):.4f}\n")
        f.write(f"Ensemble Resolved Recall Score:        {recall_score(y_val, y_pred_ensemble, zero_division=0):.4f}\n")
        f.write(f"Ensemble Resolved Balanced F1 Score:   {f1_score(y_val, y_pred_ensemble, zero_division=0):.4f}\n")
        f.write("-" * 65 + "\n")

    # ==================================================================
    # 8. GENERATE VISUAL PERFORMANCE PLOTS (3-PANEL MATRIX)
    # ==================================================================
    print("[*] Generating 3-panel comparative performance plot matrix...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6))
    
    fpr, tpr, _ = roc_curve(y_val, y_proba_ensemble)
    ax1.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Ensemble (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=11, fontweight='bold')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=12, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    precision, recall, _ = precision_recall_curve(y_val, y_proba_ensemble)
    ax2.plot(recall, precision, color='dodgerblue', lw=2.5, label=f'Ensemble (PR-AUC = {pr_auc:.3f})')
    ax2.axhline(y=baseline_floor, color='gray', lw=1.5, linestyle='--', label=f'Baseline Class floor ({baseline_floor:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision (PPV)', fontsize=11, fontweight='bold')
    ax2.set_title('Precision-Recall Curve (PRC)', fontsize=12, fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    ax3.plot(rg, pg, color='crimson', lw=2.5, label=f'Ensemble (AUPRG = {auprg:.3f})')
    ax3.plot([0, 1], [1, 0], color='black', lw=1.2, linestyle=':', label='Random Baseline (AUPRG = 0.000)')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('Recall Gain', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Precision Gain', fontsize=11, fontweight='bold')
    ax3.set_title('Precision-Recall Gain Curve', fontsize=12, fontweight='bold')
    ax3.legend(loc="lower left")
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(curves_png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[-->] Completed safely. High-resolution curves and SHAP plots are fully serialized.")

if __name__ == "__main__":
    main()
