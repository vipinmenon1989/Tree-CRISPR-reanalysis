import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')  # Required for saving plots on a headless cluster
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def main():
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Complete_dataset_sigmoid_score_A549/"
    train_file = os.path.join(working_dir, "CRISPR_ml_features_final_A549.csv")
    
    if not os.path.exists(train_file):
        print(f"CRITICAL ERROR: Matrix file missing from {working_dir}")
        sys.exit(1)
        
    print("[*] Ingesting A549 complete master matrix...")
    df = pd.read_csv(train_file, sep=',')
    df.columns = [col.lower() for col in df.columns]
    
    # Target definition
    df['class'] = (df['sigmoid_score'] > 0.25).astype(int)

    # Feature Selection: Complete (Baseline + Gene Epi + Guide Epi)
    base = [c for c in df.columns if c in {'distance_to_tss', 'mfe', 'tm', 'entropy', 'gc_count', 'gc_content', 'gc_low', 'gc_high'} or c.startswith(('count_', 'pos_', 'di_'))]
    gene = [c for c in df.columns if ('k562' in c or 'h3k' in c) and not c.startswith('guide_')]
    guide = [c for c in df.columns if c.startswith('guide_')]
    selected = base + gene + guide

    X = df[selected].select_dtypes(include=[np.number])
    y = df['class'].values
    
    # Extract Group Array (Mandatory for Unseen Gene Separation)
    if 'gene' not in df.columns:
        print("CRITICAL ERROR: 'gene' column not found in dataset. Cannot perform group split.")
        sys.exit(1)
    genes = df['gene'].values

    # ==================================================================
    # ENFORCE STRICT UNSEEN GENE HOLD-OUT SPLIT (80/20)
    # ==================================================================
    print("[*] Generating leak-proof out-of-sample cross-gene split...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=genes))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    genes_train = genes[train_idx]
    
    print(f"--> Train Matrix Structure: {X_train.shape[0]} rows (Unseen Genes)")
    print(f"--> Test Matrix Structure:  {X_test.shape[0]} rows (Unseen Genes)")

    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    # Model Setup
    xgb_clf = xgb.XGBClassifier(scale_pos_weight=ratio, random_state=42, eval_metric='auc', n_jobs=-1)
    
    # Hyperparameter Tuning Grid Specially Calibrated to Protect Smaller Subsets
    param_grid = {
        'n_estimators': [50, 100], 
        'max_depth': [2, 3], 
        'learning_rate': [0.01, 0.05], 
        'subsample': [0.7], 
        'reg_alpha': [1, 5], 
        'reg_lambda': [5, 10]
    }
    
    # ==================================================================
    # ENFORCE STRICT INTERNAL GROUPKFOLD CV
    # ==================================================================
    cv = GroupKFold(n_splits=5)
    print("[*] Tuning hyperparameters via cross-gene GroupKFold CV...")
    search = RandomizedSearchCV(xgb_clf, param_grid, n_iter=15, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train, groups=genes_train)
    
    # Evaluation
    best = search.best_estimator_
    y_p = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    
    # Export Model & Metrics
    print("--> Exporting performance metrics and model architecture...")
    best.save_model(os.path.join(working_dir, "complete_model.json"))
    with open(os.path.join(working_dir, "complete_metrics.txt"), 'w') as f:
        f.write(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}\n")
        f.write(f"AUPR: {average_precision_score(y_test, y_prob):.4f}\n")
        f.write(f"Precision: {precision_score(y_test, y_p, zero_division=0):.4f}\n")
        f.write(f"Recall: {recall_score(y_test, y_p, zero_division=0):.4f}\n")
        f.write(f"F1: {f1_score(y_test, y_p, zero_division=0):.4f}\n")

    # ==================================================================
    # SHAP Analysis
    # ==================================================================
    print("Generating SHAP plots...")
    explainer = shap.TreeExplainer(best)
    shap_values = explainer.shap_values(X_test)
    
    # 1. SHAP Summary Plot (Dot Plot)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.tight_layout()
    dot_plot_path = os.path.join(working_dir, "complete_shap_summary_dot.png")
    plt.savefig(dot_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP dot plot saved to: {dot_plot_path}")

    # 2. SHAP Bar Plot (Global Mean Absolute Impact)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    bar_plot_path = os.path.join(working_dir, "complete_shap_summary_bar.png")
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP bar plot saved to: {bar_plot_path}")
    print("[-->] Processing completed successfully.")

if __name__ == "__main__": 
    main()
