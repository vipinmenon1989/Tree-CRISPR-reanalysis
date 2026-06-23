import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def main():
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Integrated_Cell_Line/genome-wide/trial_finding"
    input_file = os.path.join(working_dir, "CRISPR_ml_features_final_K562.csv")
    
    model_output_path = os.path.join(working_dir, "baseline_xgboost_model.json")
    metrics_output_path = os.path.join(working_dir, "baseline_metrics.txt")
    
    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: Training file missing from {working_dir}")
        sys.exit(1)
        
    print("[*] Ingesting K562 matrix for raw baseline generation...")
    df = pd.read_csv(input_file)
    
    # Enforce lowercase columns to ensure exact header matching
    df.columns = [col.lower() for col in df.columns]
    
    # Target definition points directly to your baked-in class column
    y = df['class'].values

    # ==================================================================
    # STRICT EPIGENETIC ABLATION LAYER
    # ==================================================================
    # Targets all chromatin bins, methylation markers, and distance vectors
    epigenetic_keywords = ['bin_', 'methylation', 'h3k', 'atac_seq', 'distance_to_tss']
    epigenetic_cols = [col for col in df.columns if any(sub in col for sub in epigenetic_keywords)]
    
    # Drop structural metadata columns
    metadata_cols = [
        'id', 'sigmoid_score', 'class', 'gene', 'sgrna sequence', 
        'chromosome', 'strand', 'pam', 'extended_sequence', 'closest_tss_coord'
    ]
    
    cols_to_drop = list(set(epigenetic_cols + [col for col in metadata_cols if col in df.columns]))
    
    # Isolate your matrix (Retains nucleotide positioning + mfe, tm, entropy)
    X = df.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])
    
    print(f"--> Feature Matrix Isolated: {X.shape[0]} rows x {X.shape[1]} columns.")
    
    # Verification checks printed to console
    thermo_retained = [c for c in ['mfe', 'tm', 'entropy'] if c in X.columns]
    print(f"    -> Thermodynamic features retained: {thermo_retained}")
    print(f"    -> Sample of sequence features retained: {list(X.columns[:3])} ... {list(X.columns[-3:])}")

    # 3. Stratified Row-Level Split Baseline Execution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1
    )
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [2, 3],
        'learning_rate': [0.01, 0.05],
        'subsample': [0.7],
        'reg_alpha': [1, 5],
        'reg_lambda': [5, 10]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb_clf, param_distributions=param_grid, 
        n_iter=15, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42
    )
    
    print("[*] Running sequence + thermodynamic baseline optimization...")
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0)
    }
    
    best_model.save_model(model_output_path)
    
    with open(metrics_output_path, 'w') as f:
        f.write("=== Pure Sequence + Thermodynamic Baseline Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n=== Best Parameters ===\n")
        for k, v in random_search.best_params_.items():
            f.write(f"{k}: {v}\n")
            
    print(f"[-->] Completed successfully. Metrics saved to {metrics_output_path}")

if __name__ == "__main__":
    main()
