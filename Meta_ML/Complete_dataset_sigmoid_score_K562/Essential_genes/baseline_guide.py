import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def main():
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Complete_dataset_sigmoid_score_K562/Essential_genes/"
    input_file = os.path.join(working_dir, "CRISPR_ml_features_final.csv")
    
    model_output_path = os.path.join(working_dir, "guide_bins_only_ablation_model.json")
    metrics_output_path = os.path.join(working_dir, "guide_bins_only_ablation_metrics.txt")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Error: {input_file} not found.")
        
    df = pd.read_csv(input_file)
    
    # 1. Target Definition
    if 'class' not in df.columns:
        raise KeyError("Error: Pre-calculated target label column 'class' missing from data.")
    y = df['class']
    
    # 2. Corrected Feature Filter Engine (Isolating Exact Identifiers)
    leakage_keywords = ['sigmoid', 'class', 'distance_to_tss']
    
    cols_to_drop = []
    for col in df.columns:
        col_lower = col.lower()
        
        # Protect 'guide_' columns by enforcing exact string evaluation for 'id'
        if col_lower == 'id' or any(kw in col_lower for kw in leakage_keywords):
            cols_to_drop.append(col)
            
        # Drop macro locus-level promoter bins (has '_bin_' but lacks 'guide_')
        elif '_bin_' in col_lower and 'guide_' not in col_lower:
            cols_to_drop.append(col)
            
    # Build feature matrix: 611 Sequence features + 70 granular guide-level micro-bins
    X = df.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])
    
    print("======================================================================")
    print("CRISPRi ABLATION RUN: STAGE 3 (SEQUENCE + GUIDE BINS ONLY)")
    print("======================================================================")
    print(f"[*] Features remaining for model training: {X.shape[1]}")
    print(f"[*] Target distribution: Positive Class = {np.sum(y == 1)}, Negative Class = {np.sum(y == 0)}")
    print("======================================================================\n")
    
    if X.shape[1] == 611:
        print("CRITICAL LOGIC ERROR: Guide features are still being filtered out. Terminating.")
        sys.exit(1)
        
    # 3. Stratified Splitting Matrix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc'
    )
    
    # 4. Hyperparameter Search Grid
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
    
    print("[*] Commencing optimization pass over guide feature constraints...")
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    # 5. Validation and Evaluation Metrics
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC (Avg Precision)": average_precision_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0)
    }
    
    # 6. File Serialization Run
    best_model.save_model(model_output_path)
    
    with open(metrics_output_path, 'w') as f:
        f.write("=== Ablation Stage 3: Sequence + Guide Bins Only ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n=== Optimized Parameters ===\n")
        for k, v in random_search.best_params_.items():
            f.write(f"{k}: {v}\n")
            
    print(f"[-->] Metrics successfully written to: {metrics_output_path}")
    print(f"[-->] Model structure file successfully saved to: {model_output_path}")

if __name__ == "__main__":
    main()
