import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def main():
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Integrated_Cell_Line/genome-wide/ML_unseen_guide/train_test/"
    input_file = os.path.join(working_dir, "CRISPRi_ML_Train_80.txt")
    
    model_output_path = os.path.join(working_dir, "full_xgboost_model.json")
    metrics_output_path = os.path.join(working_dir, "fill_metrics.txt")
    
    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: Training file missing from {working_dir}")
        sys.exit(1)
        
    print("[*] Ingesting K562 master data pool...")
    df = pd.read_csv(input_file,sep="\t")
    
    # Standardize headers to lowercase to eliminate spelling/capitalization issues
    df.columns = [col.lower() for col in df.columns]
    
    # Target definition utilizes your baked-in class values
    if 'class' not in df.columns:
        print("CRITICAL ERROR: Target 'class' column missing from dataset matrix.")
        sys.exit(1)
        
    y = df['class'].values

    # ==================================================================
    # TOTAL FEATURE ACTIVATION LAYER (METADATA EXCLUSION ONLY)
    # ==================================================================
    # We strip out alphanumeric administrative strings and the continuous target score.
    # Everything else—all chromatin bins, guide bins, thermodynamics, and k-mers—is kept.
    metadata_cols = [
        'id', 'sigmoid_score', 'class', 'gene', 'sgrna sequence', 
        'chromosome', 'strand', 'pam', 'extended_sequence', 'closest_tss_coord'
    ]
    
    X = df.drop(columns=metadata_cols, errors='ignore').select_dtypes(include=[np.number])
    
    print(f"--> Full Feature Matrix Isolated: {X.shape[0]} rows x {X.shape[1]} active columns.")
    
    # Direct feature tracking confirmation dumps
    has_tss = 'distance_to_tss' in X.columns
    has_thermo = 'tm' in X.columns
    bin_count = sum(1 for c in X.columns if 'bin_' in c)
    print(f"    -> distance_to_tss status: {'RETAINED' if has_tss else 'MISSING'}")
    print(f"    -> Thermodynamics tracking: {'RETAINED' if has_thermo else 'MISSING'}")
    print(f"    -> Integrated epigenetic bins total: {bin_count}")

    # 3. Enforce Standard Row-Level Stratified 80/20 Splitting Matrix
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"--> Data Shuffled: Train Pool = {X_train.shape[0]} rows | Test Validation Pool = {X_test.shape[0]} rows")

    # Handle positive/negative class weights 
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1
    )
    
    # Your exact hyperparameter search grid constraints
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
    
    print("[*] Running optimization matrix loops over all active modalities...")
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    # Inference layer execution
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0)
    }
    
    # Save optimized structures back to cluster workspace paths
    best_model.save_model(model_output_path)
    
    with open(metrics_output_path, 'w') as f:
        f.write("=== Full Integrated Multi-Modality Master Model Metrics ===\n")
        f.write(f"Data Partition Layout: Standard Stratified Row Split (train_test_split)\n")
        f.write(f"Total Combined Features Used: {X.shape[1]}\n")
        f.write("-" * 75 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("-" * 75 + "\n")
        f.write("\n=== Best Parameters ===\n")
        for k, v in random_search.best_params_.items():
            f.write(f"{k}: {v}\n")
            
    print(f"[-->] Completed. Production metrics saved directly to: {metrics_output_path}")

if __name__ == "__main__":
    main()
