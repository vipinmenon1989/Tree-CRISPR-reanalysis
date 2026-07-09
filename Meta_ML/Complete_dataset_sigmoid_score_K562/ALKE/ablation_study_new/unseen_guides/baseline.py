import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def main():
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Complete_dataset_sigmoid_score_K562/ablation_study_new/unseen_guides/"
    input_file = os.path.join(working_dir, "CRISPR_ml_features_final.csv")
    
    model_output_path = os.path.join(working_dir, "baseline_xgboost_model.json")
    metrics_output_path = os.path.join(working_dir, "baseline_metrics.txt")
    
    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: Matrix file missing from {working_dir}")
        sys.exit(1)
        
    print("[*] Ingesting full master dataset...")
    df = pd.read_csv(input_file)
    
    # Target definition using the baked-in target column rules
    # If the generator script pre-calculated 'class', we use it, otherwise fallback.
    if 'class' not in df.columns:
        df['class'] = (df['Sigmoid_Score'] > 0.25).astype(int)
    
    y = df['class']
    
    # Confirm structural sequence handle is present for the group-split engine
    if 'sgRNA Sequence' not in df.columns:
        print("CRITICAL ERROR: 'sgRNA Sequence' column is missing from features file.")
        sys.exit(1)
    
    # ==================================================================
    # COMPLETE BIOLOGY ABLATION LAYER (THERMODYNAMICS KEEP LAYER)
    # ==================================================================
    # Identify and drop all epigenetic tracks, chromatin windows, and location data
    epigenetic_keywords = ['K562', 'H3K', 'guide_', '_bin_', 'methylation', 'ATAC_seq', 'distance_to_TSS', 'is_K562']
    epigenetic_cols = [col for col in df.columns if any(sub in col for sub in epigenetic_keywords)]
    
    # Administrative string metadata columns
    metadata_cols = ['ID', 'Sigmoid_Score', 'class', 'Gene', 'sgRNA Sequence', 'Chromosome', 'Strand', 'PAM', 'extended_sequence', 'closest_TSS_coord']
    
    # Combine lists BUT ensure 'MFE', 'Tm', and 'entropy' are never caught in the crossfire
    cols_to_drop = list(set(epigenetic_cols + [col for col in metadata_cols if col in df.columns]))
    
    # Isolate matrix (Leaves strictly nucleotide positioning + biophysical properties)
    X = df.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])
    
    print(f"--> Extraction Matrix Isolated: {X.shape[0]} rows x {X.shape[1]} pure sequence features.")
    thermo_check = [c for c in ['MFE', 'Tm', 'entropy'] if c in X.columns]
    print(f"    -> Thermodynamic traits preserved: {thermo_check}")

    # ==================================================================
    # STERILE CROSS-GUIDE GROUP PARTITIONING (80/20 SPLIT)
    # ==================================================================
    # This locks every unique guide string entirely to one side of the validation wall
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=df['sgRNA Sequence']))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    print(f"--> [TRAIN POOL] Rows: {X_train.shape[0]} | Unique sgRNA Sequences: {df.iloc[train_idx]['sgRNA Sequence'].nunique()}")
    print(f"--> [TEST POOL]  Rows: {X_test.shape[0]} | Unique sgRNA Sequences: {df.iloc[test_idx]['sgRNA Sequence'].nunique()}")
    
    # Class imbalance weight mapping
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc',
        n_jobs=-1
    )
    
    # Your exact constrained hyperparameter search configurations
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
    
    print("[*] Running cross-guide sequence baseline optimization...")
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    # Evaluation layer
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0)
    }
    
    # Output persistence mechanics
    best_model.save_model(model_output_path)
    
    with open(metrics_output_path, 'w') as f:
        f.write("=== Honest Cross-Guide Baseline Metrics (Sequence + Thermo) ===\n")
        f.write(f"Data Partition Layout: GroupShuffleSplit by sgRNA Sequence\n")
        f.write(f"Total Unique Evaluation Guides: {df.iloc[test_idx]['sgRNA Sequence'].nunique()}\n")
        f.write("-" * 75 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("-" * 75 + "\n")
        f.write("\n=== Best Selected Parameters ===\n")
        for k, v in random_search.best_params_.items():
            f.write(f"{k}: {v}\n")
            
    print(f"Metrics cleanly saved to {metrics_output_path}")
    print(f"Model cleanly saved to {model_output_path}")

if __name__ == "__main__":
    main()
