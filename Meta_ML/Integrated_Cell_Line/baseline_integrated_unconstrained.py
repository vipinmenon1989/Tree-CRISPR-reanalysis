import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def main():
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Integrated_Cell_Line/"
    input_file = os.path.join(working_dir, "CRISPR_merged_cell_line.csv")
    
    model_output_path = os.path.join(working_dir, "strict_baseline_xgboost_model.json")
    metrics_output_path = os.path.join(working_dir, "strict_baseline_metrics.txt")
    
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Target definition
    if 'class' not in df.columns:
        df['class'] = (df['Sigmoid_Score'] > 0.25).astype(int)
    
    # Feature Selection: STRICTLY Sequence Features
    # 1. Identify all Epigenetic columns to drop (Gene and Guide level)
    epi_keywords = ['K562', 'H3K', 'guide_', 'DNA_methylation', 'ATAC_seq']
    epigenetic_cols = [col for col in df.columns if any(sub in col for sub in epi_keywords)]
    
    # 2. Identify Metadata and Context columns to drop
    metadata_cols = [
        'ID', 'Sigmoid_Score', 'class', 'Cell_Line', 
        'Gene', 'sgRNA Sequence', 'Chromosome', 'Strand', 
        'PAM', 'extended_sequence', 'closest_TSS_coord'
    ]
    
    # Combine and drop
    cols_to_drop = list(set(epigenetic_cols + metadata_cols))
    
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns]).select_dtypes(include=[np.number])
    y = df['class']
    
    print(f"Total pure sequence features remaining for Baseline: {X.shape[1]}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Class imbalance handling
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc'
    )
    
   
   # The Universal Master Grid (Use this exact block for all 3 scripts)
    param_grid = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_depth': [2, 3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
        'reg_alpha': [0, 1, 5, 10],
        'reg_lambda': [5, 10, 20, 30],
        'gamma': [0, 1, 2, 5]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb_clf, param_distributions=param_grid, 
        n_iter=100, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42
    )
    
    print("Executing Deep Search on Strict Baseline...")
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    # Evaluation
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "AUPR": average_precision_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }
    
    # Saving Model and Metrics
    best_model.save_model(model_output_path)
    
    with open(metrics_output_path, 'w') as f:
        f.write("=== Strict Baseline Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n=== Best Parameters ===\n")
        for k, v in random_search.best_params_.items():
            f.write(f"{k}: {v}\n")
            
    print("\n" + "="*30)
    print("Strict Baseline Metrics")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    print("\nBest Parameters Chosen:")
    for k, v in random_search.best_params_.items():
        print(f"  {k}: {v}")
        
    print(f"\nMetrics saved to {metrics_output_path}")
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    main()
