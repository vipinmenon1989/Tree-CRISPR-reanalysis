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
    
    model_output_path = os.path.join(working_dir, "baseline_xgboost_model.json")
    metrics_output_path = os.path.join(working_dir, "baseline_metrics.txt")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Error: {input_file} not found.")
        
    df = pd.read_csv(input_file)
    
    # 1. Target Definition
    if 'class' not in df.columns:
        raise KeyError("Error: Pre-calculated target label column 'class' missing from data.")
    y = df['class']
    
    # 2. Strict Feature Isolation (Removing ALL Epigenetic & Non-Sequence Structural Columns)
    # This explicit mapping targets all '_bin_' variants and genomic spatial context markers
    epigenetic_keywords = [
        '_bin_',          # Catches ATAC_seq_bin, H3K*, DNA_methylation_bin, etc.
        'guide_',         # Catches all guide-specific epigenetic bins
        'distance_to_TSS' # Spatial genomic context marker (Not a pure sequence feature)
    ]
    
    epigenetic_cols = [col for col in df.columns if any(key in col for key in epigenetic_keywords)]
    
    # Non-numeric metadata / Target drop parameters
    metadata_cols = ['ID', 'Sigmoid_score', 'class']
    
    # Consolidate complete drop list
    cols_to_drop = list(set(epigenetic_cols + [col for col in metadata_cols if col in df.columns]))
    
    # Construct clean training matrix containing ONLY sequence metrics & structural thermodynamics
    X = df.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])
    
    print(f"=== Clean Sequence Baseline Matrix Verified ===")
    print(f"Total features retained for XGBoost: {X.shape[1]}")
    print(f"Target distribution: Positive Class = {np.sum(y == 1)}, Negative Class = {np.sum(y == 0)}")
    
    # 3. Splitting Strategy
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Handle class imbalance explicitly via scaling weight ratio
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc'
    )
    
    # 4. Constrained Hyperparameter Tuning
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
    
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    # 5. Validation Execution
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "PR-AUC (Avg Precision)": average_precision_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0)
    }
    
    # 6. Saving Model and Validation Outputs
    best_model.save_model(model_output_path)
    
    with open(metrics_output_path, 'w') as f:
        f.write("=== Pure Sequence + Thermo Baseline Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n=== Best Parameters ===\n")
        for k, v in random_search.best_params_.items():
            f.write(f"{k}: {v}\n")
            
    print(f"Metrics written to {metrics_output_path}")
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    main()
