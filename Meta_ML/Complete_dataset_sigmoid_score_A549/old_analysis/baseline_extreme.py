import os
import sys
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def main():
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Complete_dataset_sigmoid_score_A549/"
    input_file = os.path.join(working_dir, "CRISPR_ml_features_final.csv")
    
    model_output_path = os.path.join(working_dir, "baseline_xgboost_model_04_tuned.json")
    metrics_output_path = os.path.join(working_dir, "baseline_metrics_04_tuned.txt")
    
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file, sep=',')
    
    # --- THE 0.40 THRESHOLD ---
    df['class'] = (df['Sigmoid_Score'] > 0.40).astype(int)
    
    # --- STRICT SUBTRACTIVE FEATURE SELECTION ---
    cols_to_drop = ['ID', 'Sigmoid_Score', 'class', 'Gene', 'sgRNA Sequence', 'Chromosome', 'Strand', 'PAM', 'extended_sequence', 'closest_TSS_coord']
    X_base = df.drop(columns=cols_to_drop, errors='ignore').select_dtypes(exclude=['object'])

    # Drop ALL Epigenetics (Gene and Guide level)
    epi_cols = [col for col in X_base.columns if 'A549' in col or 'H3K' in col or 'ATAC' in col or 'methylation' in col or 'guide_' in col]
    X = X_base.drop(columns=epi_cols, errors='ignore')
    y = df['class']
    
    print(f"Feature space shape (Baseline): {X.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Dynamic Class Imbalance Ratio
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    print(f"Class imbalance ratio (scale_pos_weight): {ratio:.4f}")
    
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc'
    )
    
    # --- EXTREME IMBALANCE HYPERPARAMETER GRID ---
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.3, 0.5, 0.7],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [1, 5, 10],
        'max_delta_step': [1, 5, 10],   # Crucial for severe imbalance convergence
        'min_child_weight': [1, 2, 3]   # Lowered to allow trees to isolate rare active guides
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_grid,
        n_iter=25,
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1
    )
    
    print("Fitting RandomizedSearchCV with Extreme Imbalance Tuning...")
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    print("Randomized search completed.")
    
    best_model.save_model(model_output_path)
    
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_roc_auc = roc_auc_score(y_test, y_proba)
    test_aupr = average_precision_score(y_test, y_proba)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    print(f"Test ROC-AUC: {test_roc_auc:.6f}")
    print(f"Test AUPR: {test_aupr:.6f}")
    print(f"Test Precision: {test_precision:.6f}")
    print(f"Test Recall: {test_recall:.6f}")
    print(f"Test F1: {test_f1:.6f}")
    
    with open(metrics_output_path, 'w') as f:
        f.write("=== Baseline XGBoost Evaluation (> 0.40, Imbalance Tuned) ===\n")
        f.write(f"ROC-AUC: {test_roc_auc:.6f}\n")
        f.write(f"AUPR: {test_aupr:.6f}\n")
        f.write(f"Precision: {test_precision:.6f}\n")
        f.write(f"Recall: {test_recall:.6f}\n")
        f.write(f"F1 Score: {test_f1:.6f}\n\n")
        f.write("=== Best Hyperparameters ===\n")
        for param, val in random_search.best_params_.items():
            f.write(f"{param}: {val}\n")
            
    print("Done!")

if __name__ == "__main__":
    main()