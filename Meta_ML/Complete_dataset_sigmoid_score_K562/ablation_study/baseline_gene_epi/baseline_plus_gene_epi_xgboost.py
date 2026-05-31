import os
import sys
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def main():
    # 2. Define the working directory and input file path
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Complete_dataset_sigmoid_score_K562/"
    input_file = os.path.join(working_dir, "CRISPR_ml_features_final.csv")
    
    model_output_path = os.path.join(working_dir, "gene_epi_xgboost_model.json")
    metrics_output_path = os.path.join(working_dir, "gene_epi_metrics.txt")
    
    print(f"Loading dataset from: {input_file}")
    # 4. Load the dataset
    df = pd.read_csv(input_file, sep=',')
    
    # Create a binary target column
    df['class'] = (df['Sigmoid_Score'] > 0.25).astype(int)
    
    # 5. Feature selection logic for ABLATION STEP 2 (Baseline + Gene-level Epigenetics)
    # EXCLUDE all guide-specific epigenetic features: drop any columns containing 'guide_'
    guide_cols = [col for col in df.columns if 'guide_' in col]
    
    # Drop non-predictive metadata and target columns
    metadata_cols = ['ID', 'Sigmoid_Score', 'class', 'Gene', 'sgRNA Sequence', 'Chromosome', 'Strand', 'PAM', 'extended_sequence', 'closest_TSS_coord']
    
    # Combine columns to drop
    cols_to_drop = list(set(guide_cols + [col for col in metadata_cols if col in df.columns]))
    X_temp = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Retain baseline features AND the gene-level epigenetic features (columns containing 'K562' or 'H3K' that survived)
    baseline_prefixes = ('count_', 'pos_', 'di_')
    baseline_exacts = {'distance_to_TSS', 'MFE', 'Tm', 'entropy', 'GC_count', 'GC_content', 'gc_low', 'gc_high'}
    
    feature_cols = []
    for col in X_temp.columns:
        # Retain if it is a baseline feature
        if col in baseline_exacts or col.startswith(baseline_prefixes):
            feature_cols.append(col)
        # Retain if it is a gene-level epigenetic feature
        elif 'K562' in col or 'H3K' in col:
            feature_cols.append(col)
            
    X = X_temp[feature_cols]
    
    # Drop any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    y = df['class']
    
    print(f"Baseline + Gene-level Epigenetics feature space shape: {X.shape}")
    
    # 6. Data splitting and heavily regularized modeling pipeline logic
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Calculate the imbalance ratio
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    print(f"Class imbalance ratio (scale_pos_weight): {ratio:.4f}")
    
    # Initialize XGBClassifier
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc'
    )
    
    # Define exact same hyperparameter grid used in baseline to ensure fair comparison
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.3, 0.5, 0.7],
        'reg_alpha': [0, 0.1, 1, 10],
        'reg_lambda': [1, 5, 10]
    }
    
    # Setup CV and RandomizedSearchCV
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
    
    print("Fitting RandomizedSearchCV on training data...")
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    print("Randomized search completed.")
    
    # 7. Evaluation and export logic
    # Save the trained best model to disk as gene_epi_xgboost_model.json
    print(f"Saving best model to: {model_output_path}")
    best_model.save_model(model_output_path)
    
    # Generate test set predictions and probabilities
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate test set metrics
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
    
    # Write calculated metrics and best parameters to gene_epi_metrics.txt
    print(f"Writing metrics to: {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        f.write("=== Baseline + Gene-Level Epigenetic XGBoost Model Evaluation (Ablation Step 2) ===\n")
        f.write(f"ROC-AUC: {test_roc_auc:.6f}\n")
        f.write(f"AUPR (Average Precision): {test_aupr:.6f}\n")
        f.write(f"Precision: {test_precision:.6f}\n")
        f.write(f"Recall: {test_recall:.6f}\n")
        f.write(f"F1 Score: {test_f1:.6f}\n\n")
        f.write("=== Best Hyperparameters ===\n")
        for param, val in random_search.best_params_.items():
            f.write(f"{param}: {val}\n")
            
    print("Done!")

if __name__ == "__main__":
    main()
