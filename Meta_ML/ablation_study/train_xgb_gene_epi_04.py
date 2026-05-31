import pandas as pd
import numpy as np
import os
import time
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import joblib

def main():
    start_time = time.time()
    
    # Paths
    input_file = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/CRISPR_ml_features_final.csv"
    output_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/ablation_study/"
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_file = os.path.join(output_dir, "metrics_xgb_gene_epi_04.txt")
    feat_importance_file = os.path.join(output_dir, "xgb_gene_epi_04_feature_importance.csv")
    model_output_path = os.path.join(output_dir, "best_xgb_gene_epi_04_model.pkl")
    
    print(f"Loading dataset from: {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded dataset with dimensions: {df.shape}")
    
    # --- 1. Target Binarization (Threshold 0.4) ---
    target_col = 'Sigmoid_Score'
    if target_col not in df.columns:
        raise ValueError(f"Error: Target variable '{target_col}' not found in the dataset.")
        
    y = (df[target_col] >= 0.4).astype(int)
    
    # --- 2. Feature Selection: Gene-Level Epigenetics Only (Drop ALL guide_ bin columns) ---
    guide_bin_cols = [col for col in df.columns if col.startswith('guide_') and '_bin_' in col]
    print(f"Purging {len(guide_bin_cols)} guide-level raw epigenetic bin columns for Gene-level study...")
    
    X = df.drop(columns=['ID', target_col] + guide_bin_cols)
    print(f"Gene-level feature matrix X dimensions: {X.shape}")
    
    # Train-test split (80/20) stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # --- 3. Model Setup ---
    # Handle class imbalance
    count_0 = np.sum(y_train == 0)
    count_1 = np.sum(y_train == 1)
    scale_pos_weight = count_0 / count_1
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.4f}")
    
    base_xgb = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )
    
    # Hyperparameter Grid
    param_grid = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Setting up RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=base_xgb,
        param_distributions=param_grid,
        n_iter=30,
        scoring='roc_auc',
        cv=cv_strategy,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Starting Gene-Level Epigenetics XGBoost training...")
    random_search.fit(X_train, y_train)
    
    print("\nBest Parameters Found:")
    print(random_search.best_params_)
    
    best_model = random_search.best_estimator_
    
    # --- 4. Evaluation ---
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = best_model.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    aupr = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    
    print(f"\nTest Set Performance Metrics:")
    print(f"ROC-AUC:            {roc_auc:.6f}")
    print(f"AUPR:               {aupr:.6f}")
    print(f"F1 Score:           {f1:.6f}")
    print(f"Precision:          {precision:.6f}")
    print(f"Recall:             {recall:.6f}")
    
    # Save best model
    print(f"Saving the best model to: {model_output_path}...")
    joblib.dump(best_model, model_output_path)
    
    # Save metrics to text file
    print(f"Writing metrics to: {metrics_file}...")
    with open(metrics_file, 'w') as f:
        f.write("==================================================\n")
        f.write("XGBOOST ABLATION GENE-LEVEL EPI (0.4) PERFORMANCE METRICS\n")
        f.write("==================================================\n")
        f.write(f"ROC-AUC:                        {roc_auc:.6f}\n")
        f.write(f"AUPR:                           {aupr:.6f}\n")
        f.write(f"F1 Score:                       {f1:.6f}\n")
        f.write(f"Precision:                      {precision:.6f}\n")
        f.write(f"Recall:                         {recall:.6f}\n\n")
        f.write("==================================================\n")
        f.write("BEST HYPERPARAMETERS FOUND\n")
        f.write("==================================================\n")
        for param, val in random_search.best_params_.items():
            f.write(f"{param:25}: {val}\n")
            
    # --- 5. Feature Importances ---
    print("Extracting feature importances...")
    importances = best_model.feature_importances_
    feature_names = X.columns
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort and save top 40 to CSV
    sorted_importance = importance_df.sort_values(by='Importance', ascending=False).head(40)
    print(f"Saving top 40 feature importances to: {feat_importance_file}...")
    sorted_importance.to_csv(feat_importance_file, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"\nExecution finished! Total elapsed time: {elapsed_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()
