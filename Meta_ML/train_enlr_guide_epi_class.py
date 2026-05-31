import pandas as pd
import numpy as np
import time
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import joblib

def main():
    start_time = time.time()
    
    # Paths
    input_file = "CRISPR_ml_features_final.csv"
    metrics_file = "guide_epi_classification_metrics.txt"
    coef_csv = "guide_epi_top_features_class.csv"
    model_output_path = "best_guide_epi_logistic_elasticnet.pkl"
    
    print(f"Loading dataset from: {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded dataset with dimensions: {df.shape}")
    
    # --- 1. Target Binarization ---
    target_col = 'Sigmoid_Score'
    if target_col not in df.columns:
        raise ValueError(f"Error: Target variable '{target_col}' not found in the dataset.")
        
    y = (df[target_col] >= 0.6).astype(int)
    
    # --- 2. Feature Isolation (CRITICAL) ---
    # Keep only distance_to_TSS, is_K562, and raw guide_ epigenetic bins
    keep_cols = ['distance_to_TSS', 'is_K562']
    keep_existing = [col for col in keep_cols if col in df.columns]
    
    guide_bin_cols = [col for col in df.columns if col.startswith('guide_') and '_bin_' in col]
    print(f"Isolating {len(guide_bin_cols)} raw guide epigenetic bin columns and metadata...")
    
    X_cols = keep_existing + guide_bin_cols
    X = df[X_cols]
    print(f"Isolated feature matrix X dimensions: {X.shape}")
    
    # Train-test split (80/20) stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    
    # --- 3. Data Scaling ---
    # Scale strictly continuous features (distance_to_TSS and guide epigenetic bins), exclude is_K562
    continuous_cols = [col for col in X.columns if col != 'is_K562']
    
    print(f"Scaling {len(continuous_cols)} continuous features...")
    print(f"Leaving is_K562 unscaled...")
    
    scaler = StandardScaler()
    
    X_train_cont_scaled = scaler.fit_transform(X_train[continuous_cols])
    X_test_cont_scaled = scaler.transform(X_test[continuous_cols])
    
    X_train_cont_scaled_df = pd.DataFrame(X_train_cont_scaled, columns=continuous_cols, index=X_train.index)
    X_test_cont_scaled_df = pd.DataFrame(X_test_cont_scaled, columns=continuous_cols, index=X_test.index)
    
    # Recombine scaled continuous with unscaled features
    if 'is_K562' in X.columns:
        X_train_scaled = pd.concat([X_train_cont_scaled_df, X_train[['is_K562']]], axis=1)
        X_test_scaled = pd.concat([X_test_cont_scaled_df, X_test[['is_K562']]], axis=1)
    else:
        X_train_scaled = X_train_cont_scaled_df
        X_test_scaled = X_test_cont_scaled_df
        
    # Re-order columns to match original X
    X_train_scaled = X_train_scaled[X.columns]
    X_test_scaled = X_test_scaled[X.columns]
    
    # --- 4. Model Setup & RandomizedSearchCV ---
    base_lr = LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        class_weight='balanced',
        max_iter=10000,
        n_jobs=-1,
        random_state=42
    )
    
    param_grid = {
        'l1_ratio': np.linspace(0.1, 1.0, 10),
        'C': np.logspace(-4, 2, 7)
    }
    
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Setting up RandomizedSearchCV...")
    random_search = RandomizedSearchCV(
        estimator=base_lr,
        param_distributions=param_grid,
        n_iter=30,
        scoring='roc_auc',
        cv=cv_strategy,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("Starting guide-epigenetic Elastic Net Logistic Regression training...")
    random_search.fit(X_train_scaled, y_train)
    
    print("\nBest Parameters Found:")
    print(random_search.best_params_)
    
    best_model = random_search.best_estimator_
    
    # --- 5. Evaluation ---
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = best_model.predict(X_test_scaled)
    
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
    
    # Save the best model
    print(f"Saving the best model to: {model_output_path}...")
    joblib.dump(best_model, model_output_path)
    
    # Write metrics to a text file
    print(f"Writing metrics to: {metrics_file}...")
    with open(metrics_file, 'w') as f:
        f.write("==================================================\n")
        f.write("GUIDE-EPIGENETIC ELASTIC NET CLASSIFICATION METRICS\n")
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
            
    # --- 6. Coefficient Extraction ---
    print("Extracting model coefficients...")
    coefs = best_model.coef_[0]
    feature_names = X.columns
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefs,
        'Absolute_Magnitude': np.abs(coefs)
    })
    
    # Filter out features forced to exactly 0.0 by the L1 penalty
    active_coefs = coef_df[coef_df['Coefficient'] != 0.0]
    
    # Sort remaining active features by absolute magnitude and save to CSV
    sorted_active = active_coefs.sort_values(by='Absolute_Magnitude', ascending=False)
    print(f"Saving {len(sorted_active)} active features to: {coef_csv}...")
    sorted_active.to_csv(coef_csv, index=False)
    
    elapsed_time = time.time() - start_time
    print(f"\nExecution finished! Total elapsed time: {elapsed_time/60:.2f} minutes.")

if __name__ == "__main__":
    main()
