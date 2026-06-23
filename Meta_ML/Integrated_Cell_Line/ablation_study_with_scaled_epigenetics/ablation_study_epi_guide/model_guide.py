import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def main():
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Integrated_Cell_Line/ablation_study_with_scaled_epigenetics/ablation_study_epi_guide/"
    input_file = os.path.join(working_dir, "CRISPR_ml_features_independent_K562.csv")
    
    # Output paths
    model_output_path = os.path.join(working_dir, "guide_epi_model.json")
    metrics_output_path = os.path.join(working_dir, "guide_epi_metrics.txt")
    shap_summary_plot_path = os.path.join(working_dir, "guide_epi_shap_summary.png")
    shap_bar_plot_path = os.path.join(working_dir, "guide_epi_shap_bar.png")
    shap_csv_path = os.path.join(working_dir, "guide_epi_all_feature_importances.csv")
    
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    
    # Target definition
    if 'class' not in df.columns:
        df['class'] = (df['Sigmoid_Score'] > 0.25).astype(int)
    
    # Deduplicate headers if necessary
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Feature Selection: Baseline Sequence + Guide Epi (Explicitly dropping broad Gene Epi)
    base = [c for c in df.columns if c in {'distance_to_TSS', 'MFE', 'Tm', 'entropy', 'GC_count', 'GC_content', 'gc_low', 'gc_high'} or c.startswith(('count_', 'pos_', 'di_'))]
    guide = [c for c in df.columns if c.startswith('guide_')]
    selected = base + guide
    
    X = df[selected].select_dtypes(include=[np.number])
    y = df['class']
    
    print(f"Total features retained for Guide-Epi Model: {X.shape[1]}")
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Class imbalance handling
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    
    # Model Setup
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc'
    )
    
    # Universal Hyperparameter Grid
    param_grid = {
        'n_estimators': [50, 100], 
        'max_depth': [2, 3], 
        'learning_rate': [0.01, 0.05], 
        'subsample': [0.7], 
        'reg_alpha': [1, 5], 
        'reg_lambda': [5, 10]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        xgb_clf, param_grid, n_iter=15, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42
    )
    
    print("Executing Deep Search on Isolated Feature Space...")
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # Evaluation on internal K562 holdout matrix
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "AUPR": average_precision_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }
    
    # Saving Trained Model and Optimization parameters
    best_model.save_model(model_output_path)
    
    with open(metrics_output_path, 'w') as f:
        f.write("=== Isolated Guide-Epi Feature Model Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n=== Best Parameters ===\n")
        for k, v in search.best_params_.items():
            f.write(f"{k}: {v}\n")
            
    print("\n" + "="*30)
    print("Guide-Epi Model Metrics (Internal Validation)")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # --- SHAP EXPLAINER BLOCK ---
    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer(X_test)
    
    # 1. Generate and save the Summary (Beeswarm) Plot
    print("Generating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Global Feature Importance (Beeswarm Plot - Guide Epi Model)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(shap_summary_plot_path, dpi=300)
    plt.close()
    
    # 2. Generate and save the Bar Plot (Mean Absolute SHAP)
    print("Generating SHAP bar plot...")
    plt.figure(figsize=(12, 8))
    
    # CRITICAL FIX: Slice the SHAP Explanation object directly BEFORE plotting.
    # This forces SHAP to only see the top 20 features, completely preventing 
    # the creation or calculation of the "Sum of ottop_20_shap_values = shap_values[:, :20]
    
    # Plot the sliced object. max_display must match the sliced dimension exactly.
    shap.plots.bar(shap_values, max_display=20, show=False)
    
    plt.title("SHAP Global Feature Importance (Top 20 Individual Features)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(shap_bar_plot_path, dpi=300)
    plt.close()
    
    # 3. Export True, Non-Truncated Numerical Rankings to CSV
    print("Exporting raw feature rankings to CSV...")
    global_importances = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "Feature_Name": X_test.columns,
        "Mean_Absolute_SHAP": global_importances
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
    
    importance_df.to_csv(shap_csv_path, index=False)
    print(f"Complete un-truncated feature rankings saved to: {shap_csv_path}")
    
    print(f"\nSHAP summary plot saved to: {shap_summary_plot_path}")
    print(f"SHAP bar plot saved to: {shap_bar_plot_path}")
    print(f"Metrics saved to: {metrics_output_path}")
    print(f"Model saved to: {model_output_path}")

if __name__ == "__main__":
    main()
