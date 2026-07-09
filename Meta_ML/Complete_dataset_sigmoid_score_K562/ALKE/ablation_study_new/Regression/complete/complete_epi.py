import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')  # Prevents headless cluster rendering crashes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

def main():
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Complete_dataset_sigmoid_score_K562/"
    df = pd.read_csv(os.path.join(working_dir, "CRISPR_ml_features_final.csv"))
    
    # Feature Selection: Complete
    base = [c for c in df.columns if c in {'distance_to_TSS', 'MFE', 'Tm', 'entropy', 'GC_count', 'GC_content', 'gc_low', 'gc_high'} or c.startswith(('count_', 'pos_', 'di_'))]
    gene = [c for c in df.columns if ('K562' in c or 'H3K' in c) and not c.startswith('guide_')]
    guide = [c for c in df.columns if c.startswith('guide_')]
    selected = base + gene + guide

    X = df[selected].select_dtypes(include=[np.number])
    y = df['Sigmoid_Score']
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Setup
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    param_grid = {
        'n_estimators': [50, 100], 
        'max_depth': [2, 3], 
        'learning_rate': [0.01, 0.05], 
        'subsample': [0.7], 
        'reg_alpha': [1, 5], 
        'reg_lambda': [5, 10]
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(xgb_reg, param_grid, n_iter=15, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, random_state=42)
    search.fit(X_train, y_train)
    
    # Evaluation
    best = search.best_estimator_
    y_pred = best.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    
    # Export Model & Metrics
    best.save_model(os.path.join(working_dir, "reg_complete_model.json"))
    with open(os.path.join(working_dir, "reg_complete_metrics.txt"), 'w') as f:
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")
        f.write(f"Spearman: {spearman_corr:.4f}\n")

    # ==========================
    # SHAP Analysis
    # ==========================
    print("Generating SHAP plots for Regression...")
    explainer = shap.TreeExplainer(best)
    shap_values = explainer.shap_values(X_test)
    
    # 1. SHAP Summary Plot (Dot Plot)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, show=False, max_display=20)
    plt.tight_layout()
    dot_plot_path = os.path.join(working_dir, "reg_complete_shap_summary_dot.png")
    plt.savefig(dot_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP dot plot saved to: {dot_plot_path}")

    # 2. SHAP Bar Plot (Global Mean Absolute Impact)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    bar_plot_path = os.path.join(working_dir, "reg_complete_shap_summary_bar.png")
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP bar plot saved to: {bar_plot_path}")
    print("Processing complete.")

if __name__ == "__main__": 
    main()
