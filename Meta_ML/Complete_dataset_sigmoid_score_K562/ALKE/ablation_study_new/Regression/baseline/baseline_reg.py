import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

def main():
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Complete_dataset_sigmoid_score_K562/"
    df = pd.read_csv(os.path.join(working_dir, "CRISPR_ml_features_final.csv"))
    
    # Feature Selection: Baseline only
    base_exacts = {'distance_to_TSS', 'MFE', 'Tm', 'entropy', 'GC_count', 'GC_content', 'gc_low', 'gc_high'}
    base_prefixes = ('count_', 'pos_', 'di_')
    selected = [c for c in df.columns if c in base_exacts or c.startswith(base_prefixes)]

    X = df[selected].select_dtypes(include=[np.number])
    y = df['Sigmoid_Score'] # Continuous target for regression
    
    # Train/Test Split (No stratification for regression)
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
    
    # Export
    best.save_model(os.path.join(working_dir, "reg_baseline_model.json"))
    with open(os.path.join(working_dir, "reg_baseline_metrics.txt"), 'w') as f:
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"R2: {r2:.4f}\n")
        f.write(f"Spearman: {spearman_corr:.4f}\n")

if __name__ == "__main__": 
    main()
