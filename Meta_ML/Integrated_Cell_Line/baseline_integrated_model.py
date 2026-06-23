import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def main():
    # 1. Command-Line Arguments
    parser = argparse.ArgumentParser(description="Train Baseline Sequence XGBoost Model for CRISPRi")
    parser.add_argument("--input", required=True, help="Path to the input CRISPR_merged_features.csv")
    parser.add_argument("--out_dir", required=True, help="Directory to save the model and metrics outputs")
    args = parser.parse_args()

    input_file = args.input
    model_output_path = os.path.join(args.out_dir, "baseline_xgboost_model.json")
    metrics_output_path = os.path.join(args.out_dir, "baseline_metrics.txt")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    
    # 2. Target Definition
    if 'class' not in df.columns:
        df['class'] = (df['Sigmoid_Score'] > 0.25).astype(int)
    
    # 3. Robust Feature Selection: Baseline + Context (Cell_Line)
    base_exacts = {
        'distance_to_TSS', 'MFE', 'Tm', 'entropy', 
        'GC_count', 'GC_content', 'gc_low', 'gc_high', 
        'Cell_Line'  # CRITICAL: Context feature retained
    }
    
    selected_features = []
    for c in df.columns:
        # Check exact matches
        if c in base_exacts:
            selected_features.append(c)
        # Check count features
        elif c.startswith('count_'):
            selected_features.append(c)
        # Check positional features (e.g., 'pos1_A')
        elif c.startswith('pos') and len(c) > 3 and c[3].isdigit():
            selected_features.append(c)
        # Check dinucleotide features (e.g., 'di12_AA')
        elif c.startswith('di') and len(c) > 2 and c[2].isdigit():
            selected_features.append(c)
            
    print(f"Number of baseline features selected: {len(selected_features)}")
    
    X = df[selected_features].select_dtypes(include=[np.number])
    y = df['class']
    
    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 5. Class Imbalance Handling
    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    print(f"Scale Pos Weight (Imbalance Ratio): {ratio:.4f}")
    
    # 6. Model Setup with Strict Regularization
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=ratio,
        random_state=42,
        eval_metric='auc'
    )
    
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
    
    print("Training Baseline Model...")
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    # 7. Evaluation
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "AUPR": average_precision_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }
    
    # 8. Export Outputs
    best_model.save_model(model_output_path)
    
    with open(metrics_output_path, 'w') as f:
        f.write("=== Baseline Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n=== Best Parameters ===\n")
        for k, v in random_search.best_params_.items():
            f.write(f"{k}: {v}\n")
            
    print("\n" + "="*25)
    print("Baseline Metrics")
    print("="*25)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"\nMetrics saved to {metrics_output_path}")
    print(f"Model saved to {model_output_path}")

if __name__ == "__main__":
    main()
