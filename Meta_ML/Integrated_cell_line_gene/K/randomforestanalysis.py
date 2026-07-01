import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def execute_scaled_alke_model(file_path):
    print("======================================================================")
    print("[*] ALKE CLASSIFICATION ENGINE: HDLSS CONSTRAINED RANDOM FOREST")
    print("======================================================================\n")

    # 1. Ingest Scaled Data
    df = pd.read_csv(file_path)
    df.columns = [c.lower() for c in df.columns]

    if 'class' not in df.columns or 'gene' not in df.columns:
        print("[!] CRITICAL ERROR: Required columns missing.")
        return

    y = df['class'].values
    groups = df['gene'].values

    # 2. Isolate Pure Scaled Epigenetic Features
    metadata_cols = ['id', 'gene', 'mean_sigmoid_score', 'class']
    drop_cols = [c for c in df.columns if c in metadata_cols]
    
    X_df = df.drop(columns=drop_cols)
    X = X_df.values
    feature_names = X_df.columns.tolist()
    
    print(f"[*] Total Loci (Rows): {len(X)}")
    print(f"[*] Spatial Epigenetic Features: {X.shape[1]}")
    print(f"[*] Class Balance: {sum(y==1)} Edited (1) | {sum(y==0)} Not Edited (0)\n")

    # 3. Initialize HDLSS-Constrained Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300,        # Enough trees to smooth the variance
        max_depth=4,             # Strict shallow limit
        min_samples_leaf=4,      # Forces generalization
        max_features='sqrt',     # Decouples correlated epigenetic marks
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )

    sgkf = StratifiedGroupKFold(n_splits=5)

    oof_predictions = np.zeros(len(X))
    oof_pred_probs = np.zeros(len(X))

    # 4. Execute Isolated Fold Training
    fold = 1
    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        rf_model.fit(X_train, y_train)
        
        oof_predictions[test_idx] = rf_model.predict(X_test)
        oof_pred_probs[test_idx] = rf_model.predict_proba(X_test)[:, 1]
        
        if len(np.unique(y_test)) > 1: 
            fold_auc = roc_auc_score(y_test, oof_pred_probs[test_idx])
            print(f"--> Fold {fold} Out-of-Sample ROC-AUC: {fold_auc:.4f}")
        
        fold += 1

    # 5. Global Aggregate Validation
    print("\n" + "="*70)
    print("FINAL OUT-OF-SAMPLE METRICS (AGGREGATE)")
    print("="*70)
    
    global_auc = roc_auc_score(y, oof_pred_probs)
    print(f"Aggregate ROC-AUC Score: {global_auc:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y, oof_predictions, target_names=["Not Edited (0)", "Edited (1)"]))

    # 6. Extract Top Feature Importances (The Gatekeeper Rules)
    print("\n[*] TOP 10 SPATIAL FEATURES DRIVING ALKE EFFICIENCY:")
    rf_model.fit(X, y) # Fit once on all data just to extract global importances
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(10):
        print(f"  {i+1}. {feature_names[indices[i]]:<40} ({importances[indices[i]]:.4f})")

if __name__ == "__main__":
    merged_scaled_file = "merged_K_dataset.csv" 
    execute_scaled_alke_model(merged_scaled_file)
