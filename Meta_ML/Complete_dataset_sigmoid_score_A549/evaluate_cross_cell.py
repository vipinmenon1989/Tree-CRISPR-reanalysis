import pandas as pd
import numpy as np
import sys
import argparse
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve, 
    auc, 
    confusion_matrix, 
    f1_score, 
    precision_score, 
    recall_score
)

def canonicalize_feature_name(name):
    """
    Transforms any feature name into a cell-line agnostic, lowercase, 
    standardized key to match features across differing naming schemas.
    """
    # Strip any known cell-line identifiers
    n = name.replace("K562_", "").replace("A549_", "").replace("A375_", "")
    # Normalize naming discrepancies (e.g., coverage_bin vs bin)
    n = n.replace("coverage_bin_", "bin_").replace("Coverage_bin_", "bin_")
    return n.strip().lower()

def run_cross_cell_evaluation(data_path, model_path):
    # 1. Load the independent test file
    try:
        df = pd.read_csv(data_path, sep=',')
        print(f"Loaded independent test file: {data_path} ({df.shape[0]} rows)")
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # =========================================================================
    # DYNAMIC CLASS INJECTION LAYER
    # =========================================================================
    if 'class' not in df.columns:
        if 'Sigmoid_Score' in df.columns:
            print("Target column 'class' not found. Generating dynamically from 'Sigmoid_Score' (> 0.25)...")
            df['class'] = (df['Sigmoid_Score'] > 0.25).astype(int)
        else:
            print("CRITICAL ERROR: Neither 'class' nor 'Sigmoid_Score' columns were found in the dataset.")
            sys.exit(1)

    # 2. Isolate ground truth target labels
    y_true = df['class'].astype(int).values

    # 3. Ingest Pre-Trained Native XGBoost Model
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        expected_features = model.get_booster().feature_names
        print(f"XGBoost model loaded. Expecting exactly {len(expected_features)} features natively.")
    except Exception as e:
        print(f"Error loading native XGBoost model: {e}")
        sys.exit(1)

    # =========================================================================
    # CANONICAL FEATURE ALIGNMENT ENGINE
    # =========================================================================
    print("Building canonical feature map to cross-reference K562 model tracking with test data layout...")
    
    # Map out what is currently inside the test dataframe using agnostic keys
    test_column_lookup = {canonicalize_feature_name(col): col for col in df.columns}
    
    # Initialize an empty matrix structurally locked to the model's index length
    X_features = pd.DataFrame(index=df.index)
    missing_cols = []

    # Map the testing values into the target matrix under the model's expected names
    for f in expected_features:
        canon_f = canonicalize_feature_name(f)
        if canon_f in test_column_lookup:
            source_col = test_column_lookup[canon_f]
            X_features[f] = df[source_col]
        else:
            missing_cols.append(f)

    if missing_cols:
        print(f"\nCRITICAL ERROR: Feature Mismatch.")
        print(f"The canonical mapper could not find an equivalent match for {len(missing_cols)} model columns.")
        print(f"Sample missing: {missing_cols[:10]}")
        sys.exit(1)
        
    print(f"Successfully aligned feature matrix layout. Shape: {X_features.shape}")

    # 4. Execute Locked Model Inference
    try:
        y_probs = model.predict_proba(X_features)[:, 1]
        y_pred = model.predict(X_features)
    except Exception as e:
        print(f"Inference Failure: {e}")
        sys.exit(1)

    # =========================================================================
    # METRICS EVALUATION LAYER
    # =========================================================================
    roc_auc = roc_auc_score(y_true, y_probs)
    
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_probs)
    aupr = auc(recall_arr, precision_arr)
    aupr_baseline = sum(y_true) / len(y_true)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fdr = fp / (tp + fp) if (tp + fp) > 0 else 0

    print("\n" + "="*60)
    print("        INDEPENDENT CROSS-CELL PERFORMANCE BENCHMARK REPORT")
    print("="*60)
    print(f"Data Composition:")
    print(f"  --> Total Test Items:     {len(y_true)}")
    print(f"  --> Efficient (Class 1):  {sum(y_true)} ({aupr_baseline*100:.2f}%)")
    print(f"  --> Inefficient (Class 0): {len(y_true)-sum(y_true)} ({(1-aupr_baseline)*100:.2f}%)")
    print("-"*60)
    print(f"Core Model Metrics:")
    print(f"  --> ROC-AUC:              {roc_auc:.4f}")
    print(f"  --> AUPR (PR-AUC):         {aupr:.4f}  [Random Baseline: {aupr_baseline:.4f}]")
    print(f"  --> F1-Score:             {f1:.4f}")
    print(f"  --> Precision:            {precision:.4f}")
    print(f"  --> Recall (Sensitivity): {recall:.4f}")
    print("-"*60)
    print(f"Classification Boundary Audit (Threshold = 0.5):")
    print(f"  --> True Positives (Correct Efficient):   {tp}")
    print(f"  --> True Negatives (Correct Inefficient): {tn}")
    print(f"  --> False Positives (Type I Error):       {fp}")
    print(f"  --> False Negatives (Type II Error):      {fn}")
    print(f"  --> Specificity (True Negative Rate):     {specificity:.4f}")
    print(f"  --> False Discovery Rate (FDR):           {fdr:.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agnostic Feature Cross-Mapping Evaluation')
    parser.add_argument('-d', '--data', required=True, help='Path to test matrix file')
    parser.add_argument('-m', '--model', required=True, help='Path to XGBoost JSON model file')
    
    args = parser.parse_args()
    run_cross_cell_evaluation(args.data, args.model)
