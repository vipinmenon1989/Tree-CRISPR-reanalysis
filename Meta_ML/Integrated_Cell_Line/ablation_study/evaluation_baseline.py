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

def standardize_test_columns(col, cell_prefix='A375_'):
    """
    Strips cell-line prefixes to align test data with abstract training headers.
    """
    if col.startswith(cell_prefix):
        return col.replace(cell_prefix, "", 1)
    elif col.startswith(f"guide_{cell_prefix}"):
        return col.replace(f"guide_{cell_prefix}", "guide_", 1)
    return col

def resolve_biological_synonyms(col):
    """
    Standardizes case-insensitive methylation nomenclature variants.
    """
    if 'DNA_methylation_Coverage_bin_' in col:
        return col.replace('DNA_methylation_Coverage_bin_', 'DNA_methylation_bin_')
    if 'DNA_methylation_coverage_bin_' in col:
        return col.replace('DNA_methylation_coverage_bin_', 'DNA_methylation_bin_')
    return col

def run_sequence_evaluation(data_path, model_path):
    # 1. Load the independent test file
    try:
        df = pd.read_csv(data_path, sep=',')
        print(f"Loaded independent test file: {data_path} ({df.shape[0]} rows)")
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # 2. Isolate ground truth target labels
    if 'class' not in df.columns:
        print("Error: Target 'class' column not found in data.")
        sys.exit(1)
    y_true = df['class'].astype(int).values

    # 3. Apply Naming Rules to headers
    print("Applying cell-line prefix standardization and synonym mapping...")
    df.columns = [standardize_test_columns(c, "A375_") for c in df.columns]
    df.columns = [resolve_biological_synonyms(c) for c in df.columns]

    # 4. Ingest Pre-Trained Baseline XGBoost Model
    try:
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        expected_features = model.get_booster().feature_names
        print(f"XGBoost model loaded. Expecting exactly {len(expected_features)} features natively.")
    except Exception as e:
        print(f"Error loading native XGBoost model: {e}")
        sys.exit(1)

    # 5. Extract and Order Expected Features
    try:
        X_features = df[expected_features].copy()
    except KeyError as e:
        missing_cols = set(expected_features) - set(df.columns)
        print(f"\nCRITICAL ERROR: Feature Mismatch.")
        print(f"The model expects columns missing from your cleaned test data: {list(missing_cols)[:10]}...")
        sys.exit(1)

    # =========================================================================
    # STRICT EPIGENETIC OMISSION LAYER
    # =========================================================================
    # Identify any epigenetic features based on standard naming tokens
    epigenetic_keywords = ['bin', 'methylation', 'atac', 'h3k', 'distance_to_tss']
    epi_features_in_model = [f for f in X_features.columns if any(k in f.lower() for k in epigenetic_keywords)]
    
    if epi_features_in_model:
        print(f"Enforcing Omission: Zero-masking {len(epi_features_in_model)} epigenetic features found in model booster.")
        X_features[epi_features_in_model] = 0.0
    else:
        print("Confirmed: Loaded model structure contains zero epigenetic features. Evaluating pure sequence space.")

    print(f"Aligned evaluation matrix shape: {X_features.shape}")

    # 6. Execute Locked Model Inference
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
    print("        INDEPENDENT PERFORMANCE BENCHMARK REPORT (SEQUENCE ONLY)")
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
    parser = argparse.ArgumentParser(description='Evaluate Baseline Sequence Features on Native XGBoost Model')
    parser.add_argument('-d', '--data', required=True, help='Path to A375 test file')
    parser.add_argument('-m', '--model', required=True, help='Path to baseline XGBoost JSON model file')
    
    args = parser.parse_args()
    run_sequence_evaluation(args.data, args.model)
