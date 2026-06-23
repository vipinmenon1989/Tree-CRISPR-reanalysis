import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate the pre-trained K562 Guide-Epi model on A549.")
    parser.add_argument("--input", required=True, help="Path to the raw A549 tab-separated file")
    parser.add_argument("--model", required=True, help="Path to the pre-trained guide_epi_model.json file")
    parser.add_argument("--output", required=True, help="Path to save the computed validation metrics text file")
    args = parser.parse_args()

    print(f"Loading A549 dataset from: {args.input}")
    # Read the file (handles comma-separated format based on your file path)
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input, sep=',')
    else:
        df = pd.read_csv(args.input, sep='\t')
    
    # 1. Deduplicate headers if necessary
    df = df.loc[:, ~df.columns.duplicated()]

    # 2. Unconditionally generate/overwrite the binary class column
    if 'Sigmoid_Score' in df.columns:
        df['class'] = (df['Sigmoid_Score'] > 0.25).astype(int)
    elif 'sigmoid_score' in df.columns:
        df['class'] = (df['sigmoid_score'] > 0.25).astype(int)
    else:
        raise KeyError("CRITICAL ERROR: 'Sigmoid_Score' column missing from input file.")

    # 3. EXPLICIT GENE EPI DROP BLOCK
    gene_level_epi = [col for col in df.columns if 'bin_' in col and not col.startswith('guide_')]
    df = df.drop(columns=gene_level_epi)
    print(f"Explicitly stripped {len(gene_level_epi)} broad Gene Epigenetic features from memory.")

    # 4. Resolve A549 biological synonyms for the remaining guide tracks
    new_cols = {}
    for col in df.columns:
        if 'DNA_methylation_coverage_bin' in col:
            new_cols[col] = col.replace('DNA_methylation_coverage_bin', 'DNA_methylation_bin')
        elif 'DNA_methylation_Coverage_bin' in col:
            new_cols[col] = col.replace('DNA_methylation_Coverage_bin', 'DNA_methylation_bin')
        else:
            new_cols[col] = col
    df = df.rename(columns=new_cols)

    # 5. HEADER MASKING BLOCK
    # Convert existing cell line prefixes to match model expectations
    df = df.rename(columns=lambda x: x.replace('A549', 'K562'))
    
    # CRITICAL FIX: Handle the missing prefix anomaly for ATAC seq columns
    # Maps 'guide_ATAC_seq_bin_1' directly to 'guide_K562_ATAC_seq_bin_1'
    df = df.rename(columns=lambda x: x.replace('guide_ATAC_seq_bin_', 'guide_K562_ATAC_seq_bin_'))

    # 6. Load the pre-trained XGBoost classifier
    print(f"Loading trained model from: {args.model}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"CRITICAL ERROR: Model file not found at {args.model}")
        
    model = xgb.XGBClassifier()
    model.load_model(args.model)

    # Extract exact features the model was trained on
    expected_features = model.get_booster().feature_names
    print(f"Model expects exactly {len(expected_features)} true predictor features.")

    # 7. Check for feature coverage in the modified test matrix
    missing_features = [f for f in expected_features if f not in df.columns]
    if missing_features:
        print(f"CRITICAL ERROR: {len(missing_features)} features are missing from the input file matrix.")
        print(f"Sample of missing features: {missing_features[:5]}")
        raise ValueError("Matrix features do not match model training expectations.")

    # Slice and sort the test dataframe columns to match the model's exact input matrix structure
    X_test = df[expected_features].select_dtypes(include=[np.number])
    y_test = df['class']

    # 8. Run Model Inference
    print("Running model inference across aligned A549 predictor matrix...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 9. Compute Metric Values
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "AUPR": average_precision_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

    # 10. Export Metrics to Disk
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write("=== A549 Independent Test Evaluation Metrics ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print("\n" + "="*40)
    print("Independent Test Performance Metrics (A549)")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("="*40)
    print(f"Metrics saved cleanly to: {args.output}\n")

if __name__ == "__main__":
    main()
