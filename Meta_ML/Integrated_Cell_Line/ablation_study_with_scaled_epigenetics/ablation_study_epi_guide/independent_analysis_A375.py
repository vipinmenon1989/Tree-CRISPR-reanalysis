import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate pre-trained K562 Guide-Epi model on the A375 layout.")
    parser.add_argument("--input", required=True, help="Path to the Independent_A375_test.txt file")
    parser.add_argument("--model", required=True, help="Path to the pre-trained guide_epi_model.json file")
    parser.add_argument("--output", required=True, help="Path to save the computed validation metrics text file")
    args = parser.parse_args()

    # Absolute shield preventing raw data corruption via command line typo
    if args.input == args.output:
        raise ValueError("CRITICAL ERROR: Input and output file paths match perfectly. Change --output to avoid wiping out your data.")

    print(f"Loading A375 dataset from: {args.input}")
    df = pd.read_csv(args.input, sep='\t')
    
    # 1. Deduplicate headers if necessary
    df = df.loc[:, ~df.columns.duplicated()]

    # 2. Unconditionally generate/overwrite the binary target class
    if 'Sigmoid_Score' in df.columns:
        df['class'] = (df['Sigmoid_Score'] > 0.25).astype(int)
    elif 'sigmoid_score' in df.columns:
        df['class'] = (df['sigmoid_score'] > 0.25).astype(int)
    else:
        raise KeyError("CRITICAL ERROR: Could not locate 'Sigmoid_Score' column for target categorization.")

    # 3. EXPLICIT GENE EPI DROP BLOCK
    gene_level_epi = [col for col in df.columns if 'bin_' in col and not col.startswith('guide_')]
    df = df.drop(columns=gene_level_epi)
    print(f"Explicitly stripped {len(gene_level_epi)} broad Gene Epigenetic features from memory.")

    # 4. TARGETED HEADER MASKING FOR A375
    # Maps only the specific components that were hardcoded with K562 tokens in the training pool.
    # Leaves the 40 histone mark bins alone to align perfectly with the model layout.
    df = df.rename(columns=lambda x: x.replace('guide_DNA_methylation_', 'guide_K562_DNA_methylation_'))
    df = df.rename(columns=lambda x: x.replace('guide_ATAC_seq_bin_', 'guide_K562_ATAC_seq_bin_'))

    # 5. Load the pre-trained XGBoost classifier
    print(f"Loading trained model from: {args.model}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"CRITICAL ERROR: Model file not found at {args.model}")
        
    model = xgb.XGBClassifier()
    model.load_model(args.model)

    # Extract exact features the model was trained on
    expected_features = model.get_booster().feature_names
    print(f"Model expects exactly {len(expected_features)} true predictor features.")

    # 6. Check for feature coverage in the modified test matrix
    missing_features = [f for f in expected_features if f not in df.columns]
    if missing_features:
        print(f"CRITICAL ERROR: {len(missing_features)} features are missing from the input file matrix.")
        print(f"Sample of missing features: {missing_features[:5]}")
        raise ValueError("Matrix features do not match model training expectations.")

    # Slice and sort the test dataframe columns to match the model's exact input matrix structure
    X_test = df[expected_features].select_dtypes(include=[np.number])
    y_test = df['class']

    # 7. Run Model Inference
    print("Running model inference across aligned A375 predictor matrix...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 8. Compute Metric Values
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "AUPR": average_precision_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

    # 9. Export Metrics to Disk
    if os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
    with open(args.output, 'w') as f:
        f.write("=== A375 Independent Test Evaluation Metrics (Guide-Epi Model) ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print("\n" + "="*40)
    print("Independent Test Performance Metrics (A375)")
    print("="*40)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("="*40)
    print(f"Metrics saved cleanly to: {args.output}\n")

if __name__ == "__main__":
    main()
