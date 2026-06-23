import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score

def main():
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained XGBoost model on the independent A375 dataset.")
    parser.add_argument("--input", required=True, help="Path to the formatted independent A375 tab-separated file")
    parser.add_argument("--model", required=True, help="Path to the pre-trained full_model_xgboost_model.json file")
    parser.add_argument("--output", required=True, help="Path to save the computed validation metrics text file")
    args = parser.parse_args()

    print(f"Loading independent test dataset from: {args.input}")
    df = pd.read_csv(args.input, sep='\t')
    
    # 1. Deduplicate headers if necessary (matching training workflow)
    df = df.loc[:, ~df.columns.duplicated()]

    # Verify target class presence
    if 'class' not in df.columns:
        raise KeyError("CRITICAL ERROR: 'class' column not found. Ensure you ran the formatting script first.")

    # 2. Load the pre-trained XGBoost classifier
    print(f"Loading trained model from: {args.model}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"CRITICAL ERROR: Model file not found at {args.model}")
        
    model = xgb.XGBClassifier()
    model.load_model(args.model)

    # Extract exact features the model was trained on to guarantee structural alignment
    expected_features = model.get_booster().feature_names
    print(f"Model expects exactly {len(expected_features)} true predictor features.")

    # 3. Mirror the Training Feature Selection logic exactly
    # Drops non-predictive strings and coordinate metadata columns to isolate true predictors
    metadata_cols = ['ID', 'Sigmoid_Score', 'class', 'Cell_Line', 
        'Gene', 'sgRNA Sequence', 'Chromosome', 'Strand', 
        'PAM', 'extended_sequence', 'closest_TSS_coord'
    ]
    
    # Extract numerical matrices exactly like the training step
    X_test_clean = df.drop(columns=[col for col in metadata_cols if col in df.columns]).select_dtypes(include=[np.number])
    y_test = df['class']

    # 4. Enforce strict alignment and sequencing with the expected features
    missing_features = [f for f in expected_features if f not in X_test_clean.columns]
    if missing_features:
        print(f"CRITICAL ERROR: {len(missing_features)} features are missing from the input file matrix.")
        print(f"Sample of missing features: {missing_features[:5]}")
        raise ValueError("Matrix features do not match model training expectations.")

    # Slice and sort the test dataframe columns to match the model's exact input matrix structure
    X_test = X_test_clean[expected_features]

    # 5. Run Model Inference
    print("Running model inference across aligned A375 predictor matrix...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # 6. Compute Metric Values
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "AUPR": average_precision_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

    # 7. Export Metrics to Disk
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        f.write("=== A375 Independent Test Evaluation Metrics ===\n")
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
