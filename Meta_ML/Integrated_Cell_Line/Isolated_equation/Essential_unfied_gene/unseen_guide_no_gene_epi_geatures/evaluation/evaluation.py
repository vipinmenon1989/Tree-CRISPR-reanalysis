import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score
)

def main():
    # 1. Cluster Workspace Configuration
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Integrated_Cell_Line/Isolated_equation/Essential_unfied_gene/unseen_guide_no_gene_epi_geatures/evaluation/"
    
    # Input file dependencies
    model_input_path = os.path.join(working_dir, "unseen_guide_multimodality_model.json")
    holdout_file = os.path.join(working_dir, "CRISPRi_ML_Holdout_Test_20_unseen_guides.txt") # Adjust filename if different
    
    # Output file destinations
    predictions_output_path = os.path.join(working_dir, "prediction.txt")
    holdout_metrics_path = os.path.join(working_dir, "holdout_evaluation_metrics.txt")
    
    # Core system validation checks
    if not os.path.exists(model_input_path):
        print(f"CRITICAL ERROR: Trained model file missing from {model_input_path}")
        sys.exit(1)
    if not os.path.exists(holdout_file):
        print(f"CRITICAL ERROR: Holdout verification file missing from {holdout_file}")
        sys.exit(1)
        
    # 2. Load and Re-Align Model Architecture
    print("[*] Ingesting trained XGBoost booster instance...")
    model = xgb.XGBClassifier()
    model.load_model(model_input_path)
    
    # Extract the exact feature names the model expects
    expected_features = model.get_booster().feature_names
    print(f"--> Model structurally locked to require {len(expected_features)} specific features.")

    # 3. Ingest and Clean Holdout Matrix
    print("[*] Ingesting holdout data matrix...")
    df_holdout = pd.read_csv(holdout_file, sep='\t')
    df_holdout.columns = [col.lower() for col in df_holdout.columns]

    # Dynamically assign class using your PI's stable sigmoid threshold (> 0.25)
    if 'class' not in df_holdout.columns and 'sigmoid_score' in df_holdout.columns:
        df_holdout['class'] = (df_holdout['sigmoid_score'] > 0.25).astype(int)

    # Validate tracking identifiers before structural transformation
    required_metadata = ['unique_sgrna_id', 'sgrna sequence', 'class']
    for meta in required_metadata:
        if meta not in df_holdout.columns:
            print(f"CRITICAL ERROR: Required column '{meta}' missing from holdout header!")
            sys.exit(1)

    # ==================================================================
    # 4. STRIP THE 70 BROAD GENE EPIGENETIC FEATURES
    # ==================================================================
    gene_epi_keywords = ['atac', 'methylation', 'cpg', 'h3k']
    gene_epi_drops = [
        col for col in df_holdout.columns 
        if any(kw in col for kw in gene_epi_keywords) and not col.startswith('guide_')
    ]
    
    print(f"--> Found and removing {len(gene_epi_drops)} broad gene epigenetic features from holdout space.")
    X_holdout = df_holdout[expected_features].select_dtypes(include=[np.number])
    y_holdout = df_holdout['class'].values

    # Final dimension structural alignment assertion
    assert X_holdout.shape[1] == len(expected_features), "Matrix dimension mismatch with model requirements!"
    print(f"--> Holdout Matrix Cleaned: {X_holdout.shape[0]} rows x {X_holdout.shape[1]} features aligned.")

    # ==================================================================
    # 5. FORWARD PASS INFERENCE ON HOLDOUT PORTITION
    # ==================================================================
    print("[*] Running forward pass prediction across holdout vectors...")
    y_pred = model.predict(X_holdout)
    y_proba = model.predict_proba(X_holdout)[:, 1]

    # Calculate rigorous performance benchmarks
    roc_auc = roc_auc_score(y_holdout, y_proba)
    pr_auc = average_precision_score(y_holdout, y_proba)
    
    metrics = {
        "Holdout Validation ROC-AUC": roc_auc,
        "Holdout Validation PR-AUC (Average Precision)": pr_auc,
        "Holdout Validation Precision": precision_score(y_holdout, y_pred, zero_division=0),
        "Holdout Validation Recall": recall_score(y_holdout, y_pred, zero_division=0),
        "Holdout Validation F1 Score": f1_score(y_holdout, y_pred, zero_division=0)
    }

    # ==================================================================
    # 6. CONSTRUCT AND EXPORT CLEAN PREDICTION ROSTER
    # ==================================================================
    print(f"[*] Formatting clean output metrics array... Exporting to {predictions_output_path}")
    
    df_predictions = pd.DataFrame({
        'unique_sgrna_id': df_holdout['unique_sgrna_id'].values,
        'sgrna sequence': df_holdout['sgrna sequence'].values,
        'class': y_holdout,
        'prediction_probability': y_proba,
        'prediction_binary': y_pred
    })
    
    # Save predictions as a clean tab-separated roster
    df_predictions.to_csv(predictions_output_path, sep='\t', index=False)

    # Log evaluation results to a metric summary file
    with open(holdout_metrics_path, 'w') as f:
        f.write("=== Sequence + Local Guide Epigenetic Model: Final Holdout Test Metrics ===\n")
        f.write(f"Evaluated Holdout Pool Size: {X_holdout.shape[0]} rows\n")
        f.write("-" * 75 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("-" * 75 + "\n")

    print("\n" + "="*65)
    print("   FINAL HOLDOUT VERIFICATION EVALUATION COMPLETE")
    print("="*65)
    for k, v in metrics.items():
        print(f" -> {k:<35}: {v:.4f}")
    print("="*65)
    print(f"[-->] Output Roster saved to: {predictions_output_path}")
    print(f"[-->] Metric Report saved to: {holdout_metrics_path}")

if __name__ == "__main__":
    main()
