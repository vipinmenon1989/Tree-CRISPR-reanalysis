import os
import sys

# ======================================================================
# FORCE HEADLESS HPC RENDER BACKEND
# ======================================================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
# ======================================================================

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, 
    recall_score, f1_score, roc_curve, precision_recall_curve
)

def main():
    # 1. Cluster Workspace Configuration
    working_dir = "./"
    
    # UPDATE THIS to the exact name of your independent test file
    independent_test_file = os.path.join(working_dir, "CRISPRi_ML_Holdout_Test_20_unseen_guides.csv") 
    
    model_input_path = os.path.join(working_dir, "unseen_guide_model.json")
    metrics_output_path = os.path.join(working_dir, "independent_metrics.txt")
    curves_png_path = os.path.join(working_dir, "independent_curves.png")
    curves_pdf_path = os.path.join(working_dir, "independent_curves.pdf")
    predictions_csv_path = os.path.join(working_dir, "independent_predictions.csv")
    
    if not os.path.exists(independent_test_file):
        print(f"CRITICAL ERROR: Independent test file missing at {independent_test_file}")
        sys.exit(1)
    if not os.path.exists(model_input_path):
        print(f"CRITICAL ERROR: Trained model missing at {model_input_path}")
        sys.exit(1)
        
    # 2. Ingest Independent Dataset
    print("[*] Ingesting independent test matrix...")
    df_test = pd.read_csv(independent_test_file, sep=',')
    df_test.columns = [col.lower() for col in df_test.columns]

    # Enforce strict 0.50 threshold to match training logic exactly
    if 'sigmoid_score' in df_test.columns:
        df_test['class'] = (df_test['sigmoid_score'] > 0.5).astype(int)
    elif 'class' not in df_test.columns:
        print("CRITICAL ERROR: Neither 'sigmoid_score' nor 'class' found in independent dataset.")
        sys.exit(1)

    y_test = df_test['class'].values

    # 3. Replicate Feature Space Purge
    print("[*] Purging metadata and broad epigenetic features...")
    explicit_metadata_drops = [
        'unique_sgrna_id', 'id', 'gene', 'sgrna sequence', 'sgrna_sequence',
        'cell_line_origin', 'sigmoid_score', 'class',
        'start', 'start_30', 'end', 'end_30', 'closest_tss_coord', 
        'gene_strand', 'strand', 'guide_strand', 'pam', 'extended_sequence'
    ]
    
    gene_epi_keywords = ['atac', 'methylation', 'cpg', 'h3k']
    gene_epi_drops = [
        col for col in df_test.columns 
        if any(kw in col for kw in gene_epi_keywords) and not col.startswith('guide_')
    ]
    
    all_drops = list(set(explicit_metadata_drops + gene_epi_drops))
    cols_to_drop = [col for col in all_drops if col in df_test.columns]
    
    X_test_raw = df_test.drop(columns=cols_to_drop, errors='ignore').select_dtypes(include=[np.number])

    # 4. Load Pre-trained Architecture and Align Matrix
    print("[*] Loading optimized XGBoost architecture...")
    model = xgb.XGBClassifier()
    model.load_model(model_input_path)
    
    # Extract the exact feature list the model expects
    expected_features = model.get_booster().feature_names
    
    # Check for missing features
    missing_features = [f for f in expected_features if f not in X_test_raw.columns]
    if missing_features:
        print(f"CRITICAL ERROR: Independent dataset is missing {len(missing_features)} features required by the model.")
        print(f"Sample missing: {missing_features[:5]}")
        sys.exit(1)
        
    # Reindex the independent matrix to perfectly match the model's training order
    X_test = X_test_raw[expected_features]
    print(f"--> Matrix aligned. Testing on {X_test.shape[0]} rows x {X_test.shape[1]} features.")

    # 5. Independent Inference
    print("[*] Executing forward pass on independent dataset...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    metrics = {
        "Independent ROC-AUC": roc_auc,
        "Independent PR-AUC (Average Precision)": pr_auc,
        "Independent Precision": precision_score(y_test, y_pred, zero_division=0),
        "Independent Recall": recall_score(y_test, y_pred, zero_division=0),
        "Independent F1 Score": f1_score(y_test, y_pred, zero_division=0)
    }

    # 6. Export Records
    print(f"--> Exporting independent evaluation log to: {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        f.write("=== Sequence + Local Guide Epigenetic Model (Independent Evaluation) ===\n")
        f.write(f"Testing Pool Size: {X_test.shape[0]} rows\n")
        f.write(f"Class 1 (Hit) Count: {np.sum(y_test == 1)}\n")
        f.write(f"Class 0 (Noise) Count: {np.sum(y_test == 0)}\n")
        f.write("-" * 65 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("-" * 65 + "\n")

    # 7. Generate Performance Curves
    print("[*] Rendering evaluation curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax1.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'Model (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=11, fontweight='bold')
    ax1.set_title('Independent ROC Curve', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    baseline_pr = np.sum(y_test == 1) / len(y_test)
    ax2.plot(recall, precision, color='dodgerblue', lw=2.5, label=f'Model (PR-AUC = {pr_auc:.3f})')
    ax2.axhline(y=baseline_pr, color='gray', lw=1.5, linestyle='--', label=f'Baseline Class Ratio (PR = {baseline_pr:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
    ax2.set_title('Independent PR Curve', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc="lower left", frameon=True, facecolor='white', edgecolor='none')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(curves_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(curves_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    # ==================================================================
    # 8. EXPORT GRANULAR PREDICTIONS CSV
    # ==================================================================
    print("[*] Generating granular prediction matrix...")
    
    # Identify the correct strand column name dynamically
    strand_col = 'strand' if 'strand' in df_test.columns else ('guide_strand' if 'guide_strand' in df_test.columns else None)
    
    desired_columns = [
        'unique_sgrna_id', 
        'gene',                 # <--- ADDED GENE COLUMN HERE
        'sgrna sequence', 
        'extended_sequence', 
        'start', 
        'end', 
        strand_col, 
        'sigmoid_score', 
        'class'
    ]
    
    # Keep only the columns that actually exist in the dataframe to avoid KeyErrors
    valid_columns = [col for col in desired_columns if col and col in df_test.columns]
    df_export = df_test[valid_columns].copy()
    
    # Append the model's forward pass arrays
    df_export['probability_score'] = y_proba
    df_export['prediction_binary'] = y_pred
    
    # Save to disk
    df_export.to_csv(predictions_csv_path, index=False)
    print(f"--> Granular predictions saved to: {predictions_csv_path}")

    print(f"[-->] Independent evaluation completed successfully.")

if __name__ == "__main__":
    main()
