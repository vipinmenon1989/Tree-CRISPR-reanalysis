import os
import sys

# ======================================================================
# FORCE HEADLESS HPC RENDER BACKEND (MUST EXECUTE BEFORE PYPLOT INGEST)
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
    
    # Target configurations pointing explicitly to your unseen gene components
    test_file = os.path.join(working_dir, "CRISPRi_ML_Holdout_Test_20_unseen_genes.txt") 
    model_json_path = os.path.join(working_dir, "unseen_gene_multimodality_model.json")
    
    # Isolated validation output paths to prevent overwriting metrics history
    metrics_output_path = os.path.join(working_dir, "unseen_gene_external_evaluation_metrics.txt")
    curves_png_path = os.path.join(working_dir, "unseen_gene_external_evaluation_curves.png")
    curves_pdf_path = os.path.join(working_dir, "unseen_gene_external_evaluation_curves.pdf")
    
    if not os.path.exists(model_json_path):
        print(f"CRITICAL ERROR: Saved booster file missing at: {model_json_path}")
        sys.exit(1)
        
    if not os.path.exists(test_file):
        print(f"CRITICAL ERROR: Pure gene holdout test file missing at: {test_file}")
        sys.exit(1)

    # 2. Reconstruct Model Layer out of JSON Architecture
    print("[*] Reconstituting saved model from target JSON layout...")
    model = xgb.XGBClassifier()
    model.load_model(model_json_path)
    
    # Extract the exact feature list footprint the model was compiled with
    trained_feature_names = model.get_booster().feature_names
    print(f"--> Success. Model expects exactly {len(trained_feature_names)} active features.")

    # 3. Ingest and Align Holdout Test Matrix
    print("[*] Ingesting holdout test data matrix...")
    df_test = pd.read_csv(test_file, sep='\t')
    df_test.columns = [col.lower() for col in df_test.columns]

    # Target assignment matching your stable sigmoid score equation
    df_test['class'] = (df_test['sigmoid_score'] > 0.25).astype(int)
    y_test = df_test['class'].values

    # Strict feature mapping alignment gate
    print("[*] Re-ordering and aligning test matrix columns to trained signature...")
    try:
        X_test = df_test[trained_feature_names].select_dtypes(include=[np.number])
    except KeyError as e:
        missing_cols = [c for c in trained_feature_names if c not in df_test.columns]
        print(f"CRITICAL SHAPE FAULT: Test file lacks feature layers required by the booster!")
        print(f"First 5 missing entries: {missing_cols[:5]}")
        sys.exit(1)

    print(f"--> Clean Test Submatrix Isolated: {X_test.shape[0]} rows x {X_test.shape[1]} features.")

    # 4. Run Forward Pass Inference Engine
    print("[*] Executing forward pass predictions across the out-of-sample matrix...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Compute operational validation parameters
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    
    metrics = {
        "Test Dataset ROC-AUC": roc_auc,
        "Test Dataset PR-AUC (Average Precision)": pr_auc,
        "Test Dataset Precision": precision_score(y_test, y_pred, zero_division=0),
        "Test Dataset Recall": recall_score(y_test, y_pred, zero_division=0),
        "Test Dataset F1 Score": f1_score(y_test, y_pred, zero_division=0)
    }

    # 5. Export Logs to Disk
    print(f"--> Exporting comprehensive evaluation summaries to: {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        f.write("=== Pure Holdout Test Dataset Inference Summary (Unseen Genes) ===\n")
        f.write(f"Evaluated Matrix Target File: {os.path.basename(test_file)}\n")
        f.write(f"Evaluated Model Source:      {os.path.basename(model_json_path)}\n")
        f.write(f"Total Test Data Rows:        {X_test.shape[0]}\n")
        f.write(f"Active Feature Footprint:    {X_test.shape[1]}\n")
        f.write("-" * 65 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("-" * 65 + "\n")

    # 6. Generate Performance Curves (ROC & PRC)
    print("[*] Rendering performance evaluation curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax1.plot(fpr, tpr, color='forestgreen', lw=2.5, label=f'Model Performance (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=11, fontweight='bold')
    ax1.set_title('Test Dataset ROC Curve (Unseen Genes)', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    baseline_pr = np.sum(y_test == 1) / len(y_test)
    ax2.plot(recall, precision, color='purple', lw=2.5, label=f'Model Performance (PR-AUC = {pr_auc:.3f})')
    ax2.axhline(y=baseline_pr, color='gray', lw=1.5, linestyle='--', label=f'Baseline Class Ratio (PR = {baseline_pr:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
    ax2.set_title('Test Dataset Precision-Recall Curve (Unseen Genes)', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc="lower left", frameon=True, facecolor='white', edgecolor='none')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(curves_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(curves_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"[-->] Complete. Performance logs and vector figures written inside {working_dir}")

if __name__ == "__main__":
    main()
