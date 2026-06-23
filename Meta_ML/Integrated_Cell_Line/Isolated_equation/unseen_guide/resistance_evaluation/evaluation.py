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
    
    # Point this to your fresh, out-of-sample evaluation data file
    eval_file = os.path.join(working_dir, "CRISPR_ml_features_pure_resistance_perfect_align.csv") 
    
    # CORRECTED: Point directly to your saved unseen guide architecture
    model_json_path = os.path.join(working_dir, "unseen_guide_multimodality_model.json")
    
    # CORRECTED: Suffix markers appended to prevent overwriting your previous logs
    metrics_output_path = os.path.join(working_dir, "unseen_guide_external_eval_metrics.txt")
    curves_png_path = os.path.join(working_dir, "unseen_guide_external_eval_curves.png")
    curves_pdf_path = os.path.join(working_dir, "unseen_guide_external_eval_curves.pdf")
    
    if not os.path.exists(model_json_path):
        print(f"CRITICAL ERROR: Saved model file missing at: {model_json_path}")
        sys.exit(1)
        
    if not os.path.exists(eval_file):
        print(f"CRITICAL ERROR: Evaluation holdout dataset missing at: {eval_file}")
        sys.exit(1)

    # 2. Load and Reconstruct Saved XGBoost Architecture
    print("[*] Reconstituting saved unseen guide multi-modality model from JSON...")
    model = xgb.XGBClassifier()
    model.load_model(model_json_path)
    
    trained_feature_names = model.get_booster().feature_names
    print(f"--> Success. Model expects exactly {len(trained_feature_names)} active input features.")

    # 3. Ingest and Align Evaluation Dataset
    print("[*] Ingesting evaluation target dataset...")
    # Using tab separator to safely parse the .txt holdout format
    df_eval = pd.read_csv(eval_file, sep=',')
    df_eval.columns = [col.lower() for col in df_eval.columns]

    # Target definition from your isolated sigmoid score thresholding
    #df_eval['class'] = (df_eval['sigmoid_score'] > 0.25).astype(int)
    y_eval = df_eval['class'].values

    # Strict feature mapping alignment layer
    print("[*] Aligning evaluation matrix features to trained model signature...")
    try:
        X_eval = df_eval[trained_feature_names].select_dtypes(include=[np.number])
    except KeyError as e:
        missing_cols = [c for c in trained_feature_names if c not in df_eval.columns]
        print(f"CRITICAL ERROR: Evaluation file is missing features required by the model!")
        sys.exit(1)

    print(f"--> Evaluation Matrix Aligned: {X_eval.shape[0]} rows x {X_eval.shape[1]} columns.")

    # 4. Run External Inference Engine
    print("[*] Executing forward pass predictions across evaluation matrix...")
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]

    # Compute comparative metrics
    roc_auc = roc_auc_score(y_eval, y_proba)
    pr_auc = average_precision_score(y_eval, y_proba)
    
    metrics = {
        "External Evaluation ROC-AUC": roc_auc,
        "External Evaluation PR-AUC (Average Precision)": pr_auc,
        "External Evaluation Precision": precision_score(y_eval, y_pred, zero_division=0),
        "External Evaluation Recall": recall_score(y_eval, y_pred, zero_division=0),
        "External Evaluation F1 Score": f1_score(y_eval, y_pred, zero_division=0)
    }

    # 5. Save Performance Metrics Log
    print(f"--> Exporting external inference evaluation log to: {metrics_output_path}")
    with open(metrics_output_path, 'w') as f:
        f.write("=== Unseen Guide Model External Evaluation Metrics Summary ===\n")
        f.write(f"Evaluation Input Matrix File: {os.path.basename(eval_file)}\n")
        f.write(f"Total Evaluated Points: {X_eval.shape[0]} rows\n")
        f.write(f"Aligned Predictive Feature Dimensions: {X_eval.shape[1]}\n")
        f.write("-" * 65 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("-" * 65 + "\n")

    # 6. Generate Performance Curves (ROC & PRC)
    print("[*] Rendering external evaluation performance curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    fpr, tpr, _ = roc_curve(y_eval, y_proba)
    ax1.plot(fpr, tpr, color='forestgreen', lw=2.5, label=f'Model Performance (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.500)')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=11, fontweight='bold')
    ax1.set_title('External Dataset ROC Curve', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    precision, recall, _ = precision_recall_curve(y_eval, y_proba)
    baseline_pr = np.sum(y_eval == 1) / len(y_eval)
    ax2.plot(recall, precision, color='purple', lw=2.5, label=f'Model Performance (PR-AUC = {pr_auc:.3f})')
    ax2.axhline(y=baseline_pr, color='gray', lw=1.5, linestyle='--', label=f'Baseline Class Ratio (PR = {baseline_pr:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11, fontweight='bold')
    ax2.set_title('External Dataset Precision-Recall Curve', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc="lower left", frameon=True, facecolor='white', edgecolor='none')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(curves_png_path, dpi=300, bbox_inches='tight')
    plt.savefig(curves_pdf_path, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"[-->] Complete. External dataset metrics logs successfully saved to {working_dir}")

if __name__ == "__main__":
    main()
