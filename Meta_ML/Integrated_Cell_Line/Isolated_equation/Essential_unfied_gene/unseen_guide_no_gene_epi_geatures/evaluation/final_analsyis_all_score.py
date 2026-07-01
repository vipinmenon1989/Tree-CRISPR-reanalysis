import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score
)

def main():
    # ======================================================================
    # 1. CLUSTER PATH CONFIGURATIONS
    # ======================================================================
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Integrated_Cell_Line/Isolated_equation/Essential_unfied_gene/unseen_guide_no_gene_epi_geatures/evaluation/"
    
    input_file = os.path.join(working_dir, "independent_predictions.txt")
    metrics_file = os.path.join(working_dir, "all_models_comparative_metrics.txt")
    
    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: Source dataset missing: {input_file}")
        sys.exit(1)

    df = pd.read_csv(input_file, sep='\t')
    df.columns = [col.lower().strip() for col in df.columns]
    
    y_true = df['class'].astype(int)
    
    # Calculate true active class density to pin down the quantile target boundary
    true_positive_density = np.sum(y_true == 1) / len(y_true)

    # ======================================================================
    # 2. RUN CO-EVALUATION WITH DYNAMIC BOUNDARIES
    # ======================================================================
    predictors = {
        'XGBoost_Model_Probability': ('prediction_probability', 0.5), # Standard probability floor
        'Biophysical_CGDi_Score': ('cgdi_score', 0.5),                # Bounded scale floor
        # DYNAMIC FIX: Pin SSC threshold to the exact quantile matching active dataset density
        'Sequence_SSC_Score': ('ssc', df['ssc'].quantile(1.0 - true_positive_density)) 
    }

    summary_records = []

    print("[*] Re-evaluating metric matrices using dynamic channel splitting...")
    for label, (col_name, threshold) in predictors.items():
        if col_name not in df.columns:
            continue
            
        y_scores = df[col_name].fillna(0).values
        
        # Calculate continuous area trends
        roc_auc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
        
        # Segment discrete predictions using model-specific thresholds
        y_pred = (y_scores >= threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        summary_records.append({
            'Predictor': label,
            'Eval_Threshold': round(threshold, 4),
            'ROC-AUC': round(roc_auc, 4),
            'AUPR': round(aupr, 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1_Score': round(f1, 4)
        })

    # ======================================================================
    # 3. OVERWRITE COMPARATIVE LOG SHEET
    # ======================================================================
    df_metrics = pd.DataFrame(summary_records)
    df_metrics.to_csv(metrics_file, sep='\t', index=False)
    
    print("\n" + "="*85)
    print("      UPDATED COMPREHENSIVE PERFORMANCE SUMMARY REPORT")
    print("="*85)
    print(df_metrics.to_string(index=False))
    print("="*85)
    print(f"[-->] Completed. Corrected metric matrix saved to: {metrics_file}")

if __name__ == "__main__":
    main()
