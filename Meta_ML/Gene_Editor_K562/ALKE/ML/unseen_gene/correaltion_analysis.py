import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

def verify_unseen_gene_performance(holdout_raw_path, clash_matrix_path):
    print("======================================================================")
    print("[*] VALIDATION ENGINE: TESTING TRUE VS PREDICTED EPIGENETIC FIT")
    print("======================================================================\n")

    if not os.path.exists(holdout_raw_path) or not os.path.exists(clash_matrix_path):
        print("CRITICAL ERROR: Ground truth holdout or prediction clash matrix missing.")
        sys.exit(1)

    # 1. Ingest datasets
    df_true = pd.read_csv(holdout_raw_path, sep='\t')
    df_pred = pd.read_csv(clash_matrix_path)
    
    # Standardize all columns to lowercase to prevent KeyErrors
    df_true.columns = [c.lower() for c in df_true.columns]
    df_pred.columns = [c.lower() for c in df_pred.columns]

    # Clean up identifiers for matching
    df_true['gene'] = df_true['gene'].str.upper()
    df_pred['gene_name'] = df_pred['gene_name'].str.upper()

    # Isolate active editor columns from true data
    editor_cols = [c for c in df_true.columns if 'editor_' in c]
    
    evaluation_records = []

    # 2. Match true experimental scores against our epigenetic predictions row-by-row
    for idx, row in df_true.iterrows():
        gene = row['gene']
        true_score = row['mean_sigmoid_score']
        
        # Determine which editor was actually used on this raw row
        active_editor = None
        for ed in editor_cols:
            if row[ed] == 1:
                active_editor = ed.replace('editor_', '').lower() # FIX 1: Keep editor name lowercase
                break
                
        if active_editor is None:
            continue
            
        # Extract what our epigenetic heuristic model predicted for this gene-editor combo
        pred_match = df_pred[df_pred['gene_name'] == gene]
        if pred_match.empty:
            continue
            
        # FIX 2: Target the forced lowercase string mapping matching 'estsigmoid_under_alke'
        target_col = f'estsigmoid_under_{active_editor}'
        
        if target_col not in df_pred.columns:
            print(f"WARNING: Predicted column '{target_col}' missing from matrix sheet.")
            continue
            
        pred_score = pred_match.iloc[0][target_col]
        
        evaluation_records.append({
            'Gene': gene,
            'Editor': active_editor.upper(),
            'True_Observed_Score': true_score,
            'Model_Predicted_Score': pred_score
        })

    df_eval = pd.DataFrame(evaluation_records)
    
    if df_eval.empty:
        print("ERROR: Zero matching evaluation pairs found between files.")
        return

    # 3. Compute directional statistical correlation metrics
    p_coeff, p_val = pearsonr(df_eval['True_Observed_Score'], df_eval['Model_Predicted_Score'])
    s_coeff, s_val = spearmanr(df_eval['True_Observed_Score'], df_eval['Model_Predicted_Score'])

    print(f"--- STATISTICAL VALIDATION SUMMARY ({len(df_eval)} HOLD-OUT ASSESSMENT POINTS) ---")
    print(f"--> Pearson Linear Fit R       : {p_coeff:+.4f} (p-value: {p_val:.4e})")
    print(f"--> Spearman Rank Directional R: {s_coeff:+.4f} (p-value: {s_val:.4e})")
    print("-" * 75)

    if s_coeff > 0.25 and s_val < 0.05:
        print("\n[VERDICT]: SUCCESS. The model shows a statistically significant positive correlation.")
        print("           Your 70 independent spatial bin weights are successfully predicting")
        print("           unseen biological chromatin environments using epigenetics alone.")
    else:
        print("\n[VERDICT]: WARNING. Weak or non-significant correlation.")
        print("           The epigenetic signals alone may be compressed, or sequence constraints")
        print("           are heavily dominating these specific 20 loci.")

if __name__ == "__main__":
    working_dir = "./"
    raw_holdout_file = os.path.join(working_dir, "CRISPRi_ML_Holdout_Test_20_unseen_genes.txt")
    prediction_matrix = os.path.join(working_dir, "gene_multi_editor_clash_matrix.csv")
    
    verify_unseen_gene_performance(holdout_raw_path=raw_holdout_file, clash_matrix_path=prediction_matrix)
