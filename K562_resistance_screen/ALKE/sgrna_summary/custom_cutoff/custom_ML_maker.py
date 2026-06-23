import os
import sys
import argparse
import pandas as pd

def filter_and_extract_sgrnas(input_file, hits_output, dist_output, ml_output):
    # 1. Load data matrix from disk
    try:
        df = pd.read_csv(input_file, sep=None, engine='python')
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to ingest {input_file}: {e}")
        sys.exit(1)

    # Standardize headers immediately to handle entirely different screening data structures
    df.columns = [str(col).strip().lower() for col in df.columns]

    # Verify tracking columns are present
    required_cols = ['score_rig', 'delta_score', 'gene']
    for col in required_cols:
        if col not in df.columns:
            print(f"CRITICAL ERROR: Missing expected baseline column: '{col}'")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)

    # Identify and isolate Non-Targeting Controls globally using regex anchors
    ntc_mask = df['gene'].str.contains('^no-target|ntc', na=False, case=False, regex=True)

    # ==================================================================
    # PHASE A: FILE 1 & FILE 2 LOGIC (REPLICATION EXTRACTION)
    # ==================================================================
    qualified_pos_mask = (df['score_rig'] > 0.25) & (df['delta_score'] > 0.25)
    df_qualified_pos = df[qualified_pos_mask].copy()

    # Apply n>=2 replication filter at the gene level for positives
    pos_gene_counts = df_qualified_pos.groupby('gene').size()
    significant_pos_genes = pos_gene_counts[pos_gene_counts >= 2].index
    
    # Export File 1 (Significant Hits Only)
    final_hits_sgrna = df_qualified_pos[df_qualified_pos['gene'].isin(significant_pos_genes)].copy()
    final_hits_sgrna.to_csv(hits_output, index=False)

    # Export File 2 (Hits + NTC Background)
    df_ntc = df[ntc_mask].copy()
    distribution_df = pd.concat([final_hits_sgrna, df_ntc], axis=0)
    distribution_df.to_csv(dist_output, index=False)

    # ==================================================================
    # PHASE B: STANDALONE FILE 3 LOGIC (UN-ABIFURCATED IMPERFECT POOL)
    # ==================================================================
    print("[*] Filtering independent asymmetric ML dataset (Preserving ALL functional rows)...")
    
    # --- POOL 1: ALL Validated Active Positive Genes (Class 1) ---
    # Capturing the full 1472 array with zero down-sampling ablation
    ml_positives = final_hits_sgrna.copy()
    ml_positives['class'] = 1
    
    # --- POOL 2: ALL Validated Inactive Negative Genes (Class 0) ---
    # Threshold condition: Score_RIG <= 0.10 AND Delta_Score <= 0.10
    # Biological constraint: MUST NOT be NTCs (proper genes only)
    qualified_neg_mask = (df['score_rig'] <= 0.15) & (df['delta_score'] <= 0.15) & (~ntc_mask)
    df_qualified_neg = df[qualified_neg_mask].copy()
    
    # Replication constraint: At least 2 sgRNAs per proper gene in the inactive pool
    neg_gene_counts = df_qualified_neg.groupby('gene').size()
    significant_neg_genes = neg_gene_counts[neg_gene_counts >= 2].index
    
    ml_negatives = df_qualified_neg[df_qualified_neg['gene'].isin(significant_neg_genes)].copy()
    ml_negatives['class'] = 0

    # Calculate downstream parameters post-extraction
    pos_count = len(ml_positives)
    neg_count = len(ml_negatives)
    total_count = pos_count + neg_count

    if pos_count == 0 or neg_count == 0:
        print("CRITICAL FAULT: One of your filtered training pools has 0 rows. File 3 aborted.")
        sys.exit(1)

    # Direct concatenation of full arrays with NO random sampling reduction
    final_ml_dataset = pd.concat([ml_positives, ml_negatives], axis=0).reset_index(drop=True)
    
    # Export File 3
    final_ml_dataset.to_csv(ml_output, index=False)

    # Calculate the exact scale_pos_weight parameter ratio: negative_count / positive_count
    suggested_scale_pos_weight = float(neg_count) / pos_count

    # Cluster Verification Report
    print("-" * 65)
    print("PHASE 2 PROCESSING COMPLETE WITH ASYMMETRIC CLASS PRESERVATION")
    print(f"--> File 1 (Significant Hits Array): {pos_count} rows -> {hits_output}")
    print(f"--> File 2 (Density Curve Source):   {len(distribution_df)} rows -> {dist_output}")
    print(f"--> File 3 (Full Imbalanced ML Matrix): {total_count} rows -> {ml_output}")
    print(f"    * Active Hits (Class 1):          {pos_count} rows")
    print(f"    * Inactive Baselines (Class 0):    {neg_count} rows")
    print("-" * 65)
    print(f"[-->] CRITICAL PARAMETER FOR YOUR XGBOOST CONFIG:")
    print(f"      Set scale_pos_weight = {suggested_scale_pos_weight:.4f}")
    print("-" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRISPRi Phase 2: Autonomous Complete Dataset Engine')
    parser.add_argument('-i', '--input', required=True, help='Raw screening file matrix input')
    parser.add_argument('-o_hits', '--out_hits', default='Significant_Resistance_Hits_sgRNA_global_tuned.csv', help='File 1 output destination')
    parser.add_argument('-o_dist', '--out_dist', default='Hits_plus_NTC_Distribution_global_tuned.csv', help='File 2 output destination')
    parser.add_argument('-o_ml', '--out_ml', default='CRISPR_ml_complete_imbalanced_pool.csv', help='File 3 standalone output destination')
    
    args = parser.parse_args()
    filter_and_extract_sgrnas(args.input, args.out_hits, args.out_dist, args.out_ml)
