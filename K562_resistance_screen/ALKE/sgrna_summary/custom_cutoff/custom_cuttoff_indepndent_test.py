import pandas as pd
import argparse
import sys

def filter_and_extract_sgrnas(input_file, hits_output, dist_output, ml_delta_output):
    # 1. Load sgRNA-level data from File 1 (Preserving tab separation)
    try:
        df = pd.read_csv(input_file, sep='\t')
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        sys.exit(1)

    # Required columns for the downstream analysis matrix
    cols = ['sgrna', 'Gene', 'Score_RIG', 'Delta_Score']
    
    # Structural safety fallback if nomenclature varies
    for col in cols:
        if col not in df.columns and col == 'Score_RIG' and 'Sigmoid_Score' in df.columns:
            df.rename(columns={'Sigmoid_Score': 'Score_RIG'}, inplace=True)

    # 2. Identify "Qualified" sgRNAs for Positive Hits
    qualified_mask = (df['Score_RIG'] > 0.25) & (df['Delta_Score'] > 0.1)
    df_qualified = df[qualified_mask].copy()

    # 3. Apply the Replication Filter (at least 2 qualified sgRNAs per gene)
    gene_counts = df_qualified.groupby('Gene').size()
    significant_genes = gene_counts[gene_counts >= 2].index
    
    # 4. Extract Final Hits (Positive Class) -> Your 2,735 rows
    final_hits_df = df_qualified[df_qualified['Gene'].isin(significant_genes)][cols].copy()
    target_count = len(final_hits_df)  

    # --- OUTPUT 1: Significant_Resistance_Hits_sgRNA.csv ---
    final_hits_df.to_csv(hits_output, index=False)

    # --- OUTPUT 2: Hits_plus_NTC_Distribution.csv ---
    ntc_mask = df['Gene'].str.contains('^NO-TARGET|NTC', na=False, case=False, regex=True)
    df_ntc = df[ntc_mask][cols].copy()
    
    distribution_df = pd.concat([final_hits_df, df_ntc], axis=0)
    distribution_df.to_csv(dist_output, index=False)

    # Separate non-NTC targets for negative class baseline extraction
    target_gene_df = df[~ntc_mask].copy()

    # --- OUTPUT 4: ML Delta Rig Lower Added Dataset (Labeled Independent Test Set) ---
    # Class 0 baseline: Score_RIG <= 0.05
    strict_zero_mask = target_gene_df['Score_RIG'] <= 0.05
    strict_zero_pool = target_gene_df[strict_zero_mask][cols].copy()

    # Force exact balance against the hits pool
    if len(strict_zero_pool) >= target_count:
        balanced_strict_zeros = strict_zero_pool.sample(n=target_count, random_state=42).copy()
    else:
        print(f"Warning: Only {len(strict_zero_pool)} lower guides found. Cannot reach exact 1:1 balance.")
        balanced_strict_zeros = strict_zero_pool.copy()

    # =========================================================================
    # CLASS COLUMN INJECTION LAYER
    # =========================================================================
    # Isolate and assign Class 1 to the hits matrix slice
    final_hits_ml = final_hits_df.copy()
    final_hits_ml['class'] = 1

    # Assign Class 0 to the sampled strict background zeros slice
    balanced_strict_zeros['class'] = 0

    # Combine into the final ML evaluation test matrix containing the 'class' vector
    ml_delta_df = pd.concat([final_hits_ml, balanced_strict_zeros], axis=0).reset_index(drop=True)
    ml_delta_df.to_csv(ml_delta_output, index=False)

    print("-" * 40)
    print(f"Decision Layer Processed Successfully")
    print(f"Total Significant Genes (n>=2): {len(significant_genes)}")
    print(f"1. Hits File saved: {len(final_hits_df)} rows -> {hits_output}")
    print(f"2. QC Distribution saved: {len(distribution_df)} rows -> {dist_output}")
    print(f"3. Balanced Labeled Test Matrix saved: {len(ml_delta_df)} rows ({len(final_hits_ml)} Hits [Class 1] + {len(balanced_strict_zeros)} Lower Negatives [Class 0]) -> {ml_delta_output}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRISPRi Phase 2: sgRNA-Level Hit Extraction')
    parser.add_argument('-i', '--input', required=True, help='Output from File 1 (sgRNA level)')
    parser.add_argument('-o_hits', '--out_hits', default='Significant_Resistance_Hits_sgRNA.csv', help='Individual hits')
    parser.add_argument('-o_dist', '--out_dist', default='Hits_plus_NTC_Distribution.csv', help='Density Plot input')
    parser.add_argument('-o_delta', '--out_delta', default='ML_delta_rig_lower_added.csv', help='Balanced labeled test matrix output')
    
    args = parser.parse_args()
    filter_and_extract_sgrnas(args.input, args.out_hits, args.out_dist, args.out_delta)