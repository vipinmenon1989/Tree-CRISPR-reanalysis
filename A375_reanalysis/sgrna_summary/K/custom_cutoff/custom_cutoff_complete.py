import sys
import pandas as pd

def run_analysis(input_file, hits_output, dist_output):
    df = pd.read_csv(input_file)

    # 1. Identify Hits (Score > 0.25 AND Gene count >= 2)
    qualified_mask = df['Sigmoid_Score'] > 0.25
    qualified_sgrnas = df[qualified_mask].copy()
    gene_counts = qualified_sgrnas.groupby('Gene').size()
    significant_genes = gene_counts[gene_counts >= 2].index
    
    # Filter and select only the 3 required columns
    cols = ['sgrna', 'Gene', 'Sigmoid_Score']
    final_hits_df = qualified_sgrnas[qualified_sgrnas['Gene'].isin(significant_genes)][cols]

    # 2. Extract NTCs and select only the 3 required columns
    ntc_mask = df['Gene'].str.contains('NO-TARGET|NTC', case=False, na=False)
    ntc_df = df[ntc_mask][cols]

    # 3. Hits + ALL NTCs -> ALKE_hits_NTC_new_cutoff_complete.csv
    hits_with_ntc = pd.concat([final_hits_df, ntc_df], axis=0)
    hits_with_ntc.to_csv(dist_output, index=False)

    # 4. Hits + 379 Target Zeros -> ALKE_hits_new_cutoff_complete.csv
    # Extract only non-NTC targets
    target_gene_df = df[~ntc_mask].copy()
    zero_score_pool = target_gene_df[target_gene_df['Sigmoid_Score'] <= 0.05][cols].copy()

    # Balance 379 Hits against 379 Target-Zeros
    target_count = len(final_hits_df)
    if len(zero_score_pool) >= target_count:
        balanced_zeros = zero_score_pool.sample(n=target_count, random_state=42)
    else:
        balanced_zeros = zero_score_pool

    balanced_df = pd.concat([final_hits_df, balanced_zeros], axis=0)
    balanced_df.to_csv(hits_output, index=False)

    print(f"Processing complete.")
    print(f"File with ALL NTCs saved to: {dist_output}")
    print(f"Balanced (379 Hits + 379 Target Zeros) saved to: {hits_output}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input> <hits_output> <dist_output>")
        sys.exit(1)
    
    run_analysis(sys.argv[1], sys.argv[2], sys.argv[3])

