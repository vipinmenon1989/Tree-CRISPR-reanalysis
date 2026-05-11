import sys
import pandas as pd

def run_analysis(input_file, hits_output, dist_output):
    df = pd.read_csv(input_file)

    # 1. Define qualified sgRNAs (Sigmoid_Score > 0.25)
    qualified_mask = df['Sigmoid_Score'] > 0.25
    qualified_sgrnas = df[qualified_mask].copy()

    # 2. Identify significant genes (>= 2 qualified sgRNAs)
    gene_counts = qualified_sgrnas.groupby('Gene').size()
    significant_genes = gene_counts[gene_counts >= 2].index

    # 3. Create the "Hits" file (Both filters: Score > 0.25 AND Count >= 2)
    final_hits_df = qualified_sgrnas[qualified_sgrnas['Gene'].isin(significant_genes)][['sgrna', 'Gene', 'Sigmoid_Score']]
    final_hits_df.to_csv(hits_output, index=False)

    # 4. Create the "Distribution" file (Final Hits + NTC only)
    # Extract NTCs from the original dataframe
    ntc_mask = df['Gene'].str.contains('NO-TARGET|NTC', case=False, na=False)
    ntc_df = df[ntc_mask][['sgrna', 'Gene', 'Sigmoid_Score']]

    # Concatenate Final Hits and NTCs
    # This removes sgRNAs that passed the score > 0.25 but failed the count >= 2 filter
    dist_df = pd.concat([final_hits_df, ntc_df], axis=0)
    
    dist_df.to_csv(dist_output, index=False)

    print(f"Processing complete.")
    print(f"High-stringency hits saved to: {hits_output}")
    print(f"Distribution file (Hits + NTC) saved to: {dist_output}")
    print(f"Total rows in distribution: {len(dist_df)}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python filter_script.py <input_csv_file> <hits_output_csv> <dist_output_csv>")
        sys.exit(1)
    
    run_analysis(sys.argv[1], sys.argv[2], sys.argv[3])


