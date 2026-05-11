import pandas as pd
import argparse
import sys

def filter_and_extract_sgrnas(input_file, hits_output, dist_output):
    # 1. Load sgRNA-level data from File 1
    try:
        df = pd.read_csv(input_file, sep='\t')
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        sys.exit(1)

    # 2. Identify "Qualified" sgRNAs
    # Logic: Sigmoid > 0.25 AND Delta > 0.1 
    qualified_mask = (df['Score_RIG'] > 0.25) & (df['Delta_Score'] > 0.1)
    df_qualified = df[qualified_mask].copy()

    # 3. Apply the Replication Filter (at least 2 qualified sgRNAs per gene)
    # The filter is applied at the gene level, but we keep the sgRNAs 
    gene_counts = df_qualified.groupby('Gene').size()
    significant_genes = gene_counts[gene_counts >= 2].index
    
    # 4. Extract Final Hits (All sgRNAs belonging to Significant Genes)
    # This keeps the 'sgrna' column and individual scores intact
    final_hits_sgrna = df_qualified[df_qualified['Gene'].isin(significant_genes)].copy()

    # --- OUTPUT 1: Significant_Resistance_Hits_sgRNA.csv ---
    # This contains all individual guides for genes that passed the n>=2 rule
    final_hits_sgrna.to_csv(hits_output, index=False)

    # --- OUTPUT 2: Hits_plus_NTC_Distribution.csv ---
    # Identify NTCs using standard regex pattern 
    ntc_mask = df['Gene'].str.contains('^NO-TARGET|NTC', na=False, case=False, regex=True)
    df_ntc = df[ntc_mask].copy()
    
    # Concatenate individual Hit sgRNAs + individual NTC sgRNAs
    distribution_df = pd.concat([final_hits_sgrna, df_ntc], axis=0)
    distribution_df.to_csv(dist_output, index=False)

    print("-" * 40)
    print(f"Decision Layer Applied (sgRNA-level output)")
    print(f"Total Significant Genes (n>=2): {len(significant_genes)}")
    print(f"Significant sgRNAs saved: {len(final_hits_sgrna)} -> {hits_output}")
    print(f"Density Plot data saved: {len(distribution_df)} rows -> {dist_output}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRISPRi Phase 2: sgRNA-Level Hit Extraction')
    parser.add_argument('-i', '--input', required=True, help='Output from File 1 (sgRNA level)')
    parser.add_argument('-o_hits', '--out_hits', default='Significant_Resistance_Hits_sgRNA.csv', help='Individual hits')
    parser.add_argument('-o_dist', '--out_dist', default='Hits_plus_NTC_Distribution.csv', help='Density Plot input')
    
    args = parser.parse_args()
    filter_and_extract_sgrnas(args.input, args.out_hits, args.out_dist)


