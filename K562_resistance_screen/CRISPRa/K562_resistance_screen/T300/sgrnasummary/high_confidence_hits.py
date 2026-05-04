import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='CRISPRi Phase 2: Hit Extraction (Delta Score Only)')
    parser.add_argument('--global_scores', required=True, help='Path to Global_Sigmoid_Scores.txt (File 1)')
    parser.add_argument('--hits', required=True, help='Path to Phase 1 Validated Hits .txt')
    parser.add_argument('--prefix', default='ALKH', help='Prefix for output filenames')
    args = parser.parse_args()

    # 1. Load Data
    try:
        df_global = pd.read_csv(args.global_scores, sep='\t')
        df_hits = pd.read_csv(args.hits, sep='\t')
        hit_genes = df_hits['Gene'].tolist()
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    # Filter for hits once to use across files
    df_hit_only = df_global[df_global['Gene'].isin(hit_genes)].copy()

    # ---------------------------------------------------------
    # FILE 2: Gene-Level Aggregated (Mean Delta Score)
    # ---------------------------------------------------------
    file2 = df_hit_only.groupby('Gene').agg({
        'Delta_Score': 'mean'
    }).reset_index()
    
    file2_name = f"{args.prefix}_Gene_Mean_Delta.txt"
    file2.to_csv(file2_name, sep='\t', index=False)

    # ---------------------------------------------------------
    # FILE 3: sgRNA Level (Hits + NTCs) - Delta Score Only
    # ---------------------------------------------------------
    ntc_mask = df_global['Gene'].str.contains('^NO-TARGET', na=False, case=False, regex=True)
    df_ntc = df_global[ntc_mask].copy()
    
    file3 = pd.concat([df_hit_only, df_ntc], axis=0)
    file3 = file3[['Gene', 'sgrna', 'Delta_Score']]
    
    file3_name = f"{args.prefix}_sgRNA_with_NTC_Delta.txt"
    file3.to_csv(file3_name, sep='\t', index=False)

    # ---------------------------------------------------------
    # FILE 4: sgRNA Level (Hits Only) - Delta Score Only
    # ---------------------------------------------------------
    file4 = df_hit_only[['Gene', 'sgrna', 'Delta_Score']]
    
    file4_name = f"{args.prefix}_sgRNA_Hits_Only_Delta.txt"
    file4.to_csv(file4_name, sep='\t', index=False)

    # Reporting
    print("-" * 45)
    print(f"PHASE 2 POST-PROCESSING COMPLETE")
    print(f"File 2 (Gene Mean Delta): {len(file2)} entries")
    print(f"File 3 (sgRNA Hits+NTC Delta): {len(file3)} entries")
    print(f"File 4 (sgRNA Hits Only Delta): {len(file4)} entries")
    print("-" * 45)

if __name__ == "__main__":
    main()

