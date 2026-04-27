import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='CRISPRi Phase 2: Hit Extraction (RIG Score Only)')
    parser.add_argument('--global_scores', required=True, help='Path to Global_Sigmoid_Scores_RIG.txt (File 1)')
    parser.add_argument('--hits', required=True, help='Path to Phase 1 Validated Hits .txt')
    parser.add_argument('--prefix', default='ALKE', help='Prefix for output filenames')
    args = parser.parse_args()

    # 1. Load Data
    try:
        df_global = pd.read_csv(args.global_scores, sep='\t')
        df_hits = pd.read_csv(args.hits, sep='\t')
        hit_genes = df_hits['Gene'].tolist()
    except Exception as e:
        print(f"CRITICAL ERROR loading files: {e}")
        sys.exit(1)

    # Safety check for the new ML Target Variable
    if 'Score_RIG' not in df_global.columns:
        print("CRITICAL ERROR: 'Score_RIG' column missing. Make sure you are using the new File 1.")
        sys.exit(1)

    # Filter for hits once to use across files
    df_hit_only = df_global[df_global['Gene'].isin(hit_genes)].copy()

    # ---------------------------------------------------------
    # FILE 2: Gene-Level Aggregated (Mean RIG Score)
    # ---------------------------------------------------------
    file2 = df_hit_only.groupby('Gene').agg({
        'Score_RIG': 'mean'
    }).reset_index()
    
    file2_name = f"{args.prefix}_Gene_Mean_RIG_Score.txt"
    file2.to_csv(file2_name, sep='\t', index=False)

    # ---------------------------------------------------------
    # FILE 3: sgRNA Level (Hits + NTCs) - ML Training Set Base
    # ---------------------------------------------------------
    ntc_mask = df_global['Gene'].str.contains('^NO-TARGET', na=False, case=False, regex=True)
    df_ntc = df_global[ntc_mask].copy()
    
    file3 = pd.concat([df_hit_only, df_ntc], axis=0)
    file3 = file3[['Gene', 'sgrna', 'Score_RIG']]
    
    file3_name = f"{args.prefix}_sgRNA_with_NTC_RIG_Score.txt"
    file3.to_csv(file3_name, sep='\t', index=False)

    # ---------------------------------------------------------
    # FILE 4: sgRNA Level (Hits Only) - Pure Signal
    # ---------------------------------------------------------
    file4 = df_hit_only[['Gene', 'sgrna', 'Score_RIG']]
    
    file4_name = f"{args.prefix}_sgRNA_Hits_Only_RIG_Score.txt"
    file4.to_csv(file4_name, sep='\t', index=False)

    # Reporting
    print("-" * 50)
    print(f"PHASE 2 EXTRACTION COMPLETE (RIG SCORES ONLY)")
    print("-" * 50)
    print(f"File 2 (Gene Mean Score): {len(file2):,} entries -> {file2_name}")
    print(f"File 3 (Hits + NTCs):     {len(file3):,} entries -> {file3_name}")
    print(f"File 4 (Hits Only):       {len(file4):,} entries -> {file4_name}")
    print("-" * 50)

if __name__ == "__main__":
    main()
