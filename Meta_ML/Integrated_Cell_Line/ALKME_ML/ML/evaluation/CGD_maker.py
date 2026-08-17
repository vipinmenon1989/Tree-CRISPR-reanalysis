import pandas as pd
import sys

def build_evaluation_file(file1_path, file2_path, output_path):
    print(f"[*] Ingesting File 1: {file1_path}")
    try:
        # file1 headers are comma-separated
        df1 = pd.read_csv(file1_path, sep=',') 
    except FileNotFoundError:
        print(f"CRITICAL ERROR: {file1_path} not found.")
        sys.exit(1)

    print(f"[*] Ingesting File 2: {file2_path}")
    try:
        # file2 headers appear tab-separated based on spacing
        df2 = pd.read_csv(file2_path, sep='\t') 
    except FileNotFoundError:
        print(f"CRITICAL ERROR: {file2_path} not found.")
        sys.exit(1)

    # 1. Isolate only the necessary metrics from the massive file1 matrix
    # This prevents memory bloat during the merge operation.
    required_file1_cols = ['unique_sgrna_id', 'sigmoid_score', 'class']
    
    missing_f1 = [col for col in required_file1_cols if col not in df1.columns]
    if missing_f1:
        print(f"CRITICAL ERROR: Missing expected columns in File 1: {missing_f1}")
        sys.exit(1)
        
    df1_subset = df1[required_file1_cols]

    print("[*] Executing inner join on 'unique_sgrna_id'...")
    # 2. Merge df2 (metadata) with df1_subset (scores/class)
    merged_df = pd.merge(df2, df1_subset, on='unique_sgrna_id', how='inner')

    # 3. Define the strict output schema
    target_columns = [
        'unique_sgrna_id', 
        'id', 
        'sgrna sequence', 
        'gene', 
        'distance_to_tss', 
        'strand', 
        'chromosome', 
        'extended_sequence', 
        'sigmoid_score', 
        'class'
    ]

    # Failsafe: Verify all target columns survived the merge
    missing_final = [col for col in target_columns if col not in merged_df.columns]
    if missing_final:
        print(f"CRITICAL ERROR: Missing columns for final output: {missing_final}")
        sys.exit(1)

    # 4. Slice to the exact requested layout
    final_df = merged_df[target_columns]

    print(f"[*] Exporting target evaluation matrix to {output_path}...")
    final_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"[-->] Operation complete. Shape of evaluation file: {final_df.shape[0]} rows x {final_df.shape[1]} columns.")

if __name__ == "__main__":
    # Specify the exact paths to your data here
    FILE1 = "CRISPRi_ML_Holdout_Test_20_unseen_guides.csv" 
    FILE2 = "Unified_GenomeWide_3Cellline_meta.txt" 
    OUTPUT_FILE = "CGD_evaluation_CRISPRi.txt"
    
    build_evaluation_file(FILE1, FILE2, OUTPUT_FILE)

