import pandas as pd
import sys

def merge_and_format_files(file1_path, file2_path, output_path):
    print(f"[*] Ingesting File 1: {file1_path}")
    print(f"[*] Ingesting File 2: {file2_path}")
    
    try:
        # Assuming tab-separated values (\t) based on your prompt's spacing.
        # If your actual files are CSVs, change sep='\t' to sep=','
        df1 = pd.read_csv(file1_path, sep='\t')
        df2 = pd.read_csv(file2_path, sep='\t')
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    # 1. Identify the common keys to merge on
    merge_keys = ['unique_sgrna_id', 'sgrna sequence']
    
    # Verify the keys exist in both dataframes to prevent KeyErrors
    for key in merge_keys:
        if key not in df1.columns or key not in df2.columns:
            print(f"CRITICAL ERROR: Merge key '{key}' not found in both files.")
            sys.exit(1)

    print("[*] Executing inner join on common identifiers...")
    # 2. Merge the dataframes. 
    # An 'inner' merge ensures we only keep guides that exist in BOTH files.
    merged_df = pd.merge(df1, df2, on=merge_keys, how='inner')

    # 3. Define the exact target headers in the requested order
    target_columns = [
        'unique_sgrna_id',
	'id', 
        'sgrna sequence', 
        'class', 
        'gene', 
        'distance_to_tss', 
        'start', 
        'end', 
        'strand', 
        'prediction_probability', 
        'prediction_binary'
    ]

    # Verify all target columns exist in the merged dataframe before slicing
    missing_cols = [col for col in target_columns if col not in merged_df.columns]
    if missing_cols:
        print(f"CRITICAL ERROR: Missing columns after merge: {missing_cols}")
        sys.exit(1)

    # 4. Isolate the required columns
    final_df = merged_df[target_columns]

    # 5. Export the file
    print(f"[*] Exporting consolidated data to {output_path}...")
    final_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"[-->] Operation complete. Shape of final file: {final_df.shape[0]} rows x {final_df.shape[1]} columns.")

if __name__ == "__main__":
    # Replace these variables with your actual file paths
    FILE1 = "prediction.txt"
    FILE2 = "Unified_GenomeWide_3Cellline_meta.txt"
    OUTPUT_FILE = "merged_sgRNA_data.txt" 
    
    merge_and_format_files(FILE1, FILE2, OUTPUT_FILE)
