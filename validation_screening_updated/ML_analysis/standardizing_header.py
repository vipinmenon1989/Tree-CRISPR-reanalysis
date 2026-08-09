import pandas as pd
import os
import sys

def clean_column_prefixes(input_file, output_file):
    """
    Cleans cell-line specific prefixes/suffixes from column names in a single file,
    specifically targeting 'guide_*' tracks and standardizing formatting.
    """
    print(f"[*] Loading dataset: {input_file}")
    
    # Determine separator dynamically
    sep = ',' if input_file.endswith('.csv') else '\t'
    df = pd.read_csv(input_file, sep=sep)
    
    # Lowercase all columns for robust string matching
    df.columns = [col.lower() for col in df.columns]
    
    rename_dict = {}
    
    # Expand this list if you have other prefixes to strip in the future
    prefixes_to_strip = ['k562_', 'a549_', 'a375_']
    
    for col in df.columns:
        new_col = col
        
        # 1. Clean 'guide_{cell}_' patterns (e.g., guide_k562_atac -> guide_atac)
        if col.startswith('guide_'):
            for prefix in prefixes_to_strip:
                if col.startswith(f'guide_{prefix}'):
                    new_col = col.replace(f'guide_{prefix}', 'guide_')
                    break
                    
        # 2. Clean standard '{cell}_' prefixes (e.g., k562_feature -> feature)
        else:
            for prefix in prefixes_to_strip:
                if col.startswith(prefix):
                    new_col = col.replace(prefix, '')
                    break
        
        # 3. Handle legacy naming anomalies
        if 'dna_methylation_bin_' in new_col:
            new_col = new_col.replace('dna_methylation_bin_', 'dna_methylation_coverage_bin_')
            
        # Log if a transformation occurred
        if new_col != col:
            rename_dict[col] = new_col
            
    # Apply the renaming schema to the dataframe
    if rename_dict:
        df = df.rename(columns=rename_dict)
        print(f"[-->] Standardized {len(rename_dict)} column tracks.")
        
        # Print a preview of the first 10 changes for verification
        print("    [Preview of changes]:")
        for old, new in list(rename_dict.items())[:10]:
            print(f"      - {old}  ->  {new}")
        if len(rename_dict) > 10:
            print(f"      - ... and {len(rename_dict) - 10} more.")
    else:
        print("[!] No columns matched the cleaning criteria. Names remain unchanged.")
        
    # Export cleaned matrix
    df.to_csv(output_file, sep=sep, index=False)
    print(f"\n[*] Process complete. Cleaned matrix saved to: {output_file}")

if __name__ == "__main__":
    # Define target files here
    INPUT_PATH = "CRISPR_merged_final_ML_final.csv"
    OUTPUT_PATH = "CRISPR_merged_final_ML_final_cleaned.csv"
    
    if not os.path.exists(INPUT_PATH):
        print(f"CRITICAL ERROR: Input file '{INPUT_PATH}' not found in the current directory.")
        sys.exit(1)
        
    clean_column_prefixes(INPUT_PATH, OUTPUT_PATH)
