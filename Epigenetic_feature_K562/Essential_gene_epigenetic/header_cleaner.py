import pandas as pd
import re
import sys

def standardize_epigenetic_headers(file_path, output_path, cell_line_prefix="K562"):
    print(f"[*] Loading training dataset from: {file_path}")
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to read file: {e}")
        sys.exit(1)
        
    old_columns = df.columns.tolist()
    new_columns = []
    
    # Regex pattern: match the prefix (e.g., K562) followed by an underscore
    # This handles both 'K562_...' -> '...' and 'guide_K562_...' -> 'guide_...'
    pattern = rf"{cell_line_prefix}_"
    
    for col in old_columns:
        # Programmatically replace the cell line name with an empty string
        clean_col = re.sub(pattern, "", col)
        new_columns.append(clean_col)
        
    df.columns = new_columns
    
    # Sanity verification printout
    print("\n" + "-"*50)
    print("HEADER MAPPING SANITY CHECK")
    print("-"*50)
    sample_indices = [14, 15, 44, 45, 74, 75] # Inspecting a mix of promoter and guide bin columns
    for i in sample_indices:
        if i < len(old_columns):
            print(f"  -> Old: {old_columns[i]}")
            print(f"     New: {df.columns[i]}\n")
    print("-"*50)

    # Save standardized matrix back to disk
    df.to_csv(output_path, sep='\t', index=False)
    print(f"--> SUCCESS: Standardized matrix saved. New shape: {df.shape}")

if __name__ == "__main__":
    # Point directly to your freshly merged table
    input_matrix = 'ALKE_K562_epigenetic_training.txt'
    standardized_matrix = 'ALKE_epigenetic_training_clean.txt'
    
    standardize_epigenetic_headers(input_matrix, standardized_matrix, cell_line_prefix="K562")
