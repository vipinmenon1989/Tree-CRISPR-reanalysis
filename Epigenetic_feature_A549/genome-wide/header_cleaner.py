import os
import re
import sys
import pandas as pd

def standardize_epigenetic_headers(file_path, output_path, cell_line_prefix):
    """
    Strips cell-line specific prefixes from genome-wide matrix headers
    to enable clean concatenation across K562, A549, and A375 layers.
    """
    print("======================================================================")
    print(f"[*] STARTING HEADER STANDARDIZATION FOR PREFIX: {cell_line_prefix}")
    print("======================================================================")
    
    if not os.path.exists(file_path):
        print(f"CRITICAL ERROR: Input file not found at: {file_path}")
        sys.exit(1)
        
    # 1. Dynamic Delimiter Detection
    file_ext = os.path.splitext(file_path)[1]
    separator = ',' if file_ext == '.csv' else '\t'
    
    # Backward compatibility fix: Evaluated outside the f-string block
    delim_name = 'TAB' if separator == '\t' else 'COMMA'
    print(f"[*] Reading matrix (detected delimiter: {delim_name})...")
    
    try:
        df = pd.read_csv(file_path, sep=separator)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to parse dataframe: {e}")
        sys.exit(1)

    old_columns = df.columns.tolist()
    new_columns = []
    
    # 2. Compile Regex Pattern
    pattern = re.compile(rf"{cell_line_prefix}_")
    
    modified_tracks = []
    
    for col in old_columns:
        clean_col = pattern.sub("", col)
        new_columns.append(clean_col)
        if clean_col != col:
            modified_tracks.append((col, clean_col))
            
    df.columns = new_columns
    
    # 3. Dynamic Sanity Check Printout
    print("\n" + "-"*70)
    print(f"HEADER MAPPING REPORT (Total columns modified: {len(modified_tracks)})")
    print("-"*70)
    
    if len(modified_tracks) == 0:
        print(f"WARNING: No headers matched the prefix '{cell_line_prefix}_'. Zero changes made.")
    else:
        # Display a sample of modified columns dynamically (First 3 and Last 3)
        sample_size = 3
        display_samples = modified_tracks[:sample_size]
        if len(modified_tracks) > sample_size * 2:
            display_samples += [("...", "...")] + modified_tracks[-sample_size:]
        elif len(modified_tracks) > sample_size:
            display_samples += modified_tracks[sample_size:]
            
        for old, new in display_samples:
            if old == "...":
                print("   ... [skipping intermediate tracks] ...")
            else:
                print(f"  -> [OLD]: {old}")
                print(f"     [NEW]: {new}\n")
                
    print("-"*70)

    # 4. Save Standardized Matrix Back to Disk
    try:
        df.to_csv(output_path, sep=separator, index=False)
        print(f"--> SUCCESS: Output saved to: {output_path}")
        print(f"--> Final Matrix Structure: {df.shape[0]} rows x {df.shape[1]} columns")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed writing file to disk: {e}")
        sys.exit(1)
        
    print("======================================================================\n")

if __name__ == "__main__":
    # Target Configuration for your A375 file
    input_matrix='ALKE_hits_complete_A549_training_epigenetics_scaled.txt'
    standardized_matrix='ALKE_epigenetic_training_clean_A549.txt'
    
    standardize_epigenetic_headers(
        file_path=input_matrix, 
        output_path=standardized_matrix, 
        cell_line_prefix="A549"
    )
