import pandas as pd
import sys

def generate_final_validation_set(subset_file, master_file, output_file):
    print(f"[*] Loading Subset Data: {subset_file}")
    print(f"[*] Loading Master Data: {master_file}")
    
    try:
        df_subset = pd.read_csv(subset_file, sep='\t')
        df_master = pd.read_csv(master_file, sep='\t')
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    # 1. Clean headers to prevent matching errors
    df_subset.columns = df_subset.columns.str.strip()
    df_master.columns = df_master.columns.str.strip()
    
    # Handle terminal truncation: In your paste, the last column showed "gene_status>".
    # This ensures pandas recognizes it correctly even if the raw text file header got truncated.
    df_subset.rename(columns=lambda x: 'gene_status' if 'gene_status' in x else x, inplace=True)
    df_subset.rename(columns=lambda x: 'total_guides_for_gene' if 'total_guides' in x else x, inplace=True)

    # 2. Extract ONLY the necessary coordinate columns from the master file
    # This prevents duplicate columns like 'class_x' and 'class_y'
    coord_cols = ['id', 'sgrna sequence', 'gene', 'chromosome', 'start', 'end', 'distance_to_tss', 'strand']
    
    # Verify master has these columns
    missing = [col for col in coord_cols if col not in df_master.columns]
    if missing:
        print(f"CRITICAL ERROR: Master file is missing columns: {missing}")
        sys.exit(1)
        
    df_coords = df_master[coord_cols]

    print("[*] Merging genomic coordinates...")
    # 3. Execute a Left Join
    # 'left' ensures we keep EVERY row from your subset file, attaching coordinates where they match
    final_df = pd.merge(df_subset, df_coords, on=['id', 'sgrna sequence', 'gene'], how='left')

    # 4. Reorder columns for a clean, logical final layout
    target_columns = [
        'sgrna sequence', 'gene', 'id', 'id_ml', 
        'chromosome', 'start', 'end', 'strand', 'distance_to_tss',
        'class', 'prediction_probability', 'prediction_binary', 
        'gene_status', 'total_guides_for_gene'
    ]
    
    # Keep only existing columns to prevent errors if your subset structure slightly varies
    final_columns = [col for col in target_columns if col in final_df.columns]
    final_df = final_df[final_columns]

    # 5. Export
    print(f"[*] Exporting to {output_file}...")
    final_df.to_csv(output_file, sep='\t', index=False)
    print(f"[-->] Operation complete. Shape: {final_df.shape[0]} rows x {final_df.shape[1]} columns.")

if __name__ == "__main__":
    # Ensure these point to the correct files in your directory
    SUBSET_FILE = "Comprehensive_Validation_Set.txt"             # The file with 'Mixed_Performance' etc.
    MASTER_FILE = "independent_prediction.txt"  # The file with all the coordinates
    OUTPUT_FILE = "final_validation_set.txt"
    
    generate_final_validation_set(SUBSET_FILE, MASTER_FILE, OUTPUT_FILE)
