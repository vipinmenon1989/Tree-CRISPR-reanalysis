import pandas as pd
import os
import sys

def generate_metadata(k562_path, a549_path, a375_path, output_path):
    print("======================================================================")
    print("[*] METADATA EXTRACTION UTILITY: Unified_GenomeWide_3Cellline_meta")
    print("======================================================================\n")

    files = {'K562': k562_path, 'A549': a549_path, 'A375': a375_path}
    
    # Use lowercase strings to match the .lower() conversion performed below
    meta_cols = ['id', 'sgrna sequence', 'gene', 'distance_to_tss', 'start', 'end']
    
    combined_list = []
    
    for cell_name, path in files.items():
        if not os.path.exists(path):
            print(f"CRITICAL ERROR: {path} not found.")
            sys.exit(1)
            
        print(f"[*] Extracting metadata from {cell_name}...")
        sep = ',' if path.endswith('.csv') else '\t'
        df = pd.read_csv(path, sep=sep)
        
        # Convert all headers to lowercase to match meta_cols
        df.columns = [col.lower().strip() for col in df.columns]
        
        # Select existing metadata columns
        available = [c for c in meta_cols if c in df.columns]
        temp_df = df[available].copy()
        
        # Add cell_line_origin as the final column
        temp_df['cell_line_origin'] = cell_name
        combined_list.append(temp_df)
    
    # Concatenate all
    final_meta = pd.concat(combined_list, axis=0, ignore_index=True)
    
    # Insert unique_sgrna_id at index 0
    final_meta.insert(0, 'unique_sgrna_id', [f"sgrna_{i}" for i in range(1, len(final_meta) + 1)])
    
    # Save to file
    final_meta.to_csv(output_path, sep='\t', index=False)
    print(f"\n[SUCCESS] Metadata saved to: {output_path}")
    print(f"[METRICS] Total entries: {len(final_meta)}")

if __name__ == "__main__":
    generate_metadata(
        k562_path='CRISPR_ml_features_final_K562.csv',
        a549_path='CRISPR_ml_features_final_A549.csv',
        a375_path='CRISPR_ml_features_final_A375.csv',
        output_path='Unified_GenomeWide_3Cellline_meta.txt'
    )
