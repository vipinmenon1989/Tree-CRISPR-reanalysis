import pandas as pd
import glob
import os

def batch_process_crispri_files(hits_directory, reference_file, output_suffix="_final_annotated.txt"):
    # 1. The strict columns to extract from the massive epigenetic reference file
    epi_columns = [
        'ID', 'sgRNA Sequence', 'Gene', 'Chromosome', 'Start', 
        'End', 'Strand', 'PAM', 'distance_to_TSS', 'closest_TSS_coord'
    ]
    
    # 2. The exact final column layout you requested (combining both files)
    final_column_order = [
        'ID', 'sgRNA Sequence', 'Gene', 'Chromosome', 'Start', 
        'End', 'Strand', 'PAM', 'distance_to_TSS', 'closest_TSS_coord',
        'Sigmoid_Score', 'Category'
    ]
    
    # 3. Load the reference database ONCE into memory
    print(f"Loading reference database: {reference_file}...")
    df_epi = pd.read_csv(reference_file, sep='\t', usecols=epi_columns)
    
    # 4. Find all screening CSVs
    csv_files = glob.glob(os.path.join(hits_directory, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    # 5. Iterate and merge
    for hits_file in csv_files:
        print(f"Processing: {os.path.basename(hits_file)}")
        
        df_hits = pd.read_csv(hits_file, sep=',')
        
        # Standardize the merge key
        if 'sgrna' in df_hits.columns:
            df_hits = df_hits.rename(columns={'sgrna': 'ID'})
        
        # Strict inner join
        merged_df = pd.merge(df_hits, df_epi, on=['ID', 'Gene'], how='inner')
        
        # Verify the requested columns exist in this specific CSV to prevent crashes,
        # then apply the strict ordering.
        actual_columns_to_keep = [col for col in final_column_order if col in merged_df.columns]
        merged_df = merged_df[actual_columns_to_keep]
        
        # Export
        base_name = os.path.splitext(os.path.basename(hits_file))[0]
        output_path = os.path.join(hits_directory, f"{base_name}{output_suffix}")
        
        merged_df.to_csv(output_path, sep='\t', index=False)
        print(f"  -> Saved {len(merged_df)} validated rows to: {os.path.basename(output_path)}\n")

# --- Execution ---
target_dir = '.' 
reference_txt = 'TreeCRISPRi_lib_merged_K562_epigenetic_feature.txt'

batch_process_crispri_files(target_dir, reference_txt)
