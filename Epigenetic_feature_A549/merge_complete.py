import pandas as pd

def merge_crispri_minimal_keys(file1_path, file2_path, output_path):
    # Load datasets
    df_epi = pd.read_csv(file1_path, sep='\t')
    df_guide = pd.read_csv(file2_path, sep=',')

    # Merge strictly on 'ID' and 'Gene'. 
    # Using how='outer' ensures all row elements from both files are retained.
    # Suffixes are added to differentiate overlapping columns not in the merge key.
    merged_df = pd.merge(df_epi, df_guide, on=['ID', 'Gene'], how='outer', suffixes=('_epi', '_guide'))

    # Export
    merged_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Merge complete. Output shape: {merged_df.shape}")

# Execution variables
file_2 = 'ALKE_hits_complete.csv'
file_1 = 'TreeCRISPRi_lib_merged_K562_epigenetic_feature.txt'
output_file = 'ALKE_hits_complete_K562_Epigenetic.csv'

if __name__ == "__main__":
    merge_crispri_minimal_keys(file_1, file_2, output_file)
