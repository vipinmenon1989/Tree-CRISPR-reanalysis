import pandas as pd

def merge_crispri_on_id(file_a_path, file_b_path, output_path):
    # Load datasets
    df_a = pd.read_csv(file_a_path, sep='\t')
    df_b = pd.read_csv(file_b_path, sep=',')

    # Strip hidden whitespace from headers
    df_a.columns = df_a.columns.str.strip()
    df_b.columns = df_b.columns.str.strip()

    # Programmatically enforce 'ID' as the header for the first column in both files
    df_a.rename(columns={df_a.columns[0]: 'ID'}, inplace=True)
    df_b.rename(columns={df_b.columns[0]: 'ID'}, inplace=True)

    # Define merge keys. If 'Gene' is present in both, include it to prevent duplicate columns.
    merge_keys = ['ID']
    if 'Gene' in df_a.columns and 'Gene' in df_b.columns:
        merge_keys.append('Gene')

    # Merge strictly on the explicit ID 
    merged_df = pd.merge(df_a, df_b, on=merge_keys, how='inner', suffixes=('_epi', '_hits'))

    # Export to tab-separated file
    merged_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Merge complete. Output shape: {merged_df.shape}")

# Execution variables
file_1 = 'TreeCRISPRi_lib_merged_K562_epigenetic_feature.txt'
file_2 = 'ALKE_hits_complete.csv'
output_file = 'ALKE_hits_complete_K562_Epigenetic_ID_merged.txt'

if __name__ == "__main__":
    merge_crispri_on_id(file_1, file_2, output_file)
