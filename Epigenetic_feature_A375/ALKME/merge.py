import pandas as pd

def merge_crispri_files(file1_path, file2_path, output_path):
    # Load the datasets. Assuming tab-separated based on standard genomic data structures.
    df_epi = pd.read_csv(file1_path, sep='\t')
    df_guide = pd.read_csv(file2_path, sep='\t')

    # Define the common columns to merge on. 
    # This prevents pandas from duplicating identical metadata columns.
    common_columns = [
        'ID', 'sgRNA Sequence', 'Gene', 'Chromosome', 'Start', 'End', 
        'Strand', 'start_30', 'end_30', 'extended_sequence', 'PAM', 
        'distance_to_TSS', 'closest_TSS_coord', 'Gene_Strand'
    ]

    # Perform an inner join to ensure only rows with matching IDs in both files are kept.
    merged_df = pd.merge(df_epi, df_guide, on=common_columns, how='inner')

    # Export the merged dataframe to a new text file
    merged_df.to_csv(output_path, sep='\t', index=False)
    
    print(f"Files successfully merged. Initial row counts: Epi({len(df_epi)}), Guide({len(df_guide)}).")
    print(f"Merged output saved to {output_path} with {len(merged_df)} rows and {len(merged_df.columns)} columns.")

# Execution
file_1 = 'TreeCRISPRi_lib_TSS_revised_bins_gene_A375.txt'
file_2 = 'TreeCRISPRi_lib_TSS_revised_bins_guide_A375.txt'
output_file = 'TreeCRISPRi_lib_merged_A375.txt'

merge_crispri_files(file_1, file_2, output_file)
