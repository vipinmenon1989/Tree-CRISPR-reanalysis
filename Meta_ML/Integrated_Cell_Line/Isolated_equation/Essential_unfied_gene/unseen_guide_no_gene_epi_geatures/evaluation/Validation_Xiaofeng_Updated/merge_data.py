import pandas as pd

def custom_merge_and_format(file1_path, file2_path, output_path):
    # Load the datasets
    df1 = pd.read_csv(file1_path, sep="\t")
    df2 = pd.read_csv(file2_path, sep="\t")
    
    # Strip invisible whitespace from all column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    # Define the overlapping keys to merge on
    merge_keys = ['unique_sgrna_id', 'sgrna sequence']
    
    # Define the exact columns to extract from File 1 (dropping CGDi_Score, SSC)
    cols_file1 = merge_keys + [
        'class', 'prediction_probability', 'prediction_binary', 'Extendede_sequence'
    ]
    
    # Define the exact columns to extract from File 2 (dropping cell_line_origin)
    cols_file2 = merge_keys + [
        'id', 'gene', 'distance_to_tss'
    ]
    
    # Subset the dataframes to isolate only required columns before the join
    df1_subset = df1[cols_file1]
    df2_subset = df2[cols_file2]
    
    # Perform the inner merge
    merged_df = pd.merge(df1_subset, df2_subset, on=merge_keys, how='inner')
    
    # Define the final requested column order
    final_column_order = [
        'unique_sgrna_id', 
        'id', 
        'sgrna sequence', 
        'gene', 
        'distance_to_tss', 
        'class', 
        'prediction_probability', 
        'prediction_binary', 
        'Extendede_sequence'
    ]
    
    # Apply the strict column order to the merged dataframe
    merged_df = merged_df[final_column_order]
    
    # Save the updated dataset
    merged_df.to_csv(output_path, sep="\t", index=False)
    
    # Logic check
    print(f"Rows in File 1: {len(df1)}")
    print(f"Rows in File 2: {len(df2)}")
    print(f"Rows in Merged Output: {len(merged_df)}")
    print(f"Process complete. File saved to: {output_path}")

# Example execution:
custom_merge_and_format('independent_predictions.txt', 'Unified_target_genes_metadata.txt', 'final_formatted_output.txt')
