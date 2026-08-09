import pandas as pd

def merge_datasets(file1_path, file2_path, output_path):
    # Load the datasets
    df1 = pd.read_csv(file1_path, sep="\t")
    df2 = pd.read_csv(file2_path, sep="\t")
    
    # Strip invisible whitespace from all column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    # Define the exact columns that are common between both files
    merge_keys = ['id', 'sgrna sequence', 'gene']
    
    # Verify the required columns exist in File 1 
    # (Removed 'gene_status' from this check since we are no longer filtering on it)
    missing_in_df1 = [col for col in merge_keys if col not in df1.columns]
    if missing_in_df1:
        print("Columns detected in File 1:", df1.columns.tolist())
        raise ValueError(f"Required columns missing in File 1: {missing_in_df1}")
        
    # Verify the required columns exist in File 2 
    missing_in_df2 = [col for col in merge_keys + ['unique_sgrna_id'] if col not in df2.columns]
    if missing_in_df2:
        print("Columns detected in File 2:", df2.columns.tolist())
        raise ValueError(f"Required columns missing in File 2: {missing_in_df2}")

    # Extract only the merge keys and the target column from File 2
    df2_subset = df2[merge_keys + ['unique_sgrna_id']]
    
    # Inner merge using the list of common columns
    merged_df = pd.merge(df1, df2_subset, on=merge_keys, how='inner')
    
    # Save the updated dataset
    merged_df.to_csv(output_path, sep="\t", index=False)
    
    # Logic check
    print(f"Rows in File 1: {len(df1)}")
    print(f"Rows in File 2: {len(df2)}")
    print(f"Rows in Merged Output: {len(merged_df)}")
    print(f"Process complete. File saved to: {output_path}")

# Example execution:
merge_datasets('Final_Validated_Dataset.txt', 'Unified_target_genes_metadata.txt', 'merged_output.txt')
