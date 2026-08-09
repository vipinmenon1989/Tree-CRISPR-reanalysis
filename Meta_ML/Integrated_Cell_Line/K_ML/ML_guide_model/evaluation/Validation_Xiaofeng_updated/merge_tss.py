import pandas as pd

def merge_genomic_features(file1_path, file2_path, output_path):
    # Load File 1 as comma-separated
    df1 = pd.read_csv(file1_path, sep=",")
    # Load File 2 as tab-separated
    df2 = pd.read_csv(file2_path, sep="\t")
    
    # Strip invisible whitespace from all column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    # Define a highly rigid 7-key lock for the merge
    merge_keys = [
        'id', 'gene', 'sgrna sequence', 'extended_sequence', 
        'Start', 'End', 'Strand'
    ]
    
    # Define exact columns to retain from File 1
    cols_file1 = merge_keys + [
        'unique_sgrna_id', 'sigmoid_score', 'class', 
        'probability_score', 'prediction_binary', 'distance_to_tss'
    ]
    
    # Define exact columns to retain from File 2 
    # (Extracting only the new features to avoid duplication)
    cols_file2 = merge_keys + [
        'PAM', 'closest_TSS_coord', 'Chromosome'
    ]
    
    # Verify the required columns exist in File 1
    missing_in_df1 = [col for col in cols_file1 if col not in df1.columns]
    if missing_in_df1:
        raise ValueError(f"Required columns missing in File 1: {missing_in_df1}")
        
    # Verify the required columns exist in File 2
    missing_in_df2 = [col for col in cols_file2 if col not in df2.columns]
    if missing_in_df2:
        raise ValueError(f"Required columns missing in File 2: {missing_in_df2}")

    # Isolate dataframes before merging to eliminate unused columns (like start_30, end_30)
    df1_subset = df1[cols_file1]
    df2_subset = df2[cols_file2]
    
    # Perform the inner merge
    merged_df = pd.merge(df1_subset, df2_subset, on=merge_keys, how='inner')
    
    # Enforce the exact final column order requested
    final_column_order = [
        'unique_sgrna_id', 
        'id', 
        'gene', 
        'sgrna sequence', 
        'extended_sequence', 
        'Start', 
        'End', 
        'Strand', 
        'sigmoid_score', 
        'class', 
        'probability_score', 
        'prediction_binary', 
        'distance_to_tss',
        'PAM',
        'closest_TSS_coord',
        'Chromosome'
    ]
    
    # Apply the strict column order
    merged_df = merged_df[final_column_order]
    
    # Save the output dataset
    # Based on your prompt layout, writing as a comma-separated format
    merged_df.to_csv(output_path, sep=",", index=False)
    
    # Logic check
    print(f"Rows in File 1 (.csv): {len(df1)}")
    print(f"Rows in File 2 (.txt): {len(df2)}")
    print(f"Rows in Merged Output: {len(merged_df)}")
    print(f"Process complete. Final formatted file saved to: {output_path}")

merge_genomic_features('final_output.csv', 'TreeCRISPR_lib_TSS.txt', 'Validation_Xiaofeng.txt')
