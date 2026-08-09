import pandas as pd

def append_sigmoid_final(file1_path, file2_path, output_path):
    # Load the datasets
    df1 = pd.read_csv(file1_path, sep="\t")
    df2 = pd.read_csv(file2_path, sep="\t")
    
    # Strip invisible whitespace from all column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    # Define the 4 common keys for a high-precision merge
    merge_keys = ['unique_sgrna_id', 'id', 'sgrna sequence', 'gene']
    
    # Verify the required columns exist in File 1
    missing_in_df1 = [col for col in merge_keys if col not in df1.columns]
    if missing_in_df1:
        print("Columns detected in File 1:", df1.columns.tolist())
        raise ValueError(f"Required merge keys missing in File 1: {missing_in_df1}")
        
    # Verify the required columns exist in File 2
    missing_in_df2 = [col for col in merge_keys + ['sigmoid_score'] if col not in df2.columns]
    if missing_in_df2:
        print("Columns detected in File 2:", df2.columns.tolist())
        raise ValueError(f"Required columns missing in File 2: {missing_in_df2}")

    # Isolate only the merge keys and the target metric from File 2.
    # This explicitly drops File 2's version of 'class', preventing '_x'/'_y' suffix corruption.
    df2_subset = df2[merge_keys + ['sigmoid_score']]
    
    # Perform the inner merge
    merged_df = pd.merge(df1, df2_subset, on=merge_keys, how='inner')
    
    # Define the final requested column order exactly as specified
    final_column_order = [
        'unique_sgrna_id', 
        'id', 
        'sgrna sequence', 
        'gene', 
        'distance_to_tss', 
        'class', 
        'prediction_probability', 
        'prediction_binary', 
        'Extendede_sequence',
        'sigmoid_score'
    ]
    
    # Verify all final columns exist before reordering to prevent KeyErrors
    missing_final = [col for col in final_column_order if col not in merged_df.columns]
    if missing_final:
        raise ValueError(f"Merge succeeded, but these expected final columns are missing: {missing_final}")
    
    # Apply the strict column order
    merged_df = merged_df[final_column_order]
    
    # Save the strictly formatted dataset
    merged_df.to_csv(output_path, sep="\t", index=False)
    
    # Logic check
    print(f"Rows in File 1 (Base): {len(df1)}")
    print(f"Rows in File 2 (Features): {len(df2)}")
    print(f"Rows in Merged Output: {len(merged_df)}")
    print(f"Process complete. Final formatted file saved to: {output_path}")

# Example execution:
append_sigmoid_final('final_formatted_output.txt', 'CRISPRi_ML_Holdout_Test_20_unseen_guides.txt', 'Pre-Validation.txt')
