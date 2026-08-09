import pandas as pd

def append_sigmoid_score(merged_file_path, feature_file_path, final_output_path):
    # Load the datasets
    df_merged = pd.read_csv(merged_file_path, sep="\t")
    df_features = pd.read_csv(feature_file_path, sep="\t")
    
    # Strip invisible whitespace from all column names to guarantee exact matches
    df_merged.columns = df_merged.columns.str.strip()
    df_features.columns = df_features.columns.str.strip()
    
    # Define the 3 common keys to ensure precise row alignment
    merge_keys = ['unique_sgrna_id', 'sgrna sequence', 'gene']
    
    # Verify the required columns exist in the first file (merged_output.txt)
    missing_in_df1 = [col for col in merge_keys if col not in df_merged.columns]
    if missing_in_df1:
        print("Columns detected in File 1:", df_merged.columns.tolist())
        raise ValueError(f"Required keys missing in merged output file: {missing_in_df1}")
        
    # Verify the required columns exist in the new feature file
    missing_in_df2 = [col for col in merge_keys + ['sigmoid_score'] if col not in df_features.columns]
    if missing_in_df2:
        print("Columns detected in File 2:", df_features.columns.tolist())
        raise ValueError(f"Required columns missing in feature file: {missing_in_df2}")

    # Extract ONLY the 3 merge keys and the target 'sigmoid_score' from the massive feature file.
    # This prevents the hundreds of atac/methylation/pos columns from bloating your output.
    df_features_subset = df_features[merge_keys + ['sigmoid_score']]
    
    # Merge the dataframes. 
    # Using how='inner' keeps only sequences that exist in BOTH files.
    # Note: If you want to keep all rows from df_merged even if they lack a score in the new file,
    # change how='inner' to how='left'.
    final_df = pd.merge(df_merged, df_features_subset, on=merge_keys, how='inner')
    
    # Save the updated dataset
    final_df.to_csv(final_output_path, sep="\t", index=False)
    
    # Logic check to verify row retention
    print(f"Rows in Merged Input File: {len(df_merged)}")
    print(f"Rows in New Feature File: {len(df_features)}")
    print(f"Rows in Final Output (with sigmoid_score): {len(final_df)}")
    print(f"Process complete. Final file saved to: {final_output_path}")

# Example execution:
append_sigmoid_score('merged_output.txt', 'CRISPRi_ML_Holdout_Test_20_unseen_guides.txt', 'Xiaofeng_Validation_with_sigmoid_scores.txt')
