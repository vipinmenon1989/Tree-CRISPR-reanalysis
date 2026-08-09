import pandas as pd

def merge_csv_and_txt(file1_path, file2_path, output_path):
    # Load the datasets with their respective delimiters
    df1 = pd.read_csv(file1_path, sep="\t")
    df2 = pd.read_csv(file2_path, sep=",")
    
    # Strip invisible whitespace from all column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    # Define all common keys to lock the merge perfectly and prevent duplicate columns
    merge_keys = ['unique_sgrna_id', 'gene', 'sgrna sequence', 'start', 'end']
    
    # Define exact columns to keep from File 1 (dropping cell_line_origin)
    cols_file1 = merge_keys + ['id', 'distance_to_tss']
    
    # Define exact columns to keep from File 2
    cols_file2 = merge_keys + [
        'extended_sequence', 'strand', 'sigmoid_score', 
        'class', 'probability_score', 'prediction_binary'
    ]
    
    # Verify the required columns exist
    missing_in_df1 = [col for col in cols_file1 if col not in df1.columns]
    if missing_in_df1:
        raise ValueError(f"Required columns missing in File 1: {missing_in_df1}")
        
    missing_in_df2 = [col for col in cols_file2 if col not in df2.columns]
    if missing_in_df2:
        raise ValueError(f"Required columns missing in File 2: {missing_in_df2}")

    # Isolate dataframes before merging to eliminate bloat
    df1_subset = df1[cols_file1]
    df2_subset = df2[cols_file2]
    
    # Perform the inner merge
    merged_df = pd.merge(df1_subset, df2_subset, on=merge_keys, how='inner')
    
    # Enforce the exact final column order you requested
    final_column_order = [
        'unique_sgrna_id', 
        'id', 
        'gene', 
        'sgrna sequence', 
        'extended_sequence', 
        'start', 
        'end', 
        'strand', 
        'sigmoid_score', 
        'class', 
        'probability_score', 
        'prediction_binary', 
        'distance_to_tss'
    ]
    
    merged_df = merged_df[final_column_order]
    
    # Save the output dataset as a comma-separated file
    merged_df.to_csv(output_path, sep=",", index=False)
    
    # Logic check
    print(f"Rows in File 1: {len(df1)}")
    print(f"Rows in File 2: {len(df2)}")
    print(f"Rows in Merged Output: {len(merged_df)}")
    print(f"Process complete. Final formatted file saved to: {output_path}")

# Example execution:
merge_csv_and_txt('Unified_GenomeWide_3Cellline_meta.txt', 'independent_predictions.csv', 'final_output.csv')
