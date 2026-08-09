import pandas as pd

def add_genomic_coordinates(file1_path, file2_path, output_path):
    # Load the datasets
    df1 = pd.read_csv(file1_path, sep="\t")
    df2 = pd.read_csv(file2_path, sep="\t")
    
    # Strip invisible whitespace from all column names
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()
    
    # Define the core keys that exist in both files to ensure perfect alignment
    merge_keys = ['id', 'sgrna sequence', 'gene']
    
    # Verify the required columns exist in File 1
    missing_in_df1 = [col for col in merge_keys if col not in df1.columns]
    if missing_in_df1:
        raise ValueError(f"Required merge keys missing in File 1: {missing_in_df1}")
        
    # Define the exact columns needed from File 2
    # Note: We intentionally exclude 'extended_sequence' and 'distance_to_TSS' here
    # because File 1 already contains the definitive versions of those metrics.
    cols_file2 = merge_keys + [
        'Chromosome', 'Start', 'End', 'Strand', 
        'start_30', 'end_30', 'PAM', 'closest_TSS_coord'
    ]
    
    # Verify the required columns exist in File 2
    missing_in_df2 = [col for col in cols_file2 if col not in df2.columns]
    if missing_in_df2:
        raise ValueError(f"Required columns missing in File 2: {missing_in_df2}")

    # Isolate only the merge keys and the target genomic metrics from File 2
    df2_subset = df2[cols_file2]
    
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
        'extended_sequence', 
        'sigmoid_score',
        'Chromosome',
        'Start',
        'End',
        'Strand',
        'start_30',
        'end_30',
        'PAM',
        'closest_TSS_coord'
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
    print(f"Rows in File 2 (Coordinates): {len(df2)}")
    print(f"Rows in Merged Output: {len(merged_df)}")
    print(f"Process complete. Final formatted file saved to: {output_path}")

# Example execution:
add_genomic_coordinates('Pre-Validation.txt', 'TreeCRISPR_lib_TSS.txt', 'Validation.txt')
