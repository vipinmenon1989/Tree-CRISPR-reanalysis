import pandas as pd
import numpy as np

def filter_min_guides_and_sort(input_path, output_path):
    # Load the dataset
    df = pd.read_csv(input_path, sep="\t")
    
    # Strip invisible whitespace from all column names
    df.columns = df.columns.str.strip()
    
    # Verify the required columns exist
    required_cols = ['sigmoid_score', 'prediction_probability', 'gene']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
        
    # Condition 1: Hardcoded positive boundary
    cond_positive = (df['sigmoid_score'] > 0.25) & (df['prediction_probability'] > 0.5)
    
    # Condition 2: Hardcoded negative boundary
    cond_negative = (df['sigmoid_score'] <= 0.05) & (df['prediction_probability'] <= 0.5)
    
    # Apply threshold filters. 
    # .copy() ensures we own this new memory block so we can safely add columns to it.
    filtered_df = df[cond_positive | cond_negative].copy()
    
    # Add the 'effective' column
    # Since all rows in filtered_df are already guaranteed to be either positive or negative,
    # we assign 'Considered' to the positive rows and default everything else to 'Low'.
    filtered_df['effective'] = np.where(
        (filtered_df['sigmoid_score'] > 0.25) & (filtered_df['prediction_probability'] > 0.5),
        'High',
        'Low'
    )
    
    # Calculate pre-guide-filter row count for verification
    rows_after_thresholds = len(filtered_df)
    
    # Remove genes with fewer than 2 guides
    # transform('size') broadcasts the row count back to the original index
    guide_counts = filtered_df.groupby('gene')['gene'].transform('size')
    filtered_df = filtered_df[guide_counts >= 2]
    
    # Sort the final output alphabetically by gene
    filtered_df = filtered_df.sort_values(by='gene', ascending=True)
    
    # Save the filtered and sorted dataset
    filtered_df.to_csv(output_path, sep="\t", index=False)
    
    # Logic check: Print the exact breakdown of the filtering mechanics
    print(f"Total input rows: {len(df)}")
    print(f"Rows meeting threshold boundaries: {rows_after_thresholds}")
    print(f"Rows dropped due to having only 1 guide for the gene: {rows_after_thresholds - len(filtered_df)}")
    print(f"Total rows saved to output (Sorted by gene A-Z, Min 2 guides): {len(filtered_df)}")

# Example execution:
filter_min_guides_and_sort('Validation_sorted.txt', 'Validation_filtered_vipin_list.txt')