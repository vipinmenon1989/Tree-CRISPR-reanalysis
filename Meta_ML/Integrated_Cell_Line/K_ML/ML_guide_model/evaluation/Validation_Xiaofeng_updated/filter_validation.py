import pandas as pd
import numpy as np

def filter_xiaofeng_validation_list(input_path, output_path):
    # Load the dataset. 
    # Using sep="," assuming the input file is the comma-separated output from the previous step.
    df = pd.read_csv(input_path, sep=",")
    
    # Strip invisible whitespace from all column names
    df.columns = df.columns.str.strip()
    
    # Verify the required columns exist
    required_cols = ['sigmoid_score', 'probability_score', 'gene', 'sgrna sequence']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
        
    # Condition 1: New High boundary
    cond_high = (df['sigmoid_score'] > 0.5) & (df['probability_score'] > 0.5)
    
    # Condition 2: New Low boundary
    cond_low = (df['sigmoid_score'] < 0.25) & (df['probability_score'] < 0.5)
    
    # Isolate rows meeting either condition. 
    # .copy() forces allocation of a new memory block to safely add the Effectiveness column.
    filtered_df = df[cond_high | cond_low].copy()
    
    # Calculate pre-guide-filter row count for quality control
    rows_after_thresholds = len(filtered_df)
    
    # Add the 'Effectiveness' column
    # Vectorized assignment: 'High' for cond_high matches, 'Low' for the remaining cond_low matches.
    filtered_df['Effectiveness'] = np.where(
        (filtered_df['sigmoid_score'] > 0.5) & (filtered_df['probability_score'] > 0.5),
        'High',
        'Low'
    )
    
    # Filter for minimum 2 guides per gene
    # transform('size') computes the frequency of each gene and maps it back to the rows
    guide_counts = filtered_df.groupby('gene')['sgrna sequence'].transform('size')
    filtered_df = filtered_df[guide_counts >= 2]
    
    # Sort the final output alphabetically by gene for clean visualization
    filtered_df = filtered_df.sort_values(by='gene', ascending=True)
    
    # Save the strictly formatted dataset. 
    # Using sep="\t" for standard bioinformatics text file output.
    filtered_df.to_csv(output_path, sep="\t", index=False)
    
    # Logic check: Print the exact breakdown of the filtering mechanics
    print(f"Total input rows: {len(df)}")
    print(f"Rows meeting threshold boundaries: {rows_after_thresholds}")
    print(f"Rows dropped due to having only 1 guide per gene: {rows_after_thresholds - len(filtered_df)}")
    print(f"Total rows saved to final list: {len(filtered_df)}")
    print(f"Process complete. File saved to: {output_path}")

# Example execution:
filter_xiaofeng_validation_list('Validation_Xiaofeng.txt', 'Validation_Xiaofeng_vipin_list.txt')
