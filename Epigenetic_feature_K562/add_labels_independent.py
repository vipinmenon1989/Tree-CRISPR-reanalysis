import pandas as pd
import argparse
import sys

def label_dataset_classes(input_file):
    # 1. Load the dataset (preserving tab separation)
    try:
        df = pd.read_csv(input_file, sep='\t')
    except Exception as e:
        print(f"Error loading {input_file}: {e}")
        sys.exit(1)

    # Standardize column naming variations if present
    if 'Sigmoid_Score' in df.columns and 'Score_RIG' not in df.columns:
        df.rename(columns={'Sigmoid_Score': 'Score_RIG'}, inplace=True)

    # Verify that the required columns exist
    if 'Score_RIG' not in df.columns or 'Delta_Score' not in df.columns:
        print(f"Error: Missing required columns ('Score_RIG' and 'Delta_Score') in {input_file}")
        sys.exit(1)

    # 2. Vectorized assignment of the binary class label
    # Returns 1 if BOTH conditions are met, otherwise returns 0
    hit_condition = (df['Score_RIG'] > 0.25) & (df['Delta_Score'] > 0.1)
    df['class'] = hit_condition.astype(int)

    # 3. Overwrite and update the input file directly
    try:
        df.to_csv(input_file, sep='\t', index=False)
    except Exception as e:
        print(f"Error saving updates to {input_file}: {e}")
        sys.exit(1)

    # Print summary metrics for validation
    total_rows = len(df)
    pos_count = df['class'].sum()
    neg_count = total_rows - pos_count

    print("-" * 40)
    print(f"File Updated Successfully: {input_file}")
    print(f"Total processed rows: {total_rows}")
    print(f"  --> Class 1 (Hits): {pos_count}")
    print(f"  --> Class 0 (Rest): {neg_count}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Add binary class labels to CRISPRi dataset')
    parser.add_argument('-i', '--input', required=True, help='Path to the tab-separated input file')
    
    args = parser.parse_args()
    label_dataset_classes(args.input)
