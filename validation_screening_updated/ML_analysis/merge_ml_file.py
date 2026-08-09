import pandas as pd
import os

def main():
    # Define file paths
    file1_path = "guide_matrix_output.txt"  # File containing the guide_* epigenetic columns
    file2_path = "CRISPR_ml_features_output.csv"  # File containing the sequence ML features
    output_path = "CRISPR_merged_final_ML_final.csv"

    # Verify files exist
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        raise FileNotFoundError("Error: Ensure both file1.txt and file2.csv exist in the directory.")

    # Load datasets (Assuming file1 is tab-separated based on the header format, and file2 is comma-separated)
    df1 = pd.read_csv(file1_path, sep='\t')
    df2 = pd.read_csv(file2_path, sep=',')

    # Identify all columns in file1 that start with 'guide_'
    guide_cols = [col for col in df1.columns if col.startswith('guide_')]

    # Set the merge keys to ID and sgRNA (protospacer sequence)
    merge_keys = ['ID', 'protospacer sequence']

    # Validate that the merge keys exist in both dataframes
    for key in merge_keys:
        if key not in df1.columns:
            raise KeyError(f"Error: Merge key '{key}' not found in {file1_path}.")
        if key not in df2.columns:
            raise KeyError(f"Error: Merge key '{key}' not found in {file2_path}.")

    # Subset file1 to contain only the merge keys and the guide_* columns
    df1_subset = df1[merge_keys + guide_cols]

    # Merge all contents of file2 with the isolated guide_* columns from file1
    # A left merge ensures no rows from file2 are dropped
    merged_df = pd.merge(df2, df1_subset, on=merge_keys, how='left')

    # Export the final merged dataset
    merged_df.to_csv(output_path, index=False)
    print(f"[-->] Completed. Merged final feature matrix saved to: {output_path}")

if __name__ == "__main__":
    main()
