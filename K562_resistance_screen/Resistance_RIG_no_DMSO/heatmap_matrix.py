import os
import glob
import pandas as pd
from functools import reduce
import argparse
import sys

def build_traceable_matrix(input_folder, output_file):
    print(f"Scanning folder: {input_folder} for CSV files...")
    # Supporting both .csv and .txt
    csv_files = glob.glob(os.path.join(input_folder, "*.csv")) + glob.glob(os.path.join(input_folder, "*.txt"))
    
    if not csv_files:
        print(f"CRITICAL ERROR: No data files found in {input_folder}")
        sys.exit(1)

    dataframes = []

    for file_path in csv_files:
        raw_filename = os.path.basename(file_path)
        editor_column_name = os.path.splitext(raw_filename)[0]
        
        try:
            # Handles comma and tab automatically
            df = pd.read_csv(file_path, sep=None, engine='python')
        except Exception as e:
            print(f"Error reading {raw_filename}: {e}")
            continue
            
        # 1. DYNAMIC HEADER CHECK
        # Checks for 'Score_RIG' first, then falls back to 'Delta_Score'
        target_gene_col = 'Gene'
        possible_score_cols = ['Score_RIG', 'Delta_Score']
        actual_score_col = next((col for col in possible_score_cols if col in df.columns), None)

        if target_gene_col not in df.columns or actual_score_col is None:
            print(f"WARNING: Skipping {raw_filename} - Required columns not found.")
            print(f"Found columns: {list(df.columns)}")
            continue
            
        # 2. Aggregation
        # Explicitly select only the Gene and the Score to avoid errors with the 'sgrna' string column
        df_subset = df[[target_gene_col, actual_score_col]]
        df_mean = df_subset.groupby(target_gene_col, as_index=False)[actual_score_col].mean()
        
        # 3. Rename to the Filename
        df_mean = df_mean.rename(columns={actual_score_col: editor_column_name})
        
        dataframes.append(df_mean)
        print(f"Extracted {len(df_mean)} genes using '{actual_score_col}' -> {raw_filename}")

    if not dataframes:
        print("CRITICAL ERROR: No valid dataframes to merge.")
        sys.exit(1)

    print("\nExecuting Outer Merge...")
    
    # 4. Merge on 'Gene'
    final_matrix = reduce(lambda left, right: pd.merge(left, right, on='Gene', how='outer'), dataframes)

    # 5. Output
    final_matrix.to_csv(output_file, index=False)
    
    print(f"\nSuccess! Traceable Matrix saved to: {output_file}")
    print(f"Total Unique Genes: {final_matrix.shape[0]}")
    print(f"Editors Merged: {list(final_matrix.columns)[1:]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a gene-level matrix using Gene and Score_RIG/Delta_Score.")
    parser.add_argument("-d", "--dir", required=True, help="Directory with result files")
    parser.add_argument("-o", "--output", default="Trajectory_matrix.csv", help="Output filename")
    
    args = parser.parse_args()
    build_traceable_matrix(args.dir, args.output)
