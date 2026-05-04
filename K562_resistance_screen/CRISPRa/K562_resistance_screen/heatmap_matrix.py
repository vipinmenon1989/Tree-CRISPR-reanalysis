import os
import glob
import pandas as pd
from functools import reduce
import argparse
import sys

def build_traceable_matrix(input_folder, output_file):
    print(f"Scanning folder: {input_folder} for CSV files...")
    # Supporting both .csv and .txt (since MAGeCK outputs are often tab-separated)
    csv_files = glob.glob(os.path.join(input_folder, "*.csv")) + glob.glob(os.path.join(input_folder, "*.txt"))
    
    if not csv_files:
        print(f"CRITICAL ERROR: No data files found in {input_folder}")
        sys.exit(1)

    dataframes = []

    for file_path in csv_files:
        raw_filename = os.path.basename(file_path)
        editor_column_name = os.path.splitext(raw_filename)[0]
        
        try:
            # MAGeCK and your previous scripts use Tabs (\t). 
            # This try-block handles both comma and tab automatically.
            df = pd.read_csv(file_path, sep=None, engine='python')
        except Exception as e:
            print(f"Error reading {raw_filename}: {e}")
            continue
            
        # 1. FIXED HEADERS: Using Title Case 'Gene' and 'Delta_Score'
        target_gene_col = 'Gene'
        target_score_col = 'Delta_Score'

        if target_gene_col not in df.columns or target_score_col not in df.columns:
            print(f"WARNING: Skipping {raw_filename} - Headers must be 'Gene' and 'Delta_Score'.")
            print(f"Found columns: {list(df.columns)}")
            continue
            
        # 2. Aggregation: Collapse multiple sgRNAs per Gene into a Mean score
        # Using Delta_Score as the metric for Resistance Editors
        df_mean = df.groupby(target_gene_col, as_index=False)[target_score_col].mean()
        
        # 3. Rename to the Filename (e.g., ALKH_Resistance)
        df_mean = df_mean.rename(columns={target_score_col: editor_column_name})
        
        dataframes.append(df_mean)
        print(f"Extracted {len(df_mean)} genes from -> {raw_filename}")

    if not dataframes:
        print("CRITICAL ERROR: No valid dataframes to merge.")
        sys.exit(1)

    print("\nExecuting Outer Merge...")
    
    # 4. Merge on 'Gene' (Title Case)
    # Using 'outer' ensures we don't lose genes that only appear in one editor screen
    final_matrix = reduce(lambda left, right: pd.merge(left, right, on='Gene', how='outer'), dataframes)

    # 5. Final Cleanup
    # Optional: Fill NaN with 0 if you assume missing genes = no effect
    # final_matrix = final_matrix.fillna(0)

    final_matrix.to_csv(output_file, index=False)
    
    print(f"\nSuccess! Traceable Matrix saved to: {output_file}")
    print(f"Total Unique Genes: {final_matrix.shape[0]}")
    print(f"Editors Merged: {list(final_matrix.columns)[1:]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a gene-level matrix using Gene and Delta_Score.")
    parser.add_argument("-d", "--dir", required=True, help="Directory with ALKH/ALKE result files")
    parser.add_argument("-o", "--output", default="Master_Editor_Comparison_Matrix.csv", help="Output filename")
    
    args = parser.parse_args()
    build_traceable_matrix(args.dir, args.output)

