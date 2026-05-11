import os
import glob
import pandas as pd
from functools import reduce
import argparse
import sys

def build_traceable_matrix(input_folder, output_file):
    print(f"Scanning folder: {input_folder} for CSV files...")
    csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not csv_files:
        print(f"CRITICAL ERROR: No CSV files found in {input_folder}")
        sys.exit(1)

    dataframes = []

    for file_path in csv_files:
        raw_filename = os.path.basename(file_path)
        editor_column_name = raw_filename.replace('.csv', '')
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {raw_filename}: {e}")
            continue
            
        # 1. Normalize column names to lowercase and replace spaces with underscores for robust matching
        # This handles "Delta Score", "delta_score", "Delta_Score", etc.
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
        # 2. Strict column verification 
        if 'gene' not in df.columns or 'delta_score' not in df.columns:
            print(f"WARNING: Skipping {raw_filename} - Missing 'gene' or 'delta_score' (Case-insensitive).")
            continue
            
        # 3. Collapse multiple guides into a single gene-level Mean score
        df_mean = df.groupby('gene', as_index=False)['delta_score'].mean()
        
        # 4. RENAME THE COLUMN TO THE EXACT FILENAME
        df_mean = df_mean.rename(columns={'delta_score': editor_column_name})
        
        dataframes.append(df_mean)
        print(f"Extracted {len(df_mean)} genes from -> {raw_filename} (Column: '{editor_column_name}')")

    if not dataframes:
        print("CRITICAL ERROR: No valid dataframes to merge.")
        sys.exit(1)

    print("\nExecuting Outer Merge to align all genes...")
    
    # 5. Merge all files together. 
    final_matrix = reduce(lambda left, right: pd.merge(left, right, on='gene', how='outer'), dataframes)

    # 6. Save the final matrix
    final_matrix.to_csv(output_file, index=False, na_rep='NaN')
    
    print(f"\nSuccess! Final traceable matrix saved to: {output_file}")
    print(f"Total Genes in Matrix: {final_matrix.shape[0]}")
    print(f"Columns (Editors) Tracked: {list(final_matrix.columns)[1:]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a gene-level heatmap matrix strictly tracking filenames.")
    parser.add_argument("-d", "--dir", required=True, help="Directory with your editor CSVs")
    parser.add_argument("-o", "--output", default="Traceable_Gene_Matrix.csv", help="Output matrix filename")
    
    args = parser.parse_args()
    build_traceable_matrix(args.dir, args.output)
