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
        # 1. EXACT TRACKING: Extract the exact filename without the .csv extension
        # If the file is "ALKE_replicate_1.csv", the column will be "ALKE_replicate_1"
        raw_filename = os.path.basename(file_path)
        editor_column_name = raw_filename.replace('.csv', '')
        
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {raw_filename}: {e}")
            continue
            
        # 2. Strict column verification
        if 'gene' not in df.columns or 'Sigmoid_Score' not in df.columns:
            print(f"WARNING: Skipping {raw_filename} - Missing 'gene' or 'Sigmoid_Score'.")
            continue
            
        # 3. Collapse multiple guides into a single gene-level Mean score
        df_mean = df.groupby('gene', as_index=False)['Sigmoid_Score'].mean()
        
        # 4. RENAME THE COLUMN TO THE EXACT FILENAME
        df_mean = df_mean.rename(columns={'Sigmoid_Score': editor_column_name})
        
        dataframes.append(df_mean)
        print(f"Extracted {len(df_mean)} genes from -> {raw_filename} (Column will be: '{editor_column_name}')")

    if not dataframes:
        print("CRITICAL ERROR: No valid dataframes to merge.")
        sys.exit(1)

    print("\nExecuting Outer Merge to align all genes...")
    
    # 5. Merge all files together. 
    # 'outer' ensures genes unique to one editor are kept, filled with NaN for others.
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


