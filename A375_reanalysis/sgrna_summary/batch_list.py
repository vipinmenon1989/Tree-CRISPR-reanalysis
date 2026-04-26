import pandas as pd
import argparse
import sys
import os
import glob

def batch_process_folder(input_dir, output_dir):
    # 1. Setup output directory
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 2. Grab all files in the directory (ignoring already processed ones)
    search_path = os.path.join(input_dir, "*")
    files = [f for f in glob.glob(search_path) if os.path.isfile(f) and not f.endswith('_gene_list.csv')]
    
    if not files:
        print(f"CRITICAL ERROR: No files found in directory: {input_dir}")
        sys.exit(1)
        
    print(f"Found {len(files)} files. Starting batch execution...\n")
    success_count = 0
    
    # 3. Loop through and process each file
    for file_path in files:
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        
        # Construct the requested output filename: prefix + _gene_list.csv
        out_name = f"{base_name}_gene_list.csv"
        out_path = os.path.join(output_dir if output_dir else input_dir, out_name)
        
        try:
            df = pd.read_csv(file_path, sep=None, engine='python')
        except Exception as e:
            print(f"[SKIP] {file_name}: Could not read file. ({e})")
            continue
            
        # 4. Auto-Detect Columns (Handles both 'Gene'/'gene' and 'Sigmoid_Score'/'Delta_Score')
        gene_col = 'gene' if 'gene' in df.columns else ('Gene' if 'Gene' in df.columns else None)
        score_col = 'Sigmoid_Score' if 'Sigmoid_Score' in df.columns else ('Delta_Score' if 'Delta_Score' in df.columns else None)
        
        if not gene_col or not score_col:
            print(f"[SKIP] {file_name}: Missing target columns. Found: {list(df.columns)}")
            continue
            
        # 5. GroupBy Mean and Export
        try:
            # Collapse duplicates into a mean score
            df_mean = df.groupby(gene_col, as_index=False)[score_col].mean()
            
            # Standardize output headers to always be lowercase for downstream ML merging
            df_mean.rename(columns={gene_col: 'gene', score_col: 'score'}, inplace=True)
            
            # Save the file
            df_mean.to_csv(out_path, index=False)
            print(f"[SUCCESS] {file_name} -> {out_name} ({len(df_mean)} unique genes)")
            success_count += 1
        except Exception as e:
            print(f"[ERROR] {file_name}: Processing failed. ({e})")
            
    print(f"\nExecution Complete: Successfully collapsed {success_count} out of {len(files)} files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch calculate Mean Scores for an entire folder.")
    parser.add_argument("-d", "--dir", required=True, help="Input directory with your raw files")
    parser.add_argument("-o", "--outdir", help="Optional: Directory to save outputs (defaults to input dir)", default=None)
    
    args = parser.parse_args()
    batch_process_folder(args.dir, args.outdir)

