import os
import shutil
import pandas as pd
import re
import glob

def process_and_merge_sgrna():
    source_dir = "./sgRNA_summary"
    
    # Logic check: Ensure source directory exists before execution
    if not os.path.exists(source_dir):
        print(f"Error: Directory '{source_dir}' not found.")
        return

    # Defined parameters
    # The order matters. Longest strings are matched first to prevent 'K' matching inside 'ALKM_E'.
    groups = ['ALKM_E', 'ALKM_L', 'ALKM_H', 'ALK_E', 'ALK_L', 'ALK_H', 'WT', 'K']
    treatments = ['dmso', 'rig']

    group_pattern = '|'.join(groups)
    treatment_pattern = '|'.join(treatments)

    # Iterate through all text files in the source directory
    for file_path in glob.glob(os.path.join(source_dir, "*.txt")):
        filename = os.path.basename(file_path)
        
        # 1. Regex parsing for group and treatment
        group_match = re.search(group_pattern, filename)
        treatment_match = re.search(treatment_pattern, filename)

        if not group_match or not treatment_match:
            print(f"Skipping {filename}: Missing recognizable group/treatment pattern.")
            continue

        group_name = group_match.group()
        treatment_name = treatment_match.group()

        # 2. Directory management
        target_dir = os.path.join(group_name, treatment_name)
        if not os.path.exists(target_dir):
            print(f"Skipping {filename}: Target directory '{target_dir}' does not exist. Run previous gene filtering steps first.")
            continue

        # 3. Copy sgRNA summary to target directory
        dest_file_path = os.path.join(target_dir, filename)
        shutil.copy(file_path, dest_file_path)
        
        # 4. Identify the corresponding filtered gene file
        filtered_gene_file = os.path.join(target_dir, f"{group_name}_{treatment_name}_filtered.txt")
        if not os.path.exists(filtered_gene_file):
            print(f"Error: Missing filtered gene file '{filtered_gene_file}'. Cannot perform merge.")
            continue

        try:
            # 5. Process Filtered Gene Data
            df_genes = pd.read_csv(filtered_gene_file, sep='\t')
            
            # Hard filter: Retain only True significance
            df_genes_sig = df_genes[df_genes['p-value significant'] == True].copy()
            
            # Rename column for matching
            df_genes_sig.rename(columns={'id': 'Gene'}, inplace=True)
            
            # Extract just the 'Gene' column to act as the merge key
            key_genes = df_genes_sig[['Gene']].drop_duplicates()

            # 6. Process sgRNA Data
            df_sgrna = pd.read_csv(dest_file_path, sep='\t')
            
            # Verify necessary columns exist
            required_cols = ['Gene', 'LFC', 'p.twosided']
            if not all(col in df_sgrna.columns for col in required_cols):
                print(f"Error: {filename} is missing required columns (Gene, LFC, or p.twosided).")
                continue

            # 7. Merge Operation (Inner Join)
            # This mathematically intersects the data, dropping any sgRNA rows tied to non-significant genes.
            df_merged = pd.merge(key_genes, df_sgrna, on='Gene', how='inner')
            
            # Subset to the strictly requested columns
            df_final = df_merged[['Gene', 'LFC', 'p.twosided']]

            # 8. Save Data
            output_name = f"{group_name}_{treatment_name}_sgrna_merged.txt"
            output_path = os.path.join(target_dir, output_name)
            
            df_final.to_csv(output_path, sep='\t', index=False)
            print(f"Successfully generated merged file: {output_path}")

        except Exception as e:
            print(f"Execution failed for {group_name}/{treatment_name}: {e}")

if __name__ == "__main__":
    process_and_merge_sgrna()
