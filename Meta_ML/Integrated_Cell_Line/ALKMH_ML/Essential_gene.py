import pandas as pd
import os

def main():
    # 1. Define file paths (Update these to match your actual filenames)
    main_dataset_path = "Unified_GenomeWide_3CellLine_Matrix.txt"  # The large file with all columns
    essential_genes_path = "TreeCRISPRi_ESG_list.txt"   # The file containing your gene list
    output_path = "essential_genes_subset.csv"          # The final output file

    # Validate file existence
    if not os.path.exists(main_dataset_path):
        raise FileNotFoundError(f"Error: {main_dataset_path} not found.")
    if not os.path.exists(essential_genes_path):
        raise FileNotFoundError(f"Error: {essential_genes_path} not found.")

    # 2. Load datasets
    print("Loading datasets into memory...")
    # Adjust sep argument if your main file is tab-separated ('\t') instead of comma-separated (',')
    df_main = pd.read_csv(main_dataset_path, sep='\t') 
    df_list = pd.read_csv(essential_genes_path, sep='\t')

    print("\n==================================================")
    print(f"INITIAL MAIN DATASET: {df_main.shape[0]} rows, {df_main.shape[1]} columns")
    print("==================================================\n")

    # 3. Identify the gene columns safely 
    # Handles potential capitalization differences (e.g., 'gene' vs 'Gene')
    main_gene_col = 'gene' if 'gene' in df_main.columns else 'Gene'
    list_gene_col = 'gene' if 'gene' in df_list.columns else 'Gene'

    if main_gene_col not in df_main.columns:
        raise ValueError(f"Error: Neither 'gene' nor 'Gene' found in {main_dataset_path}")
    if list_gene_col not in df_list.columns:
        raise ValueError(f"Error: Neither 'gene' nor 'Gene' found in {essential_genes_path}")

    # Standardize to uppercase strings to prevent case-sensitive mismatches (e.g., 'AAMP' vs 'aamp')
    df_main['match_key'] = df_main[main_gene_col].astype(str).str.upper()
    df_list['match_key'] = df_list[list_gene_col].astype(str).str.upper()

    # 4. Extract target list and subset the main dataframe
    print("Extracting rows matching the essential gene list...")
    target_genes = df_list['match_key'].unique()
    
    # .isin() acts as a boolean mask, pulling only rows where the match_key is in your target list
    df_subset = df_main[df_main['match_key'].isin(target_genes)].copy()

    # 5. Clean up temporary columns
    df_subset = df_subset.drop(columns=['match_key'])

    print("\n==================================================")
    print(f"SUBSET DATASET: {df_subset.shape[0]} rows, {df_subset.shape[1]} columns")
    print(f"Total Unique Genes Captured: {df_subset[main_gene_col].nunique()}")
    print("==================================================\n")

    # 6. Save the standalone matrix
    df_subset.to_csv(output_path, index=False)
    print(f"Data successfully subset and saved to: {output_path}")

if __name__ == "__main__":
    main()
