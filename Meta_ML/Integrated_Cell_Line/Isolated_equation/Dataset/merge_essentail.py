import pandas as pd

# 1. Load both datasets
# If files are tab-separated, add sep='\t' inside read_csv()
file1 = pd.read_csv("TreeCRISPRi_ESG_list.txt",sep='\t')
file2 = pd.read_csv("Unified_GenomeWide_3CellLine_Matrix.txt",sep='\t')

# 2. Extract all unique genes from file1's 'gene' column to create a clean reference array
target_genes = file1["gene"].dropna().unique()

# 3. Filter file2 where the 'gene' value matches any element in the target_genes array
# This automatically retains ALL rows and ALL columns (sgRNA IDs, positional counts, etc.) for those genes
matched_dataset = file2[file2["gene"].isin(target_genes)]

# 4. Export the exact filtered dataset to a new CSV file
matched_dataset.to_csv("Unified_target_genes_data.txt",sep="\t",index=False)

# Validation output
print(f"Filtering complete.")
print(f"Unique target genes identified from File 1: {len(target_genes)}")
print(f"Total matching rows extracted from File 2: {matched_dataset.shape[0]}")
print(f"Total columns preserved: {matched_dataset.shape[1]}")
