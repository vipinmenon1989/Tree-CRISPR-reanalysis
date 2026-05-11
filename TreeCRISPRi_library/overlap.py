import pandas as pd
import sys
import os

def check_overlap(lib_path, tss_path):
    # Load Library
    lib_df = pd.read_csv(lib_path)
    
    # Load TSS file (assuming tab-separated based on the previous command)
    if tss_path.endswith('.txt') or tss_path.endswith('.tsv'):
        tss_df = pd.read_csv(tss_path, sep='\t')
    else:
        tss_df = pd.read_csv(tss_path)
        
    # Strip whitespace from column names
    lib_df.columns = lib_df.columns.str.strip()
    tss_df.columns = tss_df.columns.str.strip()
    
    print(f"Library columns: {list(lib_df.columns)}")
    print(f"TSS columns: {list(tss_df.columns)}")
    
    # Identify gene column dynamically
    lib_gene_col = 'Gene' if 'Gene' in lib_df.columns else 'gene'
    if 'gene' in tss_df.columns:
        tss_gene_col = 'gene'
    elif 'Gene' in tss_df.columns:
        tss_gene_col = 'Gene'
    else:
        raise KeyError(f"Could not find gene column in TSS file. Available columns: {list(tss_df.columns)}")
        
    lib_genes = set(lib_df[lib_gene_col].unique())
    tss_genes = set(tss_df[tss_gene_col].unique())
    
    print(f"\nTotal genes in Library: {len(lib_genes)}")
    print(f"Total genes in TSS file: {len(tss_genes)}")
    
    overlap = lib_genes.intersection(tss_genes)
    print(f"Genes present in BOTH files: {len(overlap)}")
    
    missing = lib_genes.difference(tss_genes)
    print(f"Genes missing from TSS file: {len(missing)}")
    
    # Show the first 5 missing genes
    print("\nSample of missing genes:", list(missing)[:5])

if __name__ == "__main__":
    check_overlap('bed_sgnra_example_ALKE.csv', 'TreeCRISPRi_lib_TSS_P1.txt')
