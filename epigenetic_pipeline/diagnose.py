import pandas as pd
import numpy as np

def diagnose_missing_genes(lib_path, tss_path):
    # Load files
    lib_df = pd.read_csv(lib_path, sep='\t')
    tss_df = pd.read_csv(tss_path)
    
    # Strip whitespace from columns
    lib_df.columns = lib_df.columns.str.strip()
    tss_df.columns = tss_df.columns.str.strip()
    
    # Find the gene column in both files
    lib_gene_col = 'Gene' if 'Gene' in lib_df.columns else 'gene'
    tss_gene_col = 'gene' if 'gene' in tss_df.columns else 'Gene'
    
    # Process TSS to P1
    t_col = 'transcript' if 'transcript' in tss_df.columns else 'Transcript'
    if t_col in tss_df.columns:
        tss_df = tss_df[tss_df[t_col] == 'P1'].copy()
    
    lib_genes = set(lib_df[lib_gene_col].dropna())
    tss_genes = set(tss_df[tss_gene_col].dropna())
    
    missing_genes = lib_genes.difference(tss_genes)
    
    print(f"Total Unique Genes in Library: {len(lib_genes)}")
    print(f"Total Unique Genes in P1 TSS File: {len(tss_genes)}")
    print(f"Genes missing from the P1 TSS file: {len(missing_genes)}")
    print("\nFirst 10 missing genes:", list(missing_genes)[:10])

if __name__ == "__main__":
    diagnose_missing_genes('TreeCRISPRi_extended_lib.txt', 'TSS_annotation_hg38.csv')
