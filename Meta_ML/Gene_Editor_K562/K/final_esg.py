import os
import sys
import pandas as pd

def main():
    # ======================================================================
    # 1. CLUSTER WORKSPACE PATH CONFIGURATIONS
    # ======================================================================
    working_dir = "./"  # Set to your active workspace path if running outside the local directory
    
    input_matrix_file = os.path.join(working_dir, "K_gene_editor_balanced_matrix.txt")
    gene_list_file = os.path.join(working_dir, "TreeCRISPRi_ESG_list.txt")
    output_final_file = os.path.join(working_dir, "K_training_ready_matrix_esg.txt")

    if not os.path.exists(input_matrix_file):
        print(f"CRITICAL ERROR: Input training matrix missing: {input_matrix_file}")
        sys.exit(1)
    if not os.path.exists(gene_list_file):
        print(f"CRITICAL ERROR: Target ESG gene list missing: {gene_list_file}")
        sys.exit(1)

    # ======================================================================
    # 2. INGEST GENE FILTER AND MASTER MATRIX
    # ======================================================================
    print("[*] Ingesting target gene filter list...")
    df_genes = pd.read_csv(gene_list_file, sep='\t')
    df_genes.columns = [col.strip() for col in df_genes.columns]
    
    if 'Gene' not in df_genes.columns:
        print("CRITICAL ERROR: 'Gene' header missing from TreeCRISPRi_ESG_list.txt!")
        sys.exit(1)
        
    # Isolate unique target genes as an uppercase set for safe matching
    target_genes = set(df_genes['Gene'].dropna().astype(str).str.strip().str.upper())
    print(f" -> Found {len(target_genes)} unique gene filter targets.")

    print("[*] Ingesting K562 training ready matrix...")
    df_master = pd.read_csv(input_matrix_file, sep='\t')
    df_master.columns = [col.strip() for col in df_master.columns]

    if 'Gene' not in df_master.columns or 'class' not in df_master.columns:
        print("CRITICAL ERROR: Missing 'Gene' or 'class' columns in the input matrix!")
        sys.exit(1)

    # ======================================================================
    # 3. SUBSET MATRIX VIA SET INTERSECTION MATCHING
    # ======================================================================
    print("[*] Subsetting rows matching target gene list...")
    
    # Create an internal helper column for case-insensitive matching
    df_master['_gene_upper'] = df_master['Gene'].astype(str).str.strip().str.upper()
    
    # Filter matrix rows
    df_final = df_master[df_master['_gene_upper'].isin(target_genes)].copy()
    
    # Drop the temporary uppercase column
    df_final = df_final.drop(columns=['_gene_upper'])

    # ======================================================================
    # 4. AUDIT CLASS DISTRIBUTION METRICS & SAVE
    # ======================================================================
    class_counts = df_final['class'].value_counts()
    count_zeros = int(class_counts.get(0, 0))
    count_ones = int(class_counts.get(1, 0))

    print("\n" + "="*50)
    print("      SUBSETTED MATRIX CLASS AUDIT REPORT")
    print("="*50)
    print(f" -> Total Rows Extracted: {df_final.shape[0]}")
    print(f" -> Total Feature Columns: {df_final.shape[1]}")
    print(f" -> Class 0 (Inactive/Non-essential): {count_zeros} rows")
    print(f" -> Class 1 (Active/Essential):     {count_ones} rows")
    print("==================================================\n")

    print(f"[*] Serializing final training subset matrix to: {output_final_file}")
    df_final.to_csv(output_final_file, sep='\t', index=False)
    print("[-->] Script completed successfully.")

if __name__ == "__main__":
    main()
