import os
import sys
import pandas as pd

def main():
    # 1. Workspace Paths (Configured for active ihc-grid-1-1-1 directories)
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Complete_dataset_sigmoid_score_K562/Isolated_equation/Resistance_data/"
    
    master_feature_file = os.path.join(working_dir, "CRISPR_ml_features_final.csv")
    resistance_genes_file = os.path.join(working_dir, "TreeCRISPRi_drug_resistant_gene_K562_list.txt") 
    output_filtered_file = os.path.join(working_dir, "CRISPR_ml_features_pure_resistance_filtered.csv")

    # 2. Safe Data Ingestion
    if not os.path.exists(master_feature_file):
        print(f"CRITICAL ERROR: Master feature file missing: {master_feature_file}")
        sys.exit(1)
    if not os.path.exists(resistance_genes_file):
        print(f"CRITICAL ERROR: Resistance gene list file missing: {resistance_genes_file}")
        sys.exit(1)

    print("[*] Ingesting datasets...")
    df_master = pd.read_csv(master_feature_file, sep=None, engine='python')
    df_res_list = pd.read_csv(resistance_genes_file, sep='\t')

    # 3. Structural Header Sanitization
    df_master.columns = [str(col).strip().lower() for col in df_master.columns]
    df_res_list.columns = [str(col).strip().lower() for col in df_res_list.columns]

    if 'class' not in df_master.columns:
        print("CRITICAL ERROR: 'class' column missing from the master feature file!")
        sys.exit(1)

    # Dynamically locate the gene column inside your whitelist reference file
    gene_col_candidates = ['gene', 'gene name', 'gene_name', 'symbol', 'approved_symbol']
    res_gene_col = next((c for c in gene_col_candidates if c in df_res_list.columns), df_res_list.columns[0])

    # ==================================================================
    # 4. CRITICAL: DE-DUPLICATE COLUMN HEADERS (AXIS PROTECTION REINDEX)
    # ==================================================================
    if 'gene' not in df_master.columns:
        print("CRITICAL ERROR: 'gene' column missing from the master feature file!")
        sys.exit(1)
        
    # Check if 'gene' tracking key returned a DataFrame (indicates duplicate column indexes exist)
    if isinstance(df_master['gene'], pd.DataFrame):
        print("[!] WARNING: Multiple columns named 'gene' detected. Forcefully dropping duplicate columns...")
        
        cols = df_master.columns.tolist()
        gene_count = 0
        new_cols = []
        
        # Sequentially rename secondary 'gene' columns to isolate the main index vector
        for col in cols:
            if col == 'gene':
                if gene_count > 0:
                    new_cols.append(f'gene_duplicate_dropped_{gene_count}')
                else:
                    new_cols.append('gene')
                gene_count += 1
            else:
                new_cols.append(col)
        
        # Apply sterile unique headers back to the dataframe
        df_master.columns = new_cols
        
        # Drop the renamed duplicate tracks to clear matrix space
        drop_targets = [f'gene_duplicate_dropped_{x}' for x in range(1, gene_count)]
        df_master = df_master.drop(columns=drop_targets)

    # Convert the now guaranteed single 'gene' Series safely to uppercase strings
    df_master['gene'] = df_master['gene'].astype(str).apply(lambda x: x.strip().upper())
    master_genes_before = df_master['gene'].nunique()
    
    # Standardize whitelist reference tokens
    pure_res_genes = set(df_res_list[res_gene_col].astype(str).str.strip().str.upper().unique())

    print(f"--> Raw Feature File Matrix: {len(df_master)} sgRNA rows across {master_genes_before} unique genes.")
    print(f"--> Whitelist Whitelist Pool: {len(pure_res_genes)} unique pure resistance genes.")

    # ==================================================================
    # 5. CROSS-REFERENCE FILTRATION (RETAIN ALL INDIVIDUAL sgRNA LINES)
    # ==================================================================
    print("[*] Intersecting feature matrix with pure resistance gene whitelist...")
    
    # This captures EVERY row where the gene is in the whitelist, preserving duplicate gene entries (different sgRNAs)
    df_filtered = df_master[df_master['gene'].isin(pure_res_genes)].copy().reset_index(drop=True)

    total_filtered_rows = len(df_filtered)
    if total_filtered_rows == 0:
        print("CRITICAL WARNING: Zero rows matched your resistance gene list. Check token nomenclature.")
        sys.exit(1)

    # Save the complete multi-modality slice to disk for downstream model building
    df_filtered.to_csv(output_filtered_file, index=False)

    # ==================================================================
    # 6. CLASS DISTRIBUTION METRICS ANALYSIS
    # ==================================================================
    class_counts = df_filtered['class'].value_counts()
    pos_count = class_counts.get(1, 0)
    neg_count = class_counts.get(0, 0)
    
    pos_pct = (pos_count / total_filtered_rows) * 100 if total_filtered_rows > 0 else 0
    neg_pct = (neg_count / total_filtered_rows) * 100 if total_filtered_rows > 0 else 0

    print("-" * 75)
    print("PURE RESISTANCE FILTERED DATASET MATRIX REPORT")
    print("-" * 75)
    print(f"Saved Filtered Matrix Destination: {output_filtered_file}")
    print(f"Total Retained sgRNA Lines:        {total_filtered_rows} (All sgRNAs preserved)")
    print(f"Total Unique Target Genes:         {df_filtered['gene'].nunique()}")
    print(f"--> Active Positives (Class 1):    {pos_count} rows ({pos_pct:.2f}%)")
    print(f"--> Inactive Baselines (Class 0):  {neg_count} rows ({neg_pct:.2f}%)")
    print("-" * 75)
    
    # Print the calculated imbalance ratio for direct copy-pasting into your XGBoost setup
    if neg_count > 0 and pos_count > 0:
        ratio_inverted = float(pos_count) / neg_count
        print(f"[-->] Suggested XGBoost scale_pos_weight for this slice: {ratio_inverted:.4f}")
        print("-" * 75)

if __name__ == "__main__":
    main()
