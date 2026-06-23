import pandas as pd
import sys
import os

def pre_merge_audit_and_link(file_epi_path, file_ml_path, output_path):
    print("======================================================================")
    print("CRISPRi ALIGNMENT ENGINE: PRE-MERGE AUDIT & INTERSECTION VERIFICATION")
    print("======================================================================\n")
    
    # ---- STEP 1: VERIFY FILE EXISTENCE ----
    if not os.path.exists(file_epi_path):
        print(f"CRITICAL ERROR: Epigenetic file missing at '{file_epi_path}'")
        sys.exit(1)
    if not os.path.exists(file_ml_path):
        print(f"CRITICAL ERROR: ML Matrix file missing at '{file_ml_path}'")
        sys.exit(1)

    # ---- STEP 2: INGEST DATA ----
    print("[*] Reading cluster files...")
    df_epi = pd.read_csv(file_epi_path, sep='\t')
    df_ml = pd.read_csv(file_ml_path, sep=',')

    # ---- STEP 3: SANITIZE HEADERS AND DATA TYPES ----
    print("[*] Sanitizing file headers and coordinate schemas...")
    df_epi.columns = df_epi.columns.str.strip()
    df_ml.columns = df_ml.columns.str.strip()

    # Enforce exact column naming for keys
    if df_ml.columns[0] != 'ID' and 'ID' not in df_ml.columns:
        df_ml.rename(columns={df_ml.columns[0]: 'ID'}, inplace=True)

    # Core Sanity Step: Force uniform string types and strip whitespace from content
    for df in [df_epi, df_ml]:
        df['ID'] = df['ID'].astype(str).str.strip()
        df['Gene'] = df['Gene'].astype(str).str.strip().str.upper() # Force uppercase to avoid case traps

    # ---- STEP 4: PRE-MERGE STRUCTURAL AUDIT ----
    print("\n" + "-"*50)
    print("PRE-MERGE DIAGNOSTIC REPORT")
    print("-"*50)
    
    epi_genes = set(df_epi['Gene'].unique())
    ml_genes = set(df_ml['Gene'].unique())
    gene_intersection = epi_genes.intersection(ml_genes)
    
    epi_ids = set(df_epi['ID'].unique())
    ml_ids = set(df_ml['ID'].unique())
    id_intersection = epi_ids.intersection(ml_ids)

    # Create compound keys (ID_Gene) to audit true coordinate matching
    epi_pairs = set((row['ID'], row['Gene']) for _, row in df_epi[['ID', 'Gene']].iterrows())
    ml_pairs = set((row['ID'], row['Gene']) for _, row in df_ml[['ID', 'Gene']].iterrows())
    pair_intersection = epi_pairs.intersection(ml_pairs)

    print(f"[*] Epigenetic File Key Matrix : {df_epi.shape[0]} rows | {len(epi_genes)} unique Genes | {len(epi_ids)} unique IDs")
    print(f"[*] ML Matrix Target File      : {df_ml.shape[0]} rows | {len(ml_genes)} unique Genes | {len(ml_ids)} unique IDs")
    print(f"[*] Shared Gene Overlap Count  : {len(gene_intersection)} genes match exactly.")
    print(f"[*] Shared ID Overlap Count    : {len(id_intersection)} coordinate indices match exactly.")
    print(f"[*] Combined ID+Gene Intersect : {len(pair_intersection)} unique row anchors match exactly.")
    print("-"*50 + "\n")

    # Hard Gatekeeper Guardrail
    if len(pair_intersection) == 0:
        print("CRITICAL PIPELINE FAILURE: Zero rows overlap between your two files using the combined ['ID', 'Gene'] key.")
        print("Possible causes: Data type mismatch (e.g. float indices vs integer strings) or divergent identifier schemas.")
        print("Aborting merge execution to prevent empty file generation.")
        sys.exit(1)

    # ---- STEP 5: SECURE INNER JOIN ----
    print("[*] Pre-check passed. Executing multi-key inner join...")
    merged_df = pd.merge(
        df_epi, 
        df_ml, 
        on=['ID', 'Gene'], 
        how='inner'
    )

    # ---- STEP 6: CLEANUP RESIDUAL ARTIFACTS ----
    duplicate_cols = [c for c in merged_df.columns if c.endswith('_hits') or c.endswith('_epi')]
    if duplicate_cols:
        cols_to_drop = [c for c in duplicate_cols if c.endswith('_hits')]
        merged_df.drop(columns=cols_to_drop, inplace=True)
        merged_df.columns = merged_df.columns.str.replace('_epi', '')

    # Ensure target class labels are pushed to the absolute end column position
    if 'class' in merged_df.columns:
        class_series = merged_df.pop('class')
        merged_df['class'] = class_series

    # ---- STEP 7: EXPORT TO HPC WORKSPACE ----
    merged_df.to_csv(output_path, sep='\t', index=False)
    print("\n======================================================================")
    print(f"SUCCESS: Matrix synchronized and verified. Exported to: {output_path}")
    print(f"Final production matrix dimensions: {merged_df.shape[0]} rows by {merged_df.shape[1]} columns.")
    print("======================================================================")

if __name__ == "__main__":
    file_1 = 'TreeCRISPRi_lib_merged_K562_epigenetic_feature_scaled.txt'
    file_2 = 'Output_File3_Balanced_ML_Matrix.csv'
    output_file = 'ALKE_K562_epigenetic_training.txt'

    pre_merge_audit_and_link(file_1, file_2, output_file)
