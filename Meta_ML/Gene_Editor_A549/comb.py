import os
import sys
import pandas as pd

def main():
    # ======================================================================
    # 1. CLUSTER WORKSPACE PATH CONFIGURATIONS
    # ======================================================================
    working_dir = "./"
    
    # Path inputs
    epigenetic_file = os.path.join(working_dir, "TreeCRISPRi_lib_TSS_revised_bins_complete_epi_A549.txt") 
    master_editor_file = os.path.join(working_dir, "A549_training_ready_matrix_subset.txt")
    output_merged_file = os.path.join(working_dir, "A549_training_ready_matrix_K.txt")

    if not os.path.exists(master_editor_file):
        print(f"CRITICAL ERROR: Master editor matrix file missing: {master_editor_file}")
        sys.exit(1)
    if not os.path.exists(epigenetic_file):
        print(f"CRITICAL ERROR: Epigenetic features file missing: {epigenetic_file}")
        sys.exit(1)

    # ======================================================================
    # 2. INGEST AND STRIP NON-EPIGENETIC COVARIATES
    # ======================================================================
    print("[*] Ingesting and filtering raw epigenetic track matrices...")
    df_epi = pd.read_csv(epigenetic_file, sep='\t')
    df_epi.columns = [col.strip() for col in df_epi.columns]

    if 'Gene' not in df_epi.columns:
        print("CRITICAL ERROR: 'Gene' column missing from epigenetic file headers!")
        sys.exit(1)

    # Expanded exclusion list to drop metadata coordinates and guide structural vectors
    exclude_cols = [
        'ID', 'sgRNA Sequence', 'Chromosome', 'Start', 'End', 'Strand',
        'start_30', 'end_30', 'extended_sequence', 'PAM', 
        'distance_to_TSS', 'closest_TSS_coord','Gene_Strand'
    ]
    
    # Isolate strictly the Gene key and the epigenetic signal tracks
    keep_cols = [col for col in df_epi.columns if col not in exclude_cols]
    df_epi_filtered = df_epi[keep_cols].copy()

    # De-duplicate to preserve strict 1:1 row integrity during join
    df_epi_filtered = df_epi_filtered.drop_duplicates(subset=['Gene'])

    # ======================================================================
    # 3. INGEST MASTER MATRIX AND JOIN
    # ======================================================================
    print("[*] Ingesting master editor sequence file...")
    df_master = pd.read_csv(master_editor_file, sep='\t')
    df_master.columns = [col.strip() for col in df_master.columns]

    print("[*] Executing strict inner merge to retain only overlapping genes...")
    # how='inner' drops any row where the 'Gene' does not exist in both files
    df_merged = pd.merge(df_master, df_epi_filtered, on='Gene', how='inner')

    # ======================================================================
    # 4. EXPORT FINALIZED READY MATRIX
    # ======================================================================
    print("\n=== Merged Matrix Audit Summary ===")
    print(f" -> Original Master Rows:      {df_master.shape[0]}")
    print(f" -> Cleaned Rows Retained:     {df_merged.shape[0]}")
    print(f" -> Dropped Unmapped Genes:    {df_master.shape[0] - df_merged.shape[0]}")
    print(f" -> Total Column Dimensions:   {df_merged.shape[1]}")
    print("===================================\n")

    print(f"[*] Serializing ML training ready matrix to: {output_merged_file}")
    df_merged.to_csv(output_merged_file, sep='\t', index=False)
    print("[-->] Merge complete. All incomplete rows purged.")

if __name__ == "__main__":
    main()
