import os
import sys
import pandas as pd

def main():
    # ======================================================================
    # 1. CLUSTER WORKSPACE PATH CONFIGURATIONS
    # ======================================================================
    working_dir = "./"
    
    # Target file output destination
    output_master_file = os.path.join(working_dir, "K562_all_editors_master_matrix.txt")

    # Explicit roster of your balanced matrix split paths
    target_files = [
        "ALKE_gene_editor_balanced_matrix.txt",
        "ALKH_gene_editor_balanced_matrix.txt",
        "ALKL_gene_editor_balanced_matrix.txt",
        "ALKME_gene_editor_balanced_matrix.txt",
        "ALKMH_gene_editor_balanced_matrix.txt",
        "ALKML_gene_editor_balanced_matrix.txt"
    ]

    if not os.path.exists(working_dir):
        print(f"CRITICAL ERROR: Target directory context missing: {working_dir}")
        sys.exit(1)

    # ======================================================================
    # 2. INGEST AND CONCATENATE DATASETS SEQUENTIALLY
    # ======================================================================
    df_list = []
    
    print("[*] Commencing structural loop across individual editor pools...")
    for file_name in target_files:
        full_path = os.path.join(working_dir, file_name)
        
        if not os.path.exists(full_path):
            print(f"CRITICAL ERROR: Expected data matrix slice missing: {full_path}")
            sys.exit(1)
            
        print(f" -> Reading {file_name}...")
        df_slice = pd.read_csv(full_path, sep='\t')
        df_slice.columns = [col.strip() for col in df_slice.columns]
        df_list.append(df_slice)

    # Vertically stack all rows
    print("[*] Performing row-wise binding of data frames...")
    df_master = pd.concat(df_list, axis=0, ignore_index=True)

    # ======================================================================
    # 3. GENERATE AND POSITION UNIQUE TRACKING ID ARRAY
    # ======================================================================
    print("[*] Generating unique indexing vector (Gene_1, Gene_2, ...)...")
    
    # Generate 1-indexed sequential strings
    df_master['ID'] = [f"Gene_{i}" for i in range(1, len(df_master) + 1)]

    # Reorder columns to force ID to be the absolute first column on the left
    all_columns = list(df_master.columns)
    ordered_columns = ['ID'] + [col for col in all_columns if col != 'ID']
    df_master = df_master[ordered_columns]

    # ======================================================================
    # 4. VERIFY MATRIX STRATIFICATION & SERIALIZE
    # ======================================================================
    print("\n=== Final Consolidated Roster Overview ===")
    print(f" -> Total Rows Accumulated: {df_master.shape[0]}")
    print(f" -> Total Feature Columns:  {df_master.shape[1]}")
    
    class_counts = df_master['class'].value_counts()
    for cls, count in class_counts.items():
        print(f"    - Class {cls}: {count} rows")
    print("==========================================\n")

    print(f"[*] Serializing index-locked master matrix to: {output_master_file}")
    df_master.to_csv(output_master_file, sep='\t', index=False)
    print("[-->] Matrix compilation completed successfully.")

if __name__ == "__main__":
    main()
