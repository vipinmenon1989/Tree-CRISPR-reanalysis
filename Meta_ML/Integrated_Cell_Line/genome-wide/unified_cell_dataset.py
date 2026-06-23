import os
import sys
import pandas as pd

def harmonize_and_combine_matrices(k562_path, a549_path, a375_path, combined_output_path):
    print("======================================================================")
    print("[*] CRIPSRi TRIPLE-CELL INTEGRATION ENGINE (v2.0 - NAMING FIX)")
    print("======================================================================\n")

    # 1. Verification Block
    files = {'K562': k562_path, 'A549': a549_path, 'A375': a375_path}
    for cell, path in files.items():
        if not os.path.exists(path):
            print(f"CRITICAL ERROR: {cell} file does not exist at path: {path}")
            sys.exit(1)

    def load_and_normalize(cell_name, path):
        print(f"[*] Parsing and normalizing {cell_name} matrix...")
        sep = ',' if path.endswith('.csv') else '\t'
        df = pd.read_csv(path, sep=sep)
        
        # Lowercase headers to resolve case-sensitivity anomalies
        df.columns = [col.lower() for col in df.columns]
        
        # ====================================================================
        # IN-MEMORY REMAPPING LAYER FOR K562 NAMING CONVENTION CONFLICT
        # ====================================================================
        if cell_name == 'K562':
            print(f"    [-->] Harmonizing K562 methylation track naming conventions...")
            rename_dict = {}
            for col in df.columns:
                # Match promoter tracks: 'dna_methylation_bin_X' -> 'dna_methylation_coverage_bin_X'
                if col.startswith('dna_methylation_bin_'):
                    new_col = col.replace('dna_methylation_bin_', 'dna_methylation_coverage_bin_')
                    rename_dict[col] = new_col
                # Match guide tracks: 'guide_dna_methylation_bin_X' -> 'guide_dna_methylation_coverage_bin_X'
                elif col.startswith('guide_dna_methylation_bin_'):
                    new_col = col.replace('guide_dna_methylation_bin_', 'guide_dna_methylation_coverage_bin_')
                    rename_dict[col] = new_col
            
            if rename_dict:
                df = df.rename(columns=rename_dict)
                print(f"    [-->] Successfully remapped {len(rename_dict)} K562 tracking headers.")
        # ====================================================================
        
        df['cell_line_origin'] = cell_name
        return df

    # Ingest data layers
    df_k562 = load_and_normalize('K562', k562_path)
    df_a549 = load_and_normalize('A549', a549_path)
    df_a375 = load_and_normalize('A375', a375_path)
    
    # 2. Advanced Column Discrepancy Audit
    cols_k562 = set(df_k562.columns)
    cols_a549 = set(df_a549.columns)
    cols_a375 = set(df_a375.columns)
    
    # Shared master intersection
    shared_columns = list(cols_k562.intersection(cols_a549).intersection(cols_a375))
    
    # Identify non-overlapping columns (dropped columns)
    all_distinct_columns = cols_k562.union(cols_a549).union(cols_a375)
    dropped_columns = all_distinct_columns.difference(set(shared_columns))
    
    print("\n" + "-"*70)
    print("COLUMN DISCREPANCY & DROP AUDIT REPORT")
    print("-"*70)
    print(f" -> Total columns present across files: {len(all_distinct_columns)}")
    print(f" -> Total aligned shared columns kept:  {len(shared_columns)}")
    print(f" -> Total mismatched columns removed:   {len(dropped_columns)}")
    print("-"*70)
    
    if len(dropped_columns) > 0:
        print("\n[!] DETECTED MISMATCHED TRACKS (REMOVED FROM FINAL POOL):")
        for col in sorted(dropped_columns):
            sources = []
            if col in cols_k562: sources.append("K562")
            if col in cols_a549: sources.append("A549")
            if col in cols_a375: sources.append("A375")
            print(f"    X {col:<45} (Found only in: {', '.join(sources)})")
    else:
        print("\n--> PERFECT HEADER HARMONY: Zero columns dropped.")
    print("-"*70 + "\n")

    # Ensure mandatory machine learning hooks survived normalization
    mandatory_keys = ['id', 'sigmoid_score', 'distance_to_tss', 'class']
    for key in mandatory_keys:
        if key not in shared_columns:
            print(f"CRITICAL ERROR: Essential tracking key '{key}' was dropped or misspelled!")
            sys.exit(1)

    # 3. Strict Column Slicing and Sequence Alignment
    shared_columns.sort()  # Alphabetically locks indices so columns stack identically
    
    df_k562_aligned = df_k562[shared_columns].copy()
    df_a549_aligned = df_a549[shared_columns].copy()
    df_a375_aligned = df_a375[shared_columns].copy()

    # 4. Vertical Appending Layer
    print("[*] Merging matrices into unified DataFrame...")
    master_matrix = pd.concat(
        [df_k562_aligned, df_a549_aligned, df_a375_aligned], 
        axis=0, 
        ignore_index=True
    )
    
    # 5. Insert Unique Sequential Identifier Column
    print("[*] Generating unique sgRNA auto-increment keys...")
    master_matrix.insert(
        0, 
        'unique_sgrna_id', 
        [f"sgrna_{i}" for i in range(1, len(master_matrix) + 1)]
    )

    # 6. Export to Flat File
    output_sep = ',' if combined_output_path.endswith('.csv') else '\t'
    master_matrix.to_csv(combined_output_path, sep=output_sep, index=False)
    
    print("\n======================================================================")
    print("INTEGRATION METRICS REPORT")
    print("======================================================================")
    print(f"--> Unified Output Saved to: {combined_output_path}")
    print(f"--> Final Matrix Structure: {master_matrix.shape[0]} rows x {master_matrix.shape[1]} columns")
    print(f"    |-- K562 tracks appended: {len(df_k562_aligned)}")
    print(f"    |-- A549 tracks appended: {len(df_a549_aligned)}")
    print(f"    |-- A375 tracks appended: {len(df_a375_aligned)}")
    print("======================================================================\n")

if __name__ == "__main__":
    harmonize_and_combine_matrices(
        k562_path='CRISPR_ml_features_final_K562.csv',
        a549_path='CRISPR_ml_features_final_A375.csv',
        a375_path='CRISPR_ml_features_final_A549.csv',
        combined_output_path='Unified_GenomeWide_3CellLine_Matrix.txt'
    )
