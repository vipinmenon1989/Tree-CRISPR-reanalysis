import os
import sys
import pandas as pd

def harmonize_and_combine_matrices(k562_path, a549_path, a375_path, combined_output_path):
    print("======================================================================")
    print("[*] CRISPRi TRIPLE-CELL INTEGRATION ENGINE (v3.0 - PREFIX STRIP)")
    print("======================================================================\n")

    files = {'K562': k562_path, 'A549': a549_path, 'A375': a375_path}
    for cell, path in files.items():
        if not os.path.exists(path):
            print(f"CRITICAL ERROR: {cell} file does not exist at path: {path}")
            sys.exit(1)

    def load_and_normalize(cell_name, path):
        print(f"[*] Parsing and strips cell prefixes from {cell_name} matrix...")
        sep = ',' if path.endswith('.csv') else '\t'
        df = pd.read_csv(path, sep=sep)
        
        df.columns = [col.lower() for col in df.columns]
        
        # ====================================================================
        # GENERIC REMAPPING LAYER: STRIPPING CELL-LINE UNIQUE PREFIXES
        # ====================================================================
        # This converts 'k562_atac_seq_bin_1' and 'a375_atac_seq_bin_1' into 'atac_seq_bin_1'
        rename_dict = {}
        prefixes_to_strip = ['k562_', 'a549_', 'a375_']
        
        for col in df.columns:
            new_col = col
            
            # 1. Handle double prefix edge cases like 'guide_k562_...' -> 'guide_...'
            if col.startswith('guide_'):
                for prefix in prefixes_to_strip:
                    if col.startswith(f'guide_{prefix}'):
                        new_col = col.replace(f'guide_{prefix}', 'guide_')
                        break
            # 2. Handle standard track prefixes
            else:
                for prefix in prefixes_to_strip:
                    if col.startswith(prefix):
                        new_col = col.replace(prefix, '')
                        break
            
            # 3. Resolve the legacy coverage vs standard naming anomaly
            if 'dna_methylation_bin_' in new_col:
                new_col = new_col.replace('dna_methylation_bin_', 'dna_methylation_coverage_bin_')
                
            if new_col != col:
                rename_dict[col] = new_col
                
        if rename_dict:
            df = df.rename(columns=rename_dict)
            print(f"    [-->] Standardized {len(rename_dict)} specific column tracks.")
            
        df['cell_line_origin'] = cell_name
        return df

    # Ingest data layers
    df_k562 = load_and_normalize('K562', k562_path)
    df_a549 = load_and_normalize('A549', a549_path)
    df_a375 = load_and_normalize('A375', a375_path)
    
    # Advanced Column Discrepancy Audit
    cols_k562 = set(df_k562.columns)
    cols_a549 = set(df_a549.columns)
    cols_a375 = set(df_a375.columns)
    
    # Shared master intersection
    shared_columns = list(cols_k562.intersection(cols_a549).intersection(cols_a375))
    all_distinct_columns = cols_k562.union(cols_a549).union(cols_a375)
    dropped_columns = all_distinct_columns.difference(set(shared_columns))
    
    print("\n" + "-"*70)
    print("COLUMN DISCREPANCY & DROP AUDIT REPORT")
    print("-"*70)
    print(f" -> Total columns present across raw fields: {len(all_distinct_columns)}")
    print(f" -> Total unified aligned features kept:     {len(shared_columns)}")
    print(f" -> Total mismatched features dropped:       {len(dropped_columns)}")
    print("-"*70)
    
    if len(dropped_columns) > 0:
        print("\n[!] MISMATCHED COLUMNS PERMANENTLY DROPPED (CHECK EXTRA FEATURES):")
        for col in sorted(dropped_columns):
            sources = []
            if col in cols_k562: sources.append("K562")
            if col in cols_a549: sources.append("A549")
            if col in cols_a375: sources.append("A375")
            print(f"    X {col:<45} (Found in: {', '.join(sources)})")
    else:
        print("\n--> PERFECT HARMONIZATION SUCCESS: Zero features dropped.")
    print("-"*70 + "\n")

    # Ensure mandatory tracking features survived the prefix removal
    mandatory_keys = ['id', 'sigmoid_score', 'distance_to_tss', 'class', 'cell_line_origin']
    for key in mandatory_keys:
        if key not in shared_columns:
            print(f"CRITICAL ERROR: Essential tracking key '{key}' missing from shared matrix!")
            sys.exit(1)

    shared_columns.sort()
    
    df_k562_aligned = df_k562[shared_columns].copy()
    df_a549_aligned = df_a549[shared_columns].copy()
    df_a375_aligned = df_a375[shared_columns].copy()

    print("[*] Merging matrices horizontally over unified tracking columns...")
    master_matrix = pd.concat(
        [df_k562_aligned, df_a549_aligned, df_a375_aligned], 
        axis=0, 
        ignore_index=True
    )
    
    master_matrix.insert(
        0, 
        'unique_sgrna_id', 
        [f"sgrna_{i}" for i in range(1, len(master_matrix) + 1)]
    )

    output_sep = ',' if combined_output_path.endswith('.csv') else '\t'
    master_matrix.to_csv(combined_output_path, sep=output_sep, index=False)
    
    print("\n======================================================================")
    print("INTEGRATION METRICS REPORT")
    print("======================================================================")
    print(f"--> Unified Output Saved to: {combined_output_path}")
    print(f"--> Final Matrix Structure: {master_matrix.shape[0]} rows x {master_matrix.shape[1]} features")
    print("======================================================================\n")

if __name__ == "__main__":
    harmonize_and_combine_matrices(
        k562_path='CRISPR_ml_features_final_K562.csv',
        a549_path='CRISPR_ml_features_final_A549.csv',
        a375_path='CRISPR_ml_features_final_A375.csv',
        combined_output_path='Unified_GenomeWide_3CellLine_Matrix.txt'
    )
