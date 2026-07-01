import pandas as pd
import numpy as np

def clean_scale_and_merge_cell_lines(file1_path, file2_path, file3_path, output_path="merged_alke_dataset.csv"):
    print("======================================================================")
    print("[*] RE-ENGINEERING EXPERIMENTAL DATASETS: RANK-SCALED FEATURE UNIFICATION")
    print("======================================================================\n")

    # ------------------------------------------------------------------------
    # STEP 1: LOAD DATASETS & TRACK INITIAL DIMENSIONS
    # ------------------------------------------------------------------------
    df1 = pd.read_csv(file1_path, sep=None, engine='python')
    df2 = pd.read_csv(file2_path, sep=None, engine='python')
    df3 = pd.read_csv(file3_path, sep=None, engine='python')

    print("[*] STAGE 1: RAW INGESTION DIMENSIONS")
    print(f"  -> File 1 (A375) Input: {df1.shape[0]} rows x {df1.shape[1]} columns")
    print(f"  -> File 2 (A549) Input: {df2.shape[0]} rows x {df2.shape[1]} columns")
    print(f"  -> File 3 (K562) Input: {df3.shape[0]} rows x {df3.shape[1]} columns\n")

    if 'ID' in df3.columns:
        df3 = df3.drop(columns=['ID'])
        print("[!] Dropped 'ID' column from File 3.")
    elif 'id' in df3.columns:
        df3 = df3.drop(columns=['id'])
        print("[!] Dropped 'id' column from File 3.")

    # ------------------------------------------------------------------------
    # STEP 2: STAGE BEFORE-NORMALIZATION COLUMN COMPARISON
    # ------------------------------------------------------------------------
    cols1_raw = set(df1.columns)
    cols2_raw = set(df2.columns)
    cols3_raw = set(df3.columns)
    
    raw_intersection = cols1_raw.intersection(cols2_raw).intersection(cols3_raw)
    print("\n[*] STAGE 2: COLUMN ALIGNMENT (PRE-CLEANING)")
    print(f"  -> Identical column names shared across all 3 files: {len(raw_intersection)}")

    # ------------------------------------------------------------------------
    # STEP 3: COLUMN NAME NORMALIZATION FUNCTION
    # ------------------------------------------------------------------------
    def normalize_headers(df):
        new_cols = []
        for col in df.columns:
            c = col.strip()
            if c.lower() in ['gene', 'mean_sigmoid_score', 'class']:
                new_cols.append(c.lower())
                continue
                
            for prefix in ['A375_', 'A549_', 'K562_']:
                if c.startswith(prefix):
                    c = c[len(prefix):]
                    break
            
            c = c.lower()
            if 'dna_methylation_coverage_bin_' in c:
                c = c.replace('dna_methylation_coverage_bin_', 'dna_methylation_bin_')
                
            new_cols.append(c)
        return new_cols

    df1.columns = normalize_headers(df1)
    df2.columns = normalize_headers(df2)
    df3.columns = normalize_headers(df3)

    # ------------------------------------------------------------------------
    # STEP 4: STAGE AFTER-NORMALIZATION COLUMN COMPARISON
    # ------------------------------------------------------------------------
    cols1_clean = set(df1.columns)
    cols2_clean = set(df2.columns)
    cols3_clean = set(df3.columns)
    
    clean_intersection = cols1_clean.intersection(cols2_clean).intersection(cols3_clean)
    print("\n[*] STAGE 3: COLUMN ALIGNMENT (POST-CLEANING)")
    print(f"  -> File 1 Clean Columns: {len(cols1_clean)}")
    print(f"  -> File 2 Clean Columns: {len(cols2_clean)}")
    print(f"  -> File 3 Clean Columns: {len(cols3_clean)}")
    print(f"  -> Identical column names aligned for merge: {len(clean_intersection)}")

    if len(clean_intersection) != len(cols1_clean) or len(clean_intersection) != len(cols2_clean) or len(clean_intersection) != len(cols3_clean):
        print("\n[!] CRITICAL WARNING: Schema alignment failed. Features dropped from intersection.")
    else:
        print("  -> [+] SUCCESS: Perfect 1-to-1 feature schema synchronization achieved.")

    # ------------------------------------------------------------------------
    # STEP 4.5: PERCENTILE RANK SCALING (INTRA-DATASET)
    # ------------------------------------------------------------------------
    print("\n[*] STAGE 4: INDEPENDENT PERCENTILE RANK SCALING")
    metadata_keys = ['gene', 'mean_sigmoid_score', 'class']
    
    # Isolate only the true epigenetic spatial columns for scaling
    feature_cols = [c for c in clean_intersection if c not in metadata_keys]
    
    # Scale each dataframe independently to prevent batch effect leakage
    df1[feature_cols] = df1[feature_cols].rank(pct=True)
    print(f"  -> File 1 (A375) spatial features mathematically scaled (0.0 to 1.0).")
    
    df2[feature_cols] = df2[feature_cols].rank(pct=True)
    print(f"  -> File 2 (A549) spatial features mathematically scaled (0.0 to 1.0).")
    
    df3[feature_cols] = df3[feature_cols].rank(pct=True)
    print(f"  -> File 3 (K562) spatial features mathematically scaled (0.0 to 1.0).")

    # ------------------------------------------------------------------------
    # STEP 5: APPEND DATASETS AND GENERATE SEQUENTIAL METADATA ID
    # ------------------------------------------------------------------------
    ordered_cols = sorted(list(clean_intersection))
    final_column_order = metadata_keys + [c for c in ordered_cols if c not in metadata_keys]

    merged_df = pd.concat([
        df1[final_column_order], 
        df2[final_column_order], 
        df3[final_column_order]
    ], axis=0, ignore_index=True)

    # Inject Sequential ID starting at index 1
    merged_df.insert(0, 'ID', [f"Gene_{i}" for i in range(1, len(merged_df) + 1)])

    merged_df.to_csv(output_path, index=False)
    
    print("\n======================================================================")
    print("[*] FINAL OUTPUT VERIFICATION")
    print("======================================================================")
    expected_rows = df1.shape[0] + df2.shape[0] + df3.shape[0]
    expected_cols = len(clean_intersection) + 1 
    
    print(f"  -> Expected Dimensions: {expected_rows} rows x {expected_cols} columns")
    print(f"  -> Actual Dimensions:   {merged_df.shape[0]} rows x {merged_df.shape[1]} columns")
    
    if merged_df.shape[0] == expected_rows and merged_df.shape[1] == expected_cols:
         print("  -> [+] Integrity Check: PASSED. No rows or aligned columns lost during merge.")
    else:
         print("  -> [!] Integrity Check: FAILED. Dimensions do not match mathematical expectation.")
         
    print(f"\n[+] Merged matrix successfully written to disk: {output_path}")

if __name__ == "__main__":
    clean_scale_and_merge_cell_lines(
        "K_training_ready_matrix_A375.txt", 
        "K_training_ready_matrix_A549.txt", 
        "K_training_ready_matrix_K562.txt"
    )
