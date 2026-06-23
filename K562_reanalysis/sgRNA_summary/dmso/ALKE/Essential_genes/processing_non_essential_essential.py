import os
import sys
import pandas as pd
import numpy as np
import argparse

def run_crispr_preprocessing_pipeline(input_mageck, input_essential, control_keyword="NO-TARGET", z_threshold=1.5):
    print("======================================================================")
    print("CRISPRi PIPELINE EXECUTION ENGINE: DYNAMIC COEF SOLVER & BALANCING")
    print("======================================================================\n")
    
    # ---- STEP 1: DATA INGESTION ----
    try:
        df = pd.read_csv(input_mageck, sep='\t')
        print(f"[*] Ingested raw MAGeCK file: {len(df)} total rows.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to read MAGeCK input file: {e}")
        sys.exit(1)
        
    try:
        ess_df = pd.read_csv(input_essential, sep='\t')
        essential_list = ess_df['Gene'].dropna().unique() if 'Gene' in ess_df.columns else ess_df.iloc[:, 0].dropna().unique()
        print(f"[*] Ingested essential gene filter: {len(essential_list)} unique genes loaded.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to read essential genes file: {e}")
        sys.exit(1)

    # Automatically identify the sgRNA column regardless of raw casing
    sgrna_candidates = [c for c in df.columns if c.lower() == 'sgrna']
    sgrna_col_raw = sgrna_candidates[0] if sgrna_candidates else df.columns[0]
    print(f"[*] Identified sgRNA identifier column from input: '{sgrna_col_raw}'")

    # ---- STEP 2: DYNAMIC ALGEBRAIC COEFFICIENT SOLVER ----
    print(f"[*] Target Boundary Profile: S({z_threshold}) = 0.25 and S(-2.0) = 0.005")
    n_param = 4.194713 / (2.0 + z_threshold)
    b_param = 5.293305 - (2.0 * n_param)
    print(f"[*] Calculated Sigmoid Parameters -> Calculated n: {n_param:.4f}, Calculated b: {b_param:.4f}")

    # ---- STEP 3: GLOBAL BASELINE CALIBRATION ----
    ntc_mask = df['Gene'].str.contains(control_keyword, case=False, na=False)
    df_ntc = df[ntc_mask].copy()
    
    if df_ntc.empty:
        print(f"CRITICAL ERROR: Zero control guides found matching prefix '{control_keyword}'.")
        sys.exit(1)
        
    mu_ntc = df_ntc['LFC'].mean()
    sigma_ntc = df_ntc['LFC'].std()
    print(f"[*] Global NTC Background -> Mean LFC: {mu_ntc:.4f}, Std Dev: {sigma_ntc:.4f}")
    
    df['Z_Score'] = (df['LFC'] - mu_ntc) / sigma_ntc
    df['Inverted_Z'] = -df['Z_Score']
    df['Sigmoid_Score'] = 1 / (1 + np.exp(-n_param * df['Inverted_Z'] + b_param))

    # ---- STEP 4: FILE 1 GENERATION - HIGH CONFIDENCE ESSENTIAL HITS ----
    print("\n[Processing File 1: High-Confidence Essential Hits Matrix]")
    df_targets = df[~ntc_mask].copy()
    df_essential_all = df_targets[df_targets['Gene'].isin(essential_list)].copy()
    
    essential_hits_raw = df_essential_all[df_essential_all['Sigmoid_Score'] > 0.25].copy()
    hit_counts_per_gene = essential_hits_raw.groupby('Gene').size()
    genes_passing_sg_cutoff = hit_counts_per_gene[hit_counts_per_gene >= 2].index
    
    file_1_df = essential_hits_raw[essential_hits_raw['Gene'].isin(genes_passing_sg_cutoff)].copy()
    file_1_path = "Output_File1_Essential_Hits.csv"
    file_1_df.to_csv(file_1_path, index=False)
    print(f"--> SUCCESS: Saved File 1 ({len(file_1_df)} rows) targeting {len(genes_passing_sg_cutoff)} verified genes.")
    
    # ---- STEP 5: FILE 2 GENERATION - VISUALIZATION DISTRIBUTION ----
    print("\n[Processing File 2: NTC + Essential Hits Density Visualization Matrix]")
    df_ntc_tagged = df[ntc_mask].copy()
    df_ntc_tagged['Visualization_Group'] = 'NTC'
    
    file_1_tagged = file_1_df.copy()
    file_1_tagged['Visualization_Group'] = 'Essential_Hit'
    
    file_2_df = pd.concat([df_ntc_tagged, file_1_tagged], axis=0).reset_index(drop=True)
    file_2_path = "Output_File2_Density_Distribution.csv"
    file_2_df.to_csv(file_2_path, index=False)
    print(f"--> SUCCESS: Saved File 2 ({len(file_2_df)} rows total). Breakdown: {len(df_ntc_tagged)} NTCs | {len(file_1_tagged)} Essential Hits.")
    
    # ---- STEP 6: FILE 3 GENERATION - BALANCED 4-COLUMN ML MATRIX ----
    print("\n[Processing File 3: Pure Isolated Balanced ML Training Dataset]")
    ml_base_pool = df_essential_all[df_essential_all['Gene'].isin(genes_passing_sg_cutoff)].copy()
    
    class_1_pool = ml_base_pool[ml_base_pool['Sigmoid_Score'] > 0.25].copy()
    class_1_pool['class'] = 1
    
    class_0_pool = ml_base_pool[ml_base_pool['Sigmoid_Score'] <= 0.1].copy()
    class_0_pool['class'] = 0
    
    target_sample_size = len(class_1_pool)
    print(f"[*] Subsetting Layout -> Class 1 (Hits): {target_sample_size} rows | Class 0 (Failures Available): {len(class_0_pool)} rows.")
    
    if len(class_0_pool) >= target_sample_size:
        class_0_balanced = class_0_pool.sample(n=target_sample_size, random_state=42)
    else:
        print("[!] WARNING: Class 0 pool is smaller than Class 1. Using all available negative rows.")
        class_0_balanced = class_0_pool
        
    file_3_df = pd.concat([class_1_pool, class_0_balanced], axis=0)
    file_3_df = file_3_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    file_3_df = file_3_df.rename(columns={sgrna_col_raw: 'sgrna', 'Sigmoid_Score': 'Sigmoid_score'})
    file_3_df = file_3_df[['sgrna', 'Gene', 'Sigmoid_score', 'class']]
    
    file_3_path = "Output_File3_Balanced_ML_Matrix.csv"
    file_3_df.to_csv(file_3_path, index=False)
    print(f"--> SUCCESS: Saved File 3 ({len(file_3_df)} rows total). Columns locked to: {list(file_3_df.columns)}")
    
    # ---- STEP 7: FILE 4 GENERATION - TRIPLE COHORT DENSITY MATRIX (NTC vs ESSENTIAL HITS vs NON-ESSENTIAL HITS) ----
    print("\n[Processing File 4: Triple Cohort Density Distribution Matrix]")
    
    # Isolate all non-essential target rows from the matrix
    df_non_essential_all = df_targets[~df_targets['Gene'].isin(essential_list)].copy()
    
    # Isolate active non-essential guides using the exact same phenotypic threshold
    non_essential_hits_raw = df_non_essential_all[df_non_essential_all['Sigmoid_Score'] > 0.25].copy()
    
    # Group by non-essential gene identifiers to apply the strict sgRNA volume filter
    hit_counts_per_non_ess = non_essential_hits_raw.groupby('Gene').size()
    non_ess_genes_passing_cutoff = hit_counts_per_non_ess[hit_counts_per_non_ess >= 2].index
    
    # Build clean dataframe of high-confidence non-essential phenotypic hits
    file_4_non_ess_df = non_essential_hits_raw[non_essential_hits_raw['Gene'].isin(non_ess_genes_passing_cutoff)].copy()
    
    # Assign specific cohort tracking tags
    df_ntc_f4 = df_ntc.copy()
    df_ntc_f4['Visualization_Group'] = 'NTC'
    
    file_1_f4 = file_1_df.copy()
    file_1_f4['Visualization_Group'] = 'Essential_Hit'
    
    file_4_non_ess_tagged = file_4_non_ess_df.copy()
    file_4_non_ess_tagged['Visualization_Group'] = 'Non_Essential_Hit'
    
    # Consolidate cohorts into a single visualization file
    file_4_df = pd.concat([df_ntc_f4, file_1_f4, file_4_non_ess_tagged], axis=0).reset_index(drop=True)
    file_4_path = "Output_File4_Density_Three_Groups.csv"
    file_4_df.to_csv(file_4_path, index=False)
    
    print(f"--> SUCCESS: Saved File 4 ({len(file_4_df)} rows total) to {file_4_path}")
    print(f"    |-- NTC Baseline Rows: {len(df_ntc_f4)}")
    print(f"    |-- High-Confidence Essential Hits: {len(file_1_f4)} (from {len(genes_passing_sg_cutoff)} genes)")
    print(f"    |-- High-Confidence Non-Essential Hits: {len(file_4_non_ess_tagged)} (from {len(non_ess_genes_passing_cutoff)} genes)")
    
    print("\n======================================================================")
    print("ANALYSIS PREPROCESSING COMPLETE. ALL 4 PIPELINE OUTPUTS SECURED.")
    print("======================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRISPR Preprocessing Engine")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-e", "--essential", required=True)
    parser.add_argument("-c", "--control", default="NO-TARGET")
    parser.add_argument("-z", "--z_threshold", type=float, default=1.5)
    args = parser.parse_args()
    
    run_crispr_preprocessing_pipeline(args.input, args.essential, args.control, args.z_threshold)
