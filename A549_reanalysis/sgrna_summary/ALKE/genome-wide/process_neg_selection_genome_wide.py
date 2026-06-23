import os
import sys
import pandas as pd
import numpy as np
import argparse

def run_genome_wide_preprocessing_pipeline(input_mageck, control_keyword="NO-TARGET", z_threshold=1.5):
    print("======================================================================")
    print("CRISPRi PIPELINE EXECUTION ENGINE: GENOME-WIDE ACTIVE GENE SOLVER")
    print("======================================================================\n")
    
    # ---- STEP 1: DATA INGESTION ----
    try:
        df = pd.read_csv(input_mageck, sep='\t')
        print(f"[*] Ingested raw MAGeCK file: {len(df)} total rows.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to read MAGeCK input file: {e}")
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
    
    # Apply score transformations across the entire dataset
    df['Z_Score'] = (df['LFC'] - mu_ntc) / sigma_ntc
    df['Inverted_Z'] = -df['Z_Score']
    df['Sigmoid_Score'] = 1 / (1 + np.exp(-n_param * df['Inverted_Z'] + b_param))

    # ---- STEP 4: FILE 1 GENERATION - ALL HIGH-CONFIDENCE PHENOTYPIC HITS ----
    print("\n[Processing File 1: High-Confidence Genome-Wide Hits Matrix]")
    df_targets = df[~ntc_mask].copy()
    
    # Isolate all targeting guides clearing the functional activity threshold
    all_hits_raw = df_targets[df_targets['Sigmoid_Score'] > 0.25].copy()
    
    # Group by Gene across the entire genome to find lines containing >= 2 active guides
    hit_counts_per_gene = all_hits_raw.groupby('Gene').size()
    genes_passing_sg_cutoff = hit_counts_per_gene[hit_counts_per_gene >= 2].index
    
    # Build your high-confidence phenotypic hit dataframe
    file_1_df = all_hits_raw[all_hits_raw['Gene'].isin(genes_passing_sg_cutoff)].copy()
    file_1_path = "Output_File1_All_Phenotypic_Hits.csv"
    file_1_df.to_csv(file_1_path, index=False)
    print(f"--> SUCCESS: Saved File 1 ({len(file_1_df)} rows) targeting {len(genes_passing_sg_cutoff)} verified genome-wide genes.")
    
    # ---- STEP 5: FILE 2 GENERATION - VISUALIZATION DISTRIBUTION ----
    print("\n[Processing File 2: NTC + Genome Hits Density Visualization Matrix]")
    df_ntc_tagged = df[ntc_mask].copy()
    df_ntc_tagged['Visualization_Group'] = 'NTC'
    
    file_1_tagged = file_1_df.copy()
    file_1_tagged['Visualization_Group'] = 'Genome_Wide_Hit'
    
    file_2_df = pd.concat([df_ntc_tagged, file_1_tagged], axis=0).reset_index(drop=True)
    file_2_path = "Output_File2_Density_Distribution.csv"
    file_2_df.to_csv(file_2_path, index=False)
    print(f"--> SUCCESS: Saved File 2 ({len(file_2_df)} rows total). Breakdown: {len(df_ntc_tagged)} NTCs | {len(file_1_tagged)} Phenotypic Hits.")
    
    # ---- STEP 6: FILE 3 GENERATION - BALANCED 4-COLUMN ML TRAINING MATRIX ----
    print("\n[Processing File 3: Genome-Wide Balanced ML Training Dataset]")
    
    # Pool ALL guides belonging to the verified active genes pool to protect sequence features
    ml_base_pool = df_targets[df_targets['Gene'].isin(genes_passing_sg_cutoff)].copy()
    
    class_1_pool = ml_base_pool[ml_base_pool['Sigmoid_Score'] > 0.25].copy()
    class_1_pool['class'] = 1
    
    # Strict dead-zone separation gate (Score <= 0.1) for true structural/physical failures
    class_0_pool = ml_base_pool[ml_base_pool['Sigmoid_Score'] <= 0.05].copy()
    class_0_pool['class'] = 0
    
    target_sample_size = len(class_1_pool)
    print(f"[*] Subsetting Layout -> Class 1 (Hits): {target_sample_size} rows | Class 0 (Failures Available): {len(class_0_pool)} rows.")
    
    if len(class_0_pool) >= target_sample_size:
        class_0_balanced = class_0_pool.sample(n=target_sample_size, random_state=42)
    else:
        print("[!] WARNING: Class 0 pool is smaller than Class 1. Retaining all available negative rows.")
        class_0_balanced = class_0_pool
        
    file_3_df = pd.concat([class_1_pool, class_0_balanced], axis=0)
    file_3_df = file_3_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Normalize casings and slice strictly to your requested 4-column schema
    file_3_df = file_3_df.rename(columns={sgrna_col_raw: 'sgrna', 'Sigmoid_Score': 'Sigmoid_score'})
    file_3_df = file_3_df[['sgrna', 'Gene', 'Sigmoid_score', 'class']]
    
    file_3_path = "Output_File3_Balanced_ML_Matrix.csv"
    file_3_df.to_csv(file_3_path, index=False)
    print(f"--> SUCCESS: Saved File 3 ({len(file_3_df)} rows total). Columns locked to: {list(file_3_df.columns)}")
    print("\n======================================================================")
    print("ANALYSIS PREPROCESSING COMPLETE. WHOLE-GENOME OUTPUTS SECURED.")
    print("======================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genome-Wide CRISPR Preprocessing Engine")
    parser.add_argument("-i", "--input", required=True, help="Raw MAGeCK input dataset")
    parser.add_argument("-c", "--control", default="NO-TARGET", help="Control guide string prefix")
    parser.add_argument("-z", "--z_threshold", type=float, default=1.5, help="Z-score cutoff threshold")
    args = parser.parse_args()
    
    run_genome_wide_preprocessing_pipeline(args.input, args.control, args.z_threshold)
