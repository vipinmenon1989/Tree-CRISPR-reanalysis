import pandas as pd
import numpy as np
import argparse

def process_mageck_negative_selection(sgrna_file, control_label, b_param=1.5, n_param=0.66, ntc_percentile=0.95):
    """
    Processes a standard MAGeCK sgRNA output for a Negative Selection screen.
    Calculates Z-scores against NTCs, applies Sigmoid, calculates a dynamic threshold, 
    and aggregates hits.
    """
    print(f"Loading MAGeCK file: {sgrna_file}")
    
    # 1. Load the MAGeCK output
    try:
        df = pd.read_csv(sgrna_file, sep='\t')
    except Exception as e:
        raise RuntimeError(f"Failed to load file: {e}")

    # 2. Isolate the Non-Targeting Controls
    ntc_mask = df['Gene'].str.contains(control_label, case=False, na=False)
    df_ntc = df[ntc_mask]
    
    if df_ntc.empty:
        raise ValueError(f"CRITICAL ERROR: No controls found matching '{control_label}'. Check your naming.")
        
    print(f"Successfully isolated {len(df_ntc)} NTC guides using prefix '{control_label}'")

    # 3. Establish the Baseline Noise
    mu_ntc = df_ntc['LFC'].mean()
    sigma_ntc = df_ntc['LFC'].std()
    print(f"NTC Baseline Noise -> Mean LFC: {mu_ntc:.4f}, Std Dev: {sigma_ntc:.4f}")

    if sigma_ntc == 0:
        raise ValueError("Standard deviation of NTCs is 0. Data might be corrupted.")

    # 4. Calculate Raw Z-Scores for all guides
    df['Z_Score'] = (df['LFC'] - mu_ntc) / sigma_ntc

    # 5. Negative Selection Inversion
    # Multiply by -1. A massive dropout (LFC -4) becomes a massive positive Z-score (+4)
    df['Inverted_Z'] = -df['Z_Score']

    # 6. Apply Sigmoid Normalization
    df['Sigmoid_Score'] = 1 / (1 + np.exp(-(df['Inverted_Z'] - b_param) / n_param))

    # --- THE CUSTOM DYNAMIC THRESHOLD ---
    # Re-isolate the NTCs now that they have Sigmoid Scores
    df_ntc_scored = df[df['Gene'].str.contains(control_label, case=False, na=False)]
    
    # Calculate the empirical threshold based on the requested percentile (default 95th)
    dynamic_threshold = df_ntc_scored['Sigmoid_Score'].quantile(ntc_percentile)
    
    print(f"Empirical Threshold Calculated: {dynamic_threshold:.4f} ({ntc_percentile*100}th Percentile of NTCs)")

    # 7. Apply the Dynamic Threshold & Record it
    df['Is_Functional'] = df['Sigmoid_Score'] > dynamic_threshold
    
    # Broadcast the exact threshold used to a new column for data provenance
    df['Dynamic_Threshold'] = dynamic_threshold 

    # Isolate targeting guides to apply the Rule of 2/3
    df_targets = df[~ntc_mask].copy()

    return df_targets, df

def aggregate_rule_of_three(df_targets):
    """
    Finds genes that have at least 3 functional guides causing dropouts.
    """
    print("\nApplying the Rule of 3 for Gene Aggregation...")
    
    functional_guides = df_targets[df_targets['Is_Functional']]
    gene_counts = functional_guides.groupby('Gene').size().reset_index(name='Functional_Hit_Count')
    
    # Updated to require >= 3 guides
    df_valid_genes = gene_counts[gene_counts['Functional_Hit_Count'] >= 3]
    
    print(f"Total Target Genes in Screen: {df_targets['Gene'].nunique()}")
    print(f"Genes passing the Rule of 3 threshold: {len(df_valid_genes)}")
    
    return df_valid_genes

if __name__ == "__main__":
    # --- CONFIGURATION ---
    FILE_PATH = "output.d14_ALKM_H_dmso_R1,d14_ALKM_H_dmso_R2_vs_d0_ALKM_H_R1,d0_ALKM_H_R2.sgrna_summary.txt" 
    CONTROL_LABEL = "NO-TARGET"  
    
    # Sigmoid & Threshold Parameters
    B_SHIFT = 1.5  
    N_SLOPE = 0.66  
    NTC_PERCENTILE = 0.95 
    
    # --- EXECUTION ---
    df_targets, df_full_scored = process_mageck_negative_selection(
        FILE_PATH, CONTROL_LABEL, B_SHIFT, N_SLOPE, NTC_PERCENTILE
    )
    
    # Calling the updated function
    df_valid_genes = aggregate_rule_of_three(df_targets)
    
    # --- SAVE ---
    df_full_scored.to_csv("Scored_sgRNA_Matrix.csv", index=False)
    df_valid_genes.to_csv("Negative_Selection_Hits_RuleOf3.csv", index=False)
    
    print("\nPipeline Complete. Outputs saved.")
