import pandas as pd
import numpy as np
import argparse
import sys

def process_mageck_negative_selection(sgrna_file, control_label, target_hit_rate=0.15):
    """
    Processes MAGeCK sgRNA output for Negative Selection using an auto-calibrated 
    Sigmoid formula to ensure data stays perfectly in tune without starvation.
    """
    print(f"Loading MAGeCK file: {sgrna_file}")
    try:
        df = pd.read_csv(sgrna_file, sep='\t')
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load file: {e}")
        sys.exit(1)

    # 1. Isolate Non-Targeting Controls
    ntc_mask = df['Gene'].str.contains(control_label, case=False, na=False)
    df_ntc = df[ntc_mask]
    
    if df_ntc.empty:
        print(f"CRITICAL ERROR: No controls found matching '{control_label}'.")
        sys.exit(1)
        
    # 2. Baseline Noise Calculation
    mu_ntc = df_ntc['LFC'].mean()
    sigma_ntc = df_ntc['LFC'].std()
    print(f"NTC Baseline -> Mean LFC: {mu_ntc:.4f}, Std Dev: {sigma_ntc:.4f}")

    # 3. Calculate Inverted Z-Score for Negative Selection
    df['Z_Score'] = (df['LFC'] - mu_ntc) / sigma_ntc
    df['Inverted_Z'] = -df['Z_Score']
    
    # 4. DYNAMIC CALIBRATION OF N AND B
    # Find the Z-score value marking the top targeted hit rate (e.g., top 15%)
    z_hit = df['Inverted_Z'].quantile(1.0 - target_hit_rate)
    
    # Solve system of equations: Background Z=0 maps to 0.01, Target Z_hit maps to 0.25
    b_calibrated = np.log((1 - 0.01) / 0.01)  # approx 4.5951
    n_calibrated = (b_calibrated - np.log((1 - 0.25) / 0.25)) / z_hit
    
    print(f"\n=== Auto-Calibration Diagnostics ===")
    print(f"Targeting top {target_hit_rate*100}% of guides as hits (Inverted_Z > {z_hit:.4f})")
    print(f"Dynamically calculated n (Slope):     {n_calibrated:.4f}")
    print(f"Dynamically calculated b (Intercept): {b_calibrated:.4f}")

    # 5. Apply Aligned Sigmoid Formula
    df['Sigmoid_Score'] = 1 / (1 + np.exp(-n_calibrated * df['Inverted_Z'] + b_calibrated))

    # 6. Set logical feature flags matching downstream ML criteria
    df['class'] = (df['Sigmoid_Score'] > 0.25).astype(int)
    df['Is_Functional'] = df['class'] == 1

    print(f"Total functional guides generated (Class 1): {df['class'].sum()}")

    df_targets = df[~ntc_mask].copy()
    return df_targets, df

def aggregate_rule_of_three(df_targets):
    """
    Finds genes that have at least 3 functional guides.
    """
    print("\nApplying the Rule of 3 for Gene Aggregation...")
    functional_guides = df_targets[df_targets['Is_Functional']]
    gene_counts = functional_guides.groupby('Gene').size().reset_index(name='Functional_Hit_Count')
    
    df_valid_genes = gene_counts[gene_counts['Functional_Hit_Count'] >= 3]
    
    print(f"Genes passing Rule of 3: {len(df_valid_genes)}")
    return df_valid_genes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-Tuning CRE Sigmoid Pipeline")
    
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-c", "--control", default="NO-TARGET")
    parser.add_argument("-r", "--hit_rate", type=float, default=0.15, 
                        help="The ratio of the dataset desired to be positive hits (default 0.15 for top 15%)")
    parser.add_argument("-o", "--output", default="Scored_sgRNA_Matrix_tuned.csv")
    parser.add_argument("-g", "--gene_output", default="Negative_Selection_Hits_RuleOf3_tuned.csv")

    args = parser.parse_args()

    df_targets, df_full = process_mageck_negative_selection(
        args.input, args.control, args.hit_rate
    )
    
    df_valid_genes = aggregate_rule_of_three(df_targets)
    
    df_full.to_csv(args.output, index=False)
    df_valid_genes.to_csv(args.gene_output, index=False)
    
    print(f"\nPipeline Complete. Saved to {args.output} and {args.gene_output}")
