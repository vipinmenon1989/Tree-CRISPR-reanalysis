import pandas as pd
import numpy as np
import argparse
import sys

def process_mageck_negative_selection(sgrna_file, control_label, b_param=1.5, n_param=0.66, ntc_percentile=0.95):
    """
    Processes a standard MAGeCK sgRNA output for a Negative Selection screen.
    """
    print(f"Loading MAGeCK file: {sgrna_file}")
    
    try:
        df = pd.read_csv(sgrna_file, sep='\t')
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load file: {e}")
        sys.exit(1)

    # 1. Isolate the Non-Targeting Controls
    ntc_mask = df['Gene'].str.contains(control_label, case=False, na=False)
    df_ntc = df[ntc_mask]
    
    if df_ntc.empty:
        print(f"CRITICAL ERROR: No controls found matching '{control_label}'. Check your naming.")
        sys.exit(1)
        
    print(f"Successfully isolated {len(df_ntc)} NTC guides using prefix '{control_label}'")

    # 2. Establish the Baseline Noise
    mu_ntc = df_ntc['LFC'].mean()
    sigma_ntc = df_ntc['LFC'].std()
    print(f"NTC Baseline Noise -> Mean LFC: {mu_ntc:.4f}, Std Dev: {sigma_ntc:.4f}")

    if sigma_ntc == 0:
        print("CRITICAL ERROR: Standard deviation of NTCs is 0. Data might be corrupted.")
        sys.exit(1)

    # 3. Calculate Normalized Scores
    df['Z_Score'] = (df['LFC'] - mu_ntc) / sigma_ntc
    df['Inverted_Z'] = -df['Z_Score']
    df['Sigmoid_Score'] = 1 / (1 + np.exp(-(df['Inverted_Z'] - b_param) / n_param))

    # 4. Dynamic Threshold Calculation
    df_ntc_scored = df[ntc_mask]
    dynamic_threshold = df_ntc_scored['Sigmoid_Score'].quantile(ntc_percentile)
    
    print(f"Empirical Threshold Calculated: {dynamic_threshold:.4f} ({ntc_percentile*100}th Percentile of NTCs)")

    # 5. Apply Threshold
    df['Is_Functional'] = df['Sigmoid_Score'] > dynamic_threshold
    df['Dynamic_Threshold'] = dynamic_threshold 

    df_targets = df[~ntc_mask].copy()
    return df_targets, df

def aggregate_rule_of_three(df_targets):
    """
    Finds genes that have at least 3 functional guides causing dropouts.
    """
    print("\nApplying the Rule of 3 for Gene Aggregation...")
    
    functional_guides = df_targets[df_targets['Is_Functional']]
    gene_counts = functional_guides.groupby('Gene').size().reset_index(name='Functional_Hit_Count')
    
    df_valid_genes = gene_counts[gene_counts['Functional_Hit_Count'] >= 3]
    
    print(f"Total Target Genes in Screen: {df_targets['Gene'].nunique()}")
    print(f"Genes passing the Rule of 3 threshold: {len(df_valid_genes)}")
    
    return df_valid_genes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MAGeCK sgRNA output with Sigmoid Normalization and Rule of 3.")
    
    # Required Arguments
    parser.add_argument("-i", "--input", required=True, help="Path to the MAGeCK sgrna_summary.txt file")
    
    # Optional Arguments with Defaults
    parser.add_argument("-c", "--control", default="NO-TARGET", help="Label for NTC guides (default: NO-TARGET)")
    parser.add_argument("-p", "--percentile", type=float, default=0.95, help="NTC percentile for dynamic threshold (default: 0.95)")
    parser.add_argument("-b", "--b_shift", type=float, default=0, help="Sigmoid center shift (default: 0)")
    parser.add_argument("-n", "--n_slope", type=float, default=1, help="Sigmoid slope parameter (default: 1)")
    parser.add_argument("-o", "--output", default="Scored_sgRNA_Matrix.csv", help="Filename for the scored sgRNA matrix")
    parser.add_argument("-g", "--gene_output", default="Negative_Selection_Hits_RuleOf3.csv", help="Filename for the gene hits file")

    args = parser.parse_args()

    # Execution
    df_targets, df_full_scored = process_mageck_negative_selection(
        args.input, args.control, args.b_shift, args.n_slope, args.percentile
    )
    
    df_valid_genes = aggregate_rule_of_three(df_targets)
    
    # Save Outputs
    df_full_scored.to_csv(args.output, index=False)
    df_valid_genes.to_csv(args.gene_output, index=False)
    
    print(f"\nPipeline Complete. Outputs saved to {args.output} and {args.gene_output}")

