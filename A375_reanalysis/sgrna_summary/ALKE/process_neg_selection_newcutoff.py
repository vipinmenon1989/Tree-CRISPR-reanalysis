import pandas as pd
import numpy as np
import argparse
import sys

def process_mageck_negative_selection(sgrna_file, control_label, n_param=1.0, b_param=2.0, ntc_percentile=0.95):
    """
    Processes MAGeCK sgRNA output for Negative Selection using the CRE paper formula:
    S(Z) = 1 / (1 + exp(-n*Z + b))
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
        print(f"CRITICAL ERROR: No controls found matching '{control_label}'. Check your naming.")
        sys.exit(1)
        
    # 2. Baseline Noise Calculation
    mu_ntc = df_ntc['LFC'].mean()
    sigma_ntc = df_ntc['LFC'].std()
    print(f"NTC Baseline -> Mean LFC: {mu_ntc:.4f}, Std Dev: {sigma_ntc:.4f}")

    # 3. Calculate Inverted Z-Score for Negative Selection
    # High negative LFC (dropout) becomes high Inverted Z-Score
    df['Z_Score'] = (df['LFC'] - mu_ntc) / sigma_ntc
    df['Inverted_Z'] = -df['Z_Score']
    
    # 4. Apply Official CRE Formula: S(Z) = 1 / (1 + exp(-n*Z + b))
    # We use Inverted_Z as the input variable Z
    df['Sigmoid_Score'] = 1 / (1 + np.exp(-n_param * df['Inverted_Z'] + b_param))

    # 5. Dynamic Threshold Calculation
    dynamic_threshold = df.loc[ntc_mask, 'Sigmoid_Score'].quantile(ntc_percentile)
    print(f"Empirical Threshold: {dynamic_threshold:.4f} ({ntc_percentile*100}th Percentile)")

    # 6. Apply Threshold
    df['Is_Functional'] = df['Sigmoid_Score'] > dynamic_threshold
    df['Dynamic_Threshold'] = dynamic_threshold 

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
    parser = argparse.ArgumentParser(description="CRE Paper Math Pipeline")
    
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-c", "--control", default="NO-TARGET")
    parser.add_argument("-p", "--percentile", type=float, default=0.95)
    # Defaulting to the benchmark parameters derived from the CRE screenshot
    parser.add_argument("-b", "--b_shift", type=float, default=2.0)
    parser.add_argument("-n", "--n_slope", type=float, default=1.0)
    parser.add_argument("-o", "--output", default="Scored_sgRNA_Matrix_new_cutoff.csv")
    parser.add_argument("-g", "--gene_output", default="Negative_Selection_Hits_RuleOf3_new_cutoff.csv")

    args = parser.parse_args()

    df_targets, df_full = process_mageck_negative_selection(
        args.input, args.control, args.n_slope, args.b_shift, args.percentile
    )
    
    df_valid_genes = aggregate_rule_of_three(df_targets)
    
    df_full.to_csv(args.output, index=False)
    df_valid_genes.to_csv(args.gene_output, index=False)
    
    print(f"\nPipeline Complete. Saved to {args.output} and {args.gene_output}")


