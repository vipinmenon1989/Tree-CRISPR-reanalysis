import pandas as pd
import numpy as np
import argparse
import sys

def sigmoid(z, b=1.5, n=0.66):
    """
    Standardized Sigmoid for Resistance Screens.
    Midpoint (b) = 1.5 ensures noise suppression (93% NTC exclusion).
    Slope (n) = 0.66 preserves behavioral gradients for K-Means.
    """
    # Formula: 1 / (1 + exp(-(Z - b)/n))
    return 1 / (1 + np.exp(-(z - b) / n))

def main():
    parser = argparse.ArgumentParser(description='CRISPRi Phase 2: sgRNA Standardization')
    parser.add_argument('--sgrna_dmso', required=True, help='Path to sgrna_summary.txt (DMSO)')
    parser.add_argument('--sgrna_rig', required=True, help='Path to sgrna_summary.txt (RIG)')
    parser.add_argument('--out', default='sgRNA_Standardized_Scores.txt', help='Output filename')
    args = parser.parse_args()

    # 1. Load Data
    try:
        df_dmso = pd.read_csv(args.sgrna_dmso, sep='\t')
        df_rig = pd.read_csv(args.sgrna_rig, sep='\t')
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    # 2. Extract LFCs and Merge
    df_dmso = df_dmso[['sgrna', 'Gene', 'LFC']].rename(columns={'LFC': 'LFC_DMSO'})
    df_rig = df_rig[['sgrna', 'LFC']].rename(columns={'LFC': 'LFC_RIG'})
    df = pd.merge(df_dmso, df_rig, on='sgrna')

    # 3. NTC Calibration using Vehicle (DMSO) as background
    ntc_mask = df['Gene'].str.contains('^NO-TARGET', na=False, case=False, regex=True)
    
    if ntc_mask.sum() == 0:
        print("ERROR: No NTCs found matching 'NO-TARGET'.")
        sys.exit(1)
        
    ntc_lfc = df[ntc_mask]['LFC_DMSO']
    mu_ntc, sigma_ntc = ntc_lfc.mean(), ntc_lfc.std()

    # 4. Standardized Z-scores (Calibrated to DMSO background)
    df['Z_DMSO'] = (df['LFC_DMSO'] - mu_ntc) / sigma_ntc
    df['Z_RIG'] = (df['LFC_RIG'] - mu_ntc) / sigma_ntc

    # 5. Sigmoid Transformation (b=1.5, n=0.66)
    df['Score_DMSO'] = sigmoid(df['Z_DMSO'])
    df['Score_RIG'] = sigmoid(df['Z_RIG'])
    
    # 6. Gain of Resistance Calculation (Delta Score)
    # This isolates the drug-specific resistance signal
    df['Delta_Score_Raw'] = df['Score_RIG'] - df['Score_DMSO']
    df['Delta_Score'] = df['Delta_Score_Raw'].clip(lower=0) # ReLU Clipping

    # Save Output
    output_cols = [
        'sgrna', 'Gene', 'LFC_DMSO', 'LFC_RIG', 
        'Score_DMSO', 'Score_RIG', 'Delta_Score'
    ]
    df[output_cols].to_csv(args.out, sep='\t', index=False)

    print("-" * 40)
    print(f"sgRNA SCORES GENERATED")
    print(f"NTC Stats (DMSO): Mean={mu_ntc:.4f}, Std={sigma_ntc:.4f}")
    print(f"Unified Params: b=1.5, n=0.66")
    print("-" * 40)

if __name__ == "__main__":
    main()



