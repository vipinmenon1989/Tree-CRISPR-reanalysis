import pandas as pd
import numpy as np
import argparse
import sys

def sigmoid(z, n=1.5, b=0):
    """
    Transforms Z-scores to a 0.0 - 1.0 range.
    """
    return 1 / (1 + np.exp(-n * (z - b)))

def main():
    parser = argparse.ArgumentParser(description='CRISPRi Phase 2: File 1 Generation (ReLU Clipping)')
    parser.add_argument('--sgrna_dmso', required=True, help='Path to sgrna_summary.txt (DMSO)')
    parser.add_argument('--sgrna_rig', required=True, help='Path to sgrna_summary.txt (RIG)')
    parser.add_argument('--out', default='Global_Sigmoid_Scores.txt', help='Output filename')
    args = parser.parse_args()

    # 1. Load Data
    try:
        df_dmso = pd.read_csv(args.sgrna_dmso, sep='\t')
        df_rig = pd.read_csv(args.sgrna_rig, sep='\t')
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    # 2. Merge sgrna files
    df_dmso = df_dmso[['sgrna', 'Gene', 'LFC']].rename(columns={'LFC': 'LFC_DMSO'})
    df_rig = df_rig[['sgrna', 'LFC']].rename(columns={'LFC': 'LFC_RIG'})
    df = pd.merge(df_dmso, df_rig, on='sgrna')

    # 3. Robust NTC Calibration (NO-TARGET-*)
    ntc_mask = df['Gene'].str.contains('^NO-TARGET', na=False, case=False, regex=True)
    
    if ntc_mask.sum() == 0:
        print("ERROR: No NTCs found matching 'NO-TARGET'.")
        sys.exit(1)
        
    ntc_lfc = df[ntc_mask]['LFC_DMSO']
    mu, sigma = ntc_lfc.mean(), ntc_lfc.std()
    
    if sigma == 0:
        print("ERROR: NTC variance is zero.")
        sys.exit(1)

    # 4. Standardized Z-scores
    df['Z_DMSO'] = (df['LFC_DMSO'] - mu) / sigma
    df['Z_RIG'] = (df['LFC_RIG'] - mu) / sigma

    # 5. Sigmoid Transformation & Delta
    df['Score_DMSO'] = sigmoid(df['Z_DMSO'])
    df['Score_RIG'] = sigmoid(df['Z_RIG'])
    
    # ---------------------------------------------------------
    # ReLU Logic: Clipping to zero for final integration of data
    # Ensures only positive shifts (resistance) are treated as signal.
    # Anything below 0 (noise or baseline drift) is zeroed out.
    # ---------------------------------------------------------
    df['Delta_Score_Raw'] = df['Score_RIG'] - df['Score_DMSO']
    df['Delta_Score'] = df['Delta_Score_Raw'].clip(lower=0)
    # ---------------------------------------------------------

    # 6. Save File 1
    # We include Delta_Score (clipped) as the primary output for ML
    output_cols = [
        'sgrna', 'Gene', 'LFC_DMSO', 'LFC_RIG', 
        'Score_DMSO', 'Score_RIG', 'Delta_Score'
    ]
    
    df[output_cols].to_csv(args.out, sep='\t', index=False)

    print("-" * 40)
    print(f"FILE 1 GENERATED (ReLU Applied)")
    print(f"NTC Stats: Mean={mu:.4f}, Std={sigma:.4f}")
    print(f"Delta_Score range: {df['Delta_Score'].min()} to {df['Delta_Score'].max()}")
    print(f"Output: {args.out}")
    print("-" * 40)

if __name__ == "__main__":
    main()