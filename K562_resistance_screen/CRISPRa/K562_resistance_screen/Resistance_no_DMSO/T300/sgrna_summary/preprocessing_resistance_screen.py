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
    parser = argparse.ArgumentParser(description='CRISPRa Phase 2: File 1 Generation (Raw RIG Method)')
    parser.add_argument('--sgrna_rig', required=True, help='Path to sgrna_summary.txt (RIG)')
    parser.add_argument('--out', default='Global_Sigmoid_Scores_RIG.txt', help='Output filename')
    args = parser.parse_args()

    # 1. Load Data
    try:
        df = pd.read_csv(args.sgrna_rig, sep='\t')
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load file. {e}")
        sys.exit(1)

    # 2. Extract Columns
    # Ensures we grab exactly what we need from the MAGeCK output
    req_cols = ['sgrna', 'Gene', 'LFC']
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        print(f"CRITICAL ERROR: Missing columns {missing} in sgrna_summary file.")
        sys.exit(1)

    df = df[req_cols].rename(columns={'LFC': 'LFC_RIG'})

    # 3. Robust NTC Calibration (NO-TARGET-*)
    ntc_mask = df['Gene'].str.contains('^NO-TARGET', na=False, case=False, regex=True)
    
    if ntc_mask.sum() == 0:
        print("CRITICAL ERROR: No NTCs found matching 'NO-TARGET'.")
        sys.exit(1)
        
    ntc_lfc = df[ntc_mask]['LFC_RIG']
    mu, sigma = ntc_lfc.mean(), ntc_lfc.std()
    
    if sigma == 0:
        print("CRITICAL ERROR: NTC variance is zero. Check your MAGeCK output.")
        sys.exit(1)

    # 4. Standardized Z-scores (Anchored to NTCs)
    df['Z_RIG'] = (df['LFC_RIG'] - mu) / sigma

    # 5. Sigmoid Transformation (Final ML Target Variable)
    # This natively restricts the output to 0.0 -> 1.0. 
    # NTC baseline (Z=0) sits exactly at 0.5.
    df['Score_RIG'] = sigmoid(df['Z_RIG'])

    # 6. Save File 1
    output_cols = ['sgrna', 'Gene', 'LFC_RIG', 'Z_RIG', 'Score_RIG']
    
    # Sort by score descending so the best guides are at the top
    df = df.sort_values(by='Score_RIG', ascending=False)
    df[output_cols].to_csv(args.out, sep='\t', index=False)

    print("-" * 50)
    print("PHASE 2 COMPLETE: RIG SIGMOID SCORES GENERATED")
    print("-" * 50)
    print(f"Total Guides Processed: {len(df):,}")
    print(f"NTC Calibrators Used:   {ntc_mask.sum():,}")
    print(f"NTC Baseline (LFC):     Mean = {mu:.4f}, Std = {sigma:.4f}")
    print(f"Score Range Output:     {df['Score_RIG'].min():.4f} to {df['Score_RIG'].max():.4f}")
    print(f"Output File:            {args.out}")
    print("-" * 50)

if __name__ == "__main__":
    main()

