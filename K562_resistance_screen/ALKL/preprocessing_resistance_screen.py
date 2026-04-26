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
    parser = argparse.ArgumentParser(description='CRISPRi Phase 2: File 1 Generation (Global Scores)')
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
    # Keeping only essential columns for File 1
    df_dmso = df_dmso[['sgrna', 'Gene', 'LFC']].rename(columns={'LFC': 'LFC_DMSO'})
    df_rig = df_rig[['sgrna', 'LFC']].rename(columns={'LFC': 'LFC_RIG'})
    df = pd.merge(df_dmso, df_rig, on='sgrna')

    # 3. Robust NTC Calibration
    # This regex ensures we catch NO-TARGET followed by anything (34, 75, etc.)
    ntc_mask = df['Gene'].str.contains('^NO-TARGET', na=False, case=False, regex=True)
    
    if ntc_mask.sum() == 0:
        print("ERROR: No NTCs found matching 'NO-TARGET'.")
        print(f"Found columns: {df.columns.tolist()}")
        print(f"First 5 Gene entries: {df['Gene'].head().tolist()}")
        sys.exit(1)
        
    ntc_lfc = df[ntc_mask]['LFC_DMSO']
    mu, sigma = ntc_lfc.mean(), ntc_lfc.std()
    
    # Check for zero variance to prevent division by zero errors
    if sigma == 0:
        print("ERROR: NTC variance is zero. Calibration impossible.")
        sys.exit(1)

    # 4. Standardized Z-scores
    df['Z_DMSO'] = (df['LFC_DMSO'] - mu) / sigma
    df['Z_RIG'] = (df['LFC_RIG'] - mu) / sigma

    # 5. Sigmoid Transformation & Delta
    df['Score_DMSO'] = sigmoid(df['Z_DMSO'])
    df['Score_RIG'] = sigmoid(df['Z_RIG'])
    df['Delta_Score'] = df['Score_RIG'] - df['Score_DMSO']

    # 6. Save File 1
    output_cols = [
        'sgrna', 'Gene', 'LFC_DMSO', 'LFC_RIG', 
        'Score_DMSO', 'Score_RIG', 'Delta_Score'
    ]
    
    df[output_cols].to_csv(args.out, sep='\t', index=False)

    print("-" * 40)
    print(f"FILE 1 GENERATED")
    print(f"NTCs used for calibration: {ntc_mask.sum()}")
    print(f"NTC Stats: Mean={mu:.4f}, Std={sigma:.4f}")
    print(f"Output: {args.out}")
    print("-" * 40)

if __name__ == "__main__":
    main()
