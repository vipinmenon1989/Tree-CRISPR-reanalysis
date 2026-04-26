import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='CRISPRi Phase 1: Resistance Discovery (Custom Headers)')
    parser.add_argument('--dmso', required=True, help='Path to K562_dmso_gene_summary.txt')
    parser.add_argument('--rig', required=True, help='Path to K562_rig_gene_summary.txt')
    parser.add_argument('--out', default='Phase1_Resistance_Hits.txt', help='Output filename')
    args = parser.parse_args()

    # 1. Load Data
    try:
        df_dmso = pd.read_csv(args.dmso, sep='\t')
        df_rig = pd.read_csv(args.rig, sep='\t')
    except Exception as e:
        print(f"Error loading files: {e}")
        sys.exit(1)

    # 2. Extract and Rename based on your specific MAGeCK headers
    # Note: id is Gene, pos|lfc is our X/Y, pos|p-value and pos|goodsgrna are our filters
    df_dmso = df_dmso[['id', 'pos|lfc']].rename(columns={'pos|lfc': 'LFC_DMSO', 'id': 'Gene'})
    
    df_rig = df_rig[['id', 'pos|lfc', 'pos|p-value', 'pos|goodsgrna']].rename(
        columns={
            'pos|lfc': 'LFC_RIG', 
            'pos|p-value': 'P_Value_RIG', 
            'pos|goodsgrna': 'Good_Guides_RIG', 
            'id': 'Gene'
        }
    )
    
    # 3. Merge
    df = pd.merge(df_dmso, df_rig, on='Gene')

    # 4. Global Linear Regression
    X = df[['LFC_DMSO']].values
    y = df['LFC_RIG'].values
    reg = LinearRegression().fit(X, y)
    
    # Calculate GI Residual
    df['GI_Residual'] = df['LFC_RIG'] - reg.predict(X)

    # 5. Apply Strict Resistance Filters
    hits = df[
        (df['P_Value_RIG'] <= 0.05) & 
        (df['GI_Residual'] > 1.0) & 
        (df['Good_Guides_RIG'] >= 3)
    ].copy()

    # 6. Final Formatting (Exactly what you asked for)
    output_cols = ['Gene', 'GI_Residual', 'P_Value_RIG', 'Good_Guides_RIG']
    hits = hits[output_cols].sort_values(by='GI_Residual', ascending=False)

    # Save as Tab-Separated .txt
    hits.to_csv(args.out, sep='\t', index=False)

    # Reporting
    print("-" * 40)
    print(f"PHASE 1 COMPLETE")
    print(f"Initial Genome Size: {len(df)}")
    print(f"Functional Hits Found: {len(hits)}")
    print(f"Results saved to: {args.out}")
    print("-" * 40)

if __name__ == "__main__":
    main()
