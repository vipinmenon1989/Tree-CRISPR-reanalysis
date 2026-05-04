import pandas as pd
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='CRISPR Resistance Discovery: Raw RIG Method (No Normalization)')
    parser.add_argument('--rig', required=True, help='Path to RIG gene_summary.txt')
    parser.add_argument('--out', default='Raw_RIG_Resistance_Hits.txt', help='Output filename')
    parser.add_argument('--pvalue', type=float, default=0.05, help='P-value threshold (default: 0.05)')
    parser.add_argument('--guides', type=int, default=3, help='Minimum good sgRNAs (default: 3)')
    parser.add_argument('--top', type=int, default=None, help='Extract exactly top N hits (for fair set comparison)')
    args = parser.parse_args()

    # 1. Load Data
    try:
        df_rig = pd.read_csv(args.rig, sep='\t')
    except Exception as e:
        print(f"CRITICAL ERROR: Failed loading file {args.rig} - {e}")
        sys.exit(1)

    # Check required columns
    req_cols = ['id', 'pos|lfc', 'pos|p-value', 'pos|goodsgrna']
    missing = [col for col in req_cols if col not in df_rig.columns]
    if missing:
        print(f"CRITICAL ERROR: Missing expected columns: {missing}")
        sys.exit(1)

    # 2. Extract and Rename
    df_rig = df_rig[req_cols].rename(
        columns={'id': 'Gene', 'pos|lfc': 'LFC_RIG', 'pos|p-value': 'P_Value_RIG', 'pos|goodsgrna': 'Good_Guides_RIG'}
    )

    # 3. Apply Quality Filters (NO DMSO Normalization)
    mask = (df_rig['P_Value_RIG'] <= args.pvalue) & (df_rig['Good_Guides_RIG'] >= args.guides)
    hits = df_rig[mask].copy()

    # 4. Sort strictly by Raw RIG LFC (Highest survival at the top)
    hits = hits.sort_values(by='LFC_RIG', ascending=False)

    # 5. Cohort Size Matching (For Set Subtraction)
    if args.top is not None:
        hits = hits.head(args.top)
        print(f"Subsetting to Top {args.top} hits for fair comparison.")

    # 6. Output
    hits.to_csv(args.out, sep='\t', index=False)

    print("-" * 50)
    print("RAW RIG PIPELINE COMPLETE (No Baseline Normalization)")
    print(f"Initial Genome Size: {len(df_rig)}")
    print(f"Functional Hits Found: {len(hits)} (P<={args.pvalue}, Guides>={args.guides})")
    print(f"Results saved to: {args.out}")
    print("-" * 50)

if __name__ == "__main__":
    main()
