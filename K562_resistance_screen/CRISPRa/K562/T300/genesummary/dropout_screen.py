import pandas as pd
import argparse
import sys

def process_essentiality_hits(input_file, output_file, p_val_cutoff, min_guides):
    """
    Extracts functional hits from a MAGeCK negative selection gene summary.
    """
    try:
        # Load the MAGeCK gene summary
        df = pd.read_csv(input_file, sep='\t')
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load file {input_file}. ({e})")
        sys.exit(1)

    # Check if required MAGeCK columns exist
    req_cols = ['id', 'neg|lfc', 'neg|p-value', 'neg|goodsgrna']
    missing = [col for col in req_cols if col not in df.columns]
    if missing:
        print(f"CRITICAL ERROR: Missing expected columns in MAGeCK file: {missing}")
        sys.exit(1)

    # Extract and clean up the columns
    df_clean = df[['id', 'neg|lfc', 'neg|p-value', 'neg|goodsgrna']].copy()
    df_clean.columns = ['Gene_ID', 'LFC', 'P_Value', 'Good_sgRNAs']

    # --- APPLY FILTERS ---
    mask = (df_clean['P_Value'] <= p_val_cutoff) & (df_clean['Good_sgRNAs'] >= min_guides)
    hits = df_clean[mask].copy()

    # Sort by strongest dropout (most negative LFC)
    hits = hits.sort_values(by='LFC', ascending=True)

    # --- SAVE DATA ---
    try:
        hits.to_csv(output_file, sep='\t', index=False)
        print("-" * 40)
        print(f"SUCCESS: Processed {len(df_clean)} total genes from {input_file}")
        print(f"Functional Hits Found: {len(hits)} (P <= {p_val_cutoff}, Guides >= {min_guides})")
        print(f"Saved results to: {output_file}")
        print("-" * 40)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to save output to {output_file}. ({e})")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Extract strictly functional essentiality hits from a MAGeCK gene summary.")
    
    # Required Arguments
    parser.add_argument("-i", "--input", required=True, help="Path to the input MAGeCK gene_summary.txt")
    parser.add_argument("-o", "--output", required=True, help="Path/filename for the saved output (e.g., ALKME_hits.txt)")
    
    # Optional Arguments (Configurable Cutoffs)
    parser.add_argument("-p", "--pvalue", type=float, default=0.05, help="P-value cutoff (Default: 0.05)")
    parser.add_argument("-g", "--guides", type=int, default=3, help="Minimum 'good' sgRNAs required (Default: 3)")

    args = parser.parse_args()

    process_essentiality_hits(args.input, args.output, args.pvalue, args.guides)

if __name__ == "__main__":
    main()
