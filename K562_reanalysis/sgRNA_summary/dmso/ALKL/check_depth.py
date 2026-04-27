import pandas as pd
import argparse
import sys

def check_sequencing_depth(file_path):
    """Calculates the percentage of guides with low read counts."""
    try:
        # Load the tab-separated MAGeCK summary file
        df = pd.read_csv(file_path, sep='\t')
    except FileNotFoundError:
        print(f"CRITICAL ERROR: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read file. ({e})", file=sys.stderr)
        sys.exit(1)

    # Validate that the necessary column exists
    if 'treat_mean' not in df.columns:
        print("CRITICAL ERROR: Column 'treat_mean' not found.", file=sys.stderr)
        print("Make sure you are passing the sgRNA summary file (sgrna_summary.txt).", file=sys.stderr)
        sys.exit(1)

    # Perform the calculation
    total_guides = len(df)
    dead_guides_mask = df['treat_mean'] < 30
    dead_count = dead_guides_mask.sum()
    low_reads_pct = dead_guides_mask.mean() * 100

    # Output strictly to stdout
    print("-" * 40)
    print(f"File: {file_path}")
    print(f"Total Guides: {total_guides:,}")
    print(f"Dead Guides (< 30 reads): {dead_count:,}")
    print(f"Percentage Dead: {low_reads_pct:.2f}%")
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Check sequencing depth QC from a MAGeCK sgRNA summary file.")
    
    # Require the input file as a positional argument for speed
    parser.add_argument("input_file", help="Path to the tab-separated sgRNA summary file")
    
    args = parser.parse_args()
    check_sequencing_depth(args.input_file)

if __name__ == "__main__":
    main()
