import sys
import pandas as pd

def main():
    if len(sys.argv) != 4:
        print("Usage: python guide_matrix_to_bed.py <guide_matrix.txt> <track_name_prefix> <output_track.bed>")
        sys.exit(1)

    matrix_path = sys.argv[1]
    track_prefix = sys.argv[2]
    output_path = sys.argv[3]

    print(f"Loading guide-centered matrix: {matrix_path}...")
    df = pd.read_csv(matrix_path, sep='\t')

    # Dynamically find your guide-centered epigenetic columns
    guide_cols = sorted([c for c in df.columns if 'guide_' in c and 'bin_' in c])
    
    if not guide_cols:
        print("Error: No columns found with the prefix 'guide_' and suffix '_bin_'.")
        print("Double check your matrix column names.")
        sys.exit(1)

    # We use Guide Bin 5 (index 4) because it represents the exact 200bp footprint
    # sitting directly underneath the physical guide landing pad.
    target_bin = guide_cols[4] if len(guide_cols) >= 5 else guide_cols[0]
    print(f"Mapping track visual labels using target metric: {target_bin}")

    print(f"Writing browser track rows to: {output_path}...")
    with open(output_path, "w") as f:
        # Standard BED6 format: chrom, chromStart, chromEnd, name, score, strand
        for _, row in df.iterrows():
            chrom = str(row['Chromosome'])
            start = int(row['Start'])
            end = int(row['End'])
            strand = str(row['Strand'])
            gene = str(row['Gene'])
            
            # Extract the calculated score for the guide's localized footprint
            score_val = row[target_bin]
            score_display = round(score_val, 3) if pd.notna(score_val) else "NaN"
            
            # Create a high-visibility text label for the browser window
            label = f"{gene}_GuideLocal:{score_display}"
            
            # Write row (BED format uses 0-based start coordinates natively)
            f.write(f"{chrom}\t{start}\t{end}\t{label}\t1000\t{strand}\n")

    print(f"Success. Guide-centered browser track compiled at: {output_path}")

if __name__ == "__main__":
    main()
