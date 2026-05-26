import sys
import pandas as pd

def main():
    if len(sys.argv) != 3:
        print("Usage: python matrix_to_bed.py <step2_output.txt> <visualization_bins.bed>")
        sys.exit(1)

    input_path = sys.argv[1]
    out_bed_path = sys.argv[2]

    # Load Step 2 matrix
    df = pd.read_csv(input_path, sep='\t')

    bed_rows = []

    print("Converting horizontal bin matrix to long-form BED tracks...")
    for _, row in df.iterrows():
        chrom = str(row['Chromosome'])
        gene = row['Gene']
        tss_pos = row['closest_TSS_coord']
        strand = row['Gene_Strand']

        # Skip rows that didn't map to a TSS
        if pd.isna(tss_pos) or pd.isna(strand):
            continue

        # Standardize chromosome format to 'chr' prefix for browser stability
        if not chrom.startswith('chr'):
            chrom = f"chr{chrom}"

        # Loop through the 10 bins and pull their specific coordinates
        for i in range(1, 11):
            start_val = row[f"Bin{i}_Start"]
            end_val = row[f"Bin{i}_End"]

            if pd.isna(start_val) or pd.isna(end_val):
                continue

            # Unique name for browser visualization tracking
            track_name = f"{gene}_{int(tss_pos)}_Bin_{i}"

            bed_rows.append({
                'chrom': chrom,
                'chromStart': int(start_val),
                'chromEnd': int(end_val),
                'name': track_name,
                'score': 0,           # Placeholder score required for BED6
                'strand': strand      # Matches the Gene's strand orientation
            })

    # Export to standard BED6 format (No header, tab-separated)
    output_df = pd.DataFrame(bed_rows)
    output_df.to_csv(
        out_bed_path, 
        sep='\t', 
        index=False, 
        header=False, 
        columns=['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand']
    )
    print(f"Conversion complete! Created {len(output_df)} track bins. File saved to: {out_bed_path}")

if __name__ == "__main__":
    main()
