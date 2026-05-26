import sys
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) != 3:
        print("Usage: python step2_make_windows.py <step1_output.txt> <step2_output.txt>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # 1. Load the Step 1 output file
    # We use sep='\t' because Step 1 exports a standard tab-separated file
    df = pd.read_csv(input_path, sep='\t')

    # 2. Define our relative 200bp bin offsets (-1000 to +1000)
    bin_offsets = [
        (-1000, -800), (-800, -600), (-600, -400), (-400, -200), (-200, 0),
        (0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)
    ]

    # Initialize separate dictionary arrays to track coordinates for the 20 new columns
    window_columns = {}
    for i in range(1, 11):
        window_columns[f"Bin{i}_Start"] = []
        window_columns[f"Bin{i}_End"] = []

    print("Calculating absolute genomic coordinates for all 10 bins horizontally...")

    # 3. Compute absolute coordinates row-by-row based on the mapped TSS and Strand
    for _, row in df.iterrows():
        tss_pos = row['closest_TSS_coord']
        strand = row['Gene_Strand']

        # If a guide didn't map to a TSS, fill all its bin coordinate columns with NaN
        if pd.isna(tss_pos) or pd.isna(strand):
            for i in range(1, 11):
                window_columns[f"Bin{i}_Start"].append(np.nan)
                window_columns[f"Bin{i}_End"].append(np.nan)
            continue

        tss_pos = int(tss_pos)

        # Iterate through the 10 bin intervals
        for i, (rel_start, rel_end) in enumerate(bin_offsets, start=1):
            if strand == '+':
                # Positive strand: upstream is lower coordinate, downstream is higher
                abs_start = tss_pos + rel_start
                abs_end = tss_pos + rel_end
            elif strand == '-':
                # Negative strand: upstream is higher coordinate, downstream is lower
                # Standard genomics dictates Start must ALWAYS be smaller than End
                abs_start = tss_pos - rel_end
                abs_end = tss_pos - rel_start
            else:
                abs_start, abs_end = np.nan, np.nan

            # Coordinate check to avoid invalid negative genomic coordinates
            if pd.notna(abs_start) and (abs_start < 0 or abs_end < 0):
                abs_start, abs_end = np.nan, np.nan

            window_columns[f"Bin{i}_Start"].append(abs_start)
            window_columns[f"Bin{i}_End"].append(abs_end)

    # 4. Append the calculated coordinates directly to the DataFrame as new columns
    for col_name, coords_list in window_columns.items():
        # Using 'Int64' allows integer formatting while preserving NaNs if unmapped
        df[col_name] = pd.Series(coords_list).astype('Int64')

    # 5. Export the expanded matrix base file
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Step 2 complete. Expanded coordinate matrix saved to: {output_path}")

if __name__ == "__main__":
    main()
