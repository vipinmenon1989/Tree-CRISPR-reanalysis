import sys
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) != 3:
        print("Usage: python guide_step2_make_windows.py <step1_output.txt> <guide_step2_output.txt>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # 1. Load Step 1 base matrix
    df = pd.read_csv(input_path, sep='\t')

    # 2. Define relative 200bp bin offsets centered on 0 (-1000 to +1000)
    bin_offsets = [
        (-1000, -800), (-800, -600), (-600, -400), (-400, -200), (-200, 0),
        (0, 200), (200, 400), (400, 600), (600, 800), (800, 1000)
    ]

    # Initialize storage tracking with unique 'Guide_' prefixes to avoid collision
    window_columns = {}
    for i in range(1, 11):
        window_columns[f"Guide_Bin{i}_Start"] = []
        window_columns[f"Guide_Bin{i}_End"] = []

    print("Calculating absolute genomic coordinates centered on guide midpoints...")

    # 3. Compute absolute coordinates row-by-row
    for _, row in df.iterrows():
        g_start = row['Start']
        g_end = row['End']

        if pd.isna(g_start) or pd.isna(g_end):
            for i in range(1, 11):
                window_columns[f"Guide_Bin{i}_Start"].append(np.nan)
                window_columns[f"Guide_Bin{i}_End"].append(np.nan)
            continue

        # Mathematical midpoint of the 20bp guide sequence
        guide_midpoint = int((int(g_start) + int(g_end)) / 2)

        for i, (rel_start, rel_end) in enumerate(bin_offsets, start=1):
            abs_start = guide_midpoint + rel_start
            abs_end = guide_midpoint + rel_end

            if abs_start < 0 or abs_end < 0:
                abs_start, abs_end = np.nan, np.nan

            window_columns[f"Guide_Bin{i}_Start"].append(abs_start)
            window_columns[f"Guide_Bin{i}_End"].append(abs_end)

    # 4. Append calculated coordinates to the DataFrame
    for col_name, coords_list in window_columns.items():
        df[col_name] = pd.Series(coords_list).astype('Int64')

    # 5. Export coordinate matrix file
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Part 1 Complete. Guide coordinate matrix saved to: {output_path}")

if __name__ == "__main__":
    main()
