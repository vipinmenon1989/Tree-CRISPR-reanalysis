import sys
import os
import pandas as pd
import numpy as np
import pyBigWig

def main():
    if len(sys.argv) != 4:
        print("Usage: python guide_step3_unified_folder.py <guide_step2_output.txt> <tracks_folder_path> <final_matrix_output.txt>")
        sys.exit(1)

    coords_path = sys.argv[1]
    bw_dir = sys.argv[2]
    final_output_path = sys.argv[3]

    # 1. Load the expanded guide coordinate library from Guide Step 2
    df = pd.read_csv(coords_path, sep='\t')

    # 2. Identify all track files inside the target directory
    if not os.path.isdir(bw_dir):
        print(f"Error: The directory pathway '{bw_dir}' does not exist.")
        sys.exit(1)

    # Filtering strictly for extensions: .bigwig and .bigBed
    valid_extensions = ('.bigwig', '.bigBed')
    track_files = [f for f in os.listdir(bw_dir) if f.endswith(valid_extensions)]
    
    if not track_files:
        print(f"Error: No valid tracks found in '{bw_dir}' (looking specifically for .bigwig or .bigBed).")
        sys.exit(1)

    print(f"Found {len(track_files)} track file(s) to process.")

    # 3. Iterate through each track file found in the folder
    for track_file in track_files:
        track_path = os.path.join(bw_dir, track_file)
        file_prefix = os.path.splitext(track_file)[0]
        
        is_bigbed = track_file.endswith('.bigBed')
        
        # Add a unique "guide_" prefix to column output arrays to distinguish them from gene-level data
        if is_bigbed:
            print(f"Processing Binary CpG Methylation State Track: {file_prefix} ...")
            file_columns = {f"guide_{file_prefix}_percent_bin_{i}": [] for i in range(1, 11)}
        else:
            print(f"Processing Continuous Signal/Fold-Enrichment Track: {file_prefix} ...")
            file_columns = {f"guide_{file_prefix}_bin_{i}": [] for i in range(1, 11)}

        # Open connection handle using pyBigWig
        track = pyBigWig.open(track_path)

        for _, row in df.iterrows():
            chrom = str(row['Chromosome'])
            g_start = row['Start']

            # CRITICAL FIX 1: Safeguard row using the guide coordinate, NOT the gene TSS
            if pd.isna(g_start):
                for i in range(1, 11):
                    suffix = f"percent_bin_{i}" if is_bigbed else f"bin_{i}"
                    file_columns[f"guide_{file_prefix}_{suffix}"].append(np.nan)
                continue

            # Standardize chromosome prefixes to match track header expectations
            if not chrom.startswith('chr') and 'chr' in track.chroms().keys():
                chrom = f"chr{chrom}"
            elif chrom.startswith('chr') and chrom not in track.chroms().keys():
                chrom = chrom.replace('chr', '')

            if chrom not in track.chroms():
                for i in range(1, 11):
                    suffix = f"percent_bin_{i}" if is_bigbed else f"bin_{i}"
                    file_columns[f"guide_{file_prefix}_{suffix}"].append(np.nan)
                continue

            # Extract metrics across the 10 pre-calculated Guide bin coordinate pairs
            for i in range(1, 11):
                # CRITICAL FIX 2: Point lookups directly to your Guide_Bin column coordinates
                start_coord = row[f"Guide_Bin{i}_Start"]
                end_coord = row[f"Guide_Bin{i}_End"]

                if pd.isna(start_coord) or pd.isna(end_coord):
                    suffix = f"percent_bin_{i}" if is_bigbed else f"bin_{i}"
                    file_columns[f"guide_{file_prefix}_{suffix}"].append(np.nan)
                    continue

                try:
                    if is_bigbed:
                        # STRATEGY 1: bigBed Individual Site Entry Evaluation (Percent)
                        entries = track.entries(chrom, int(start_coord), int(end_coord))
                        if not entries:
                            mean_val = np.nan
                        else:
                            bin_scores = []
                            for entry in entries:
                                fields = entry[2].split('\t')
                                meth_score = float(fields[1]) / 100.0  # Scale down to 0.0 - 1.0
                                bin_scores.append(meth_score)
                            mean_val = np.mean(bin_scores) if bin_scores else np.nan
                    else:
                        # STRATEGY 2: Continuous bigwig Window Extraction (Fold / Coverage)
                        mean_val = track.stats(chrom, int(start_coord), int(end_coord), type="mean")[0]
                        if mean_val is None:
                            mean_val = np.nan
                except Exception:
                    mean_val = np.nan

                suffix = f"percent_bin_{i}" if is_bigbed else f"bin_{i}"
                file_columns[f"guide_{file_prefix}_{suffix}"].append(mean_val)

        track.close()

        # Append calculated track columns directly to our running DataFrame
        for col_name, score_list in file_columns.items():
            df[col_name] = score_list

    # 4. CRITICAL FIX 3: Drop intermediate Guide coordinate helper columns before saving
    cols_to_drop = []
    for i in range(1, 11):
        cols_to_drop.extend([f"Guide_Bin{i}_Start", f"Guide_Bin{i}_End"])
    df = df.drop(columns=cols_to_drop)

    # 5. Export final master flat matrix file
    df.to_csv(final_output_path, sep='\t', index=False)
    print(f"Guide Folder Processing Complete. Integrated file saved to: {final_output_path}")

if __name__ == "__main__":
    main()
