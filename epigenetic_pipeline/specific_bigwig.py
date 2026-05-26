import sys
import pandas as pd
import numpy as np
import pyBigWig

def main():
    if len(sys.argv) != 5:
        print("Usage: python step3b_dual_methylation.py <step2_output.txt> <raw_signal.bw> <cpg_methylation.bigBed> <final_output.txt>")
        sys.exit(1)

    coords_path = sys.argv[1]
    bw_path = sys.argv[2]
    bigbed_path = sys.argv[3]
    final_output_path = sys.argv[4]

    # 1. Load Step 2 matrix
    df = pd.read_csv(coords_path, sep='\t')

    # ==========================================
    # PART A: Extract Raw Coverage (BigWig)
    # ==========================================
    print(f"Extracting raw signal tracking from: {bw_path}...")
    bw = pyBigWig.open(bw_path)
    
    cov_columns = {f"dna_methylation_coverage_bin_{i}": [] for i in range(1, 11)}

    for _, row in df.iterrows():
        chrom = str(row['Chromosome'])
        tss_pos = row['closest_TSS_coord']

        if pd.isna(tss_pos):
            for i in range(1, 11):
                cov_columns[f"dna_methylation_coverage_bin_{i}"].append(np.nan)
            continue

        if not chrom.startswith('chr') and 'chr' in bw.chroms().keys():
            chrom = f"chr{chrom}"
        elif chrom.startswith('chr') and chrom not in bw.chroms().keys():
            chrom = chrom.replace('chr', '')

        if chrom not in bw.chroms():
            for i in range(1, 11):
                cov_columns[f"dna_methylation_coverage_bin_{i}"].append(np.nan)
            continue

        for i in range(1, 11):
            start = row[f"Bin{i}_Start"]
            end = row[f"Bin{i}_End"]

            if pd.isna(start) or pd.isna(end):
                cov_columns[f"dna_methylation_coverage_bin_{i}"].append(np.nan)
                continue

            try:
                mean_val = bw.stats(chrom, int(start), int(end), type="mean")[0]
                if mean_val is None:
                    mean_val = np.nan
            except Exception:
                mean_val = np.nan
            cov_columns[f"dna_methylation_coverage_bin_{i}"].append(mean_val)
    bw.close()

    for col_name, score_list in cov_columns.items():
        df[col_name] = score_list

    # ==========================================
    # PART B: Extract Fractional Percentage (bigBed)
    # ==========================================
    print(f"Calculating biological CpG methylation fractions from bigBed: {bigbed_path}...")
    
    # pyBigWig can natively open and query bigBed files using entries()
    bb = pyBigWig.open(bigbed_path)
    
    pct_columns = {f"dna_methylation_percent_bin_{i}": [] for i in range(1, 11)}

    for _, row in df.iterrows():
        chrom = str(row['Chromosome'])
        tss_pos = row['closest_TSS_coord']

        if pd.isna(tss_pos):
            for i in range(1, 11):
                pct_columns[f"dna_methylation_percent_bin_{i}"].append(np.nan)
            continue

        if not chrom.startswith('chr') and 'chr' in bb.chroms().keys():
            chrom = f"chr{chrom}"
        elif chrom.startswith('chr') and chrom not in bb.chroms().keys():
            chrom = chrom.replace('chr', '')

        if chrom not in bb.chroms():
            for i in range(1, 11):
                pct_columns[f"dna_methylation_percent_bin_{i}"].append(np.nan)
            continue

        # Extract matches for each of the 10 horizontal bin windows
        for i in range(1, 11):
            start = row[f"Bin{i}_Start"]
            end = row[f"Bin{i}_End"]

            if pd.isna(start) or pd.isna(end):
                pct_columns[f"dna_methylation_percent_bin_{i}"].append(np.nan)
                continue

            try:
                # Fetch all individual CpG entries intersecting this 200bp interval
                entries = bb.entries(chrom, int(start), int(end))
                
                if not entries:
                    pct_columns[f"dna_methylation_percent_bin_{i}"].append(np.nan)
                    continue
                
                bin_scores = []
                for entry in entries:
                    # entry format: (start, end, string_data)
                    # ENCODE bedMethyl string_data looks like: "name\tscore\tstrand\tthickStart\t..."
                    # The score percentage value (0-100) is the 2nd item inside the string data array
                    fields = entry[2].split('\t')
                    meth_score = float(fields[1]) / 100.0  # Convert 0-100 down to 0.0-1.0
                    bin_scores.append(meth_score)
                
                mean_pct = np.mean(bin_scores) if bin_scores else np.nan
                pct_columns[f"dna_methylation_percent_bin_{i}"].append(mean_pct)

            except Exception:
                pct_columns[f"dna_methylation_percent_bin_{i}"].append(np.nan)
                
    bb.close()

    for col_name, score_list in pct_columns.items():
        df[col_name] = score_list

    # ==========================================
    # PART C: Cleanup and Export
    # ==========================================
    cols_to_drop = []
    for i in range(1, 11):
        cols_to_drop.extend([f"Bin{i}_Start", f"Bin{i}_End"])
    df = df.drop(columns=cols_to_drop)

    df.to_csv(final_output_path, sep='\t', index=False)
    print(f"Step 3b Complete. Combined matrix file cleanly generated at: {final_output_path}")

if __name__ == "__main__":
    main()
