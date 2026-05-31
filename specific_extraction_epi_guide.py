import sys
import pandas as pd
import numpy as np
import pyBigWig

def main():
    if len(sys.argv) != 5:
        print("Usage: python guide_step3_extract_epi.py <guide_step2_output.txt> <raw_signal.bw> <cpg_methylation.bigBed> <guide_step3_output.txt>")
        sys.exit(1)

    coords_path = sys.argv[1]
    bw_path = sys.argv[2]
    bigbed_path = sys.argv[3]
    output_path = sys.argv[4]

    df = pd.read_csv(coords_path, sep='\t')

    # ==========================================
    # SUB-PART A: Extract Raw Coverage (BigWig)
    # ==========================================
    print(f"Extracting raw signal tracking from: {bw_path}...")
    bw = pyBigWig.open(bw_path)
    cov_columns = {f"guide_methylation_coverage_bin_{i}": [] for i in range(1, 11)}

    for _, row in df.iterrows():
        chrom = str(row['Chromosome'])
        g_start = row['Start']

        if pd.isna(g_start):
            for i in range(1, 11):
                cov_columns[f"guide_methylation_coverage_bin_{i}"].append(np.nan)
            continue

        if not chrom.startswith('chr') and 'chr' in bw.chroms().keys():
            chrom = f"chr{chrom}"
        elif chrom.startswith('chr') and chrom not in bw.chroms().keys():
            chrom = chrom.replace('chr', '')

        if chrom not in bw.chroms():
            for i in range(1, 11):
                cov_columns[f"guide_methylation_coverage_bin_{i}"].append(np.nan)
            continue

        for i in range(1, 11):
            start = row[f"Guide_Bin{i}_Start"]
            end = row[f"Guide_Bin{i}_End"]

            if pd.isna(start) or pd.isna(end):
                cov_columns[f"guide_methylation_coverage_bin_{i}"].append(np.nan)
                continue

            try:
                mean_val = bw.stats(chrom, int(start), int(end), type="mean")[0]
                if mean_val is None:
                    mean_val = np.nan
            except Exception:
                mean_val = np.nan
            cov_columns[f"guide_methylation_coverage_bin_{i}"].append(mean_val)
    bw.close()

    for col_name, score_list in cov_columns.items():
        df[col_name] = score_list

    # ==========================================
    # SUB-PART B: Extract Fractional Percentage (bigBed)
    # ==========================================
    print(f"Extracting CpG methylation fractions from: {bigbed_path}...")
    bb = pyBigWig.open(bigbed_path)
    pct_columns = {f"guide_methylation_percent_bin_{i}": [] for i in range(1, 11)}

    for _, row in df.iterrows():
        chrom = str(row['Chromosome'])
        g_start = row['Start']

        if pd.isna(g_start):
            for i in range(1, 11):
                pct_columns[f"guide_methylation_percent_bin_{i}"].append(np.nan)
            continue

        if not chrom.startswith('chr') and 'chr' in bb.chroms().keys():
            chrom = f"chr{chrom}"
        elif chrom.startswith('chr') and chrom not in bb.chroms().keys():
            chrom = chrom.replace('chr', '')

        if chrom not in bb.chroms():
            for i in range(1, 11):
                pct_columns[f"guide_methylation_percent_bin_{i}"].append(np.nan)
            continue

        for i in range(1, 11):
            start = row[f"Guide_Bin{i}_Start"]
            end = row[f"Guide_Bin{i}_End"]

            if pd.isna(start) or pd.isna(end):
                pct_columns[f"guide_methylation_percent_bin_{i}"].append(np.nan)
                continue

            try:
                entries = bb.entries(chrom, int(start), int(end))
                if not entries:
                    pct_columns[f"guide_methylation_percent_bin_{i}"].append(np.nan)
                    continue
                
                bin_scores = []
                for entry in entries:
                    fields = entry[2].split('\t')
                    meth_score = float(fields[1]) / 100.0  
                    bin_scores.append(meth_score)
                
                mean_pct = np.mean(bin_scores) if bin_scores else np.nan
                pct_columns[f"guide_methylation_percent_bin_{i}"].append(mean_pct)
            except Exception:
                pct_columns[f"guide_methylation_percent_bin_{i}"].append(np.nan)
    bb.close()

    for col_name, score_list in pct_columns.items():
        df[col_name] = score_list

    df.to_csv(output_path, sep='\t', index=False)
    print(f"Part 2 Complete. Merged array data saved to: {output_path}")

if __name__ == "__main__":
    main()
