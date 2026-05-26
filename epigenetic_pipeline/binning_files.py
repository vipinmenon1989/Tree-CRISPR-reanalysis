import sys
import pandas as pd
import os

def create_bins(row, window_size=2000, num_bins=10):
    """
    Generate 200bp bins across a 2kb window centered at the TSS.
    """
    tss_center = row['TSS']
    bin_size = window_size // num_bins
    start = tss_center - (window_size // 2)
    
    bins = []
    for i in range(num_bins):
        bin_start = start + (i * bin_size)
        bin_end = bin_start + bin_size
        bins.append({
            'gene': row['gene'],
            'chrom': row['chromosome'],
            'bin_id': i + 1,
            'bin_start': bin_start,
            'bin_end': bin_end,
            'strand': row.get('strand', '+')
        })
    return pd.DataFrame(bins)

def main():
    if len(sys.argv) < 2:
        print("Usage: python make_igv_bins.py <tss_annotation.csv>")
        sys.exit(1)

    tss_file = sys.argv[1]

    try:
        # Load the TSS annotation file
        df_tss = pd.read_csv(tss_file)
        df_tss.columns = [str(c).strip() for c in df_tss.columns]
        
        # Explicitly assign 'Primary TSS, 5\'' to 'TSS'
        if "Primary TSS, 5'" in df_tss.columns:
            df_tss['TSS'] = df_tss["Primary TSS, 5'"]
        else:
            raise ValueError("Could not find required column 'Primary TSS, 5\'' in the annotation file.")
        
        all_bins_list = []
        for _, row in df_tss.iterrows():
            all_bins_list.append(create_bins(row))
        
        bins_df = pd.concat(all_bins_list, ignore_index=True)
        
        # Export the intervals to BED format for IGV visualization
        bed_df = pd.DataFrame({
            'chrom': bins_df['chrom'],
            'start': bins_df['bin_start'],
            'end': bins_df['bin_end'],
            'name': bins_df['gene'] + "_Bin_" + bins_df['bin_id'].astype(str),
            'score': 1000,
            'strand': bins_df['strand']
        })
        
        bed_filename = 'bins_for_igv.bed'
        bed_df.to_csv(bed_filename, sep='\t', index=False, header=False)
        print(f"BED file generated for IGV visualization: {os.path.abspath(bed_filename)}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
