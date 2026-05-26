import pandas as pd
import numpy as np
import pyBigWig
import sys
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_single_bigwig(bw_path, intervals, window_total=2500, bins=10):
    """
    Worker function: Opens a single BigWig, processes ALL intervals, 
    and returns the spatial features (Mean, Max, Std) for each bin.
    """
    try:
        bw = pyBigWig.open(bw_path)
    except Exception as e:
        print(f"[ERROR] Could not open {bw_path}: {e}")
        return None, {}

    bin_size = window_total // bins
    half_window = window_total // 2
    # Use the filename (without extension) as the prefix for the ML columns
    bw_name = os.path.splitext(os.path.basename(bw_path))[0]
    
    results = {}
    
    for item in intervals:
        chrom = str(item['chrom'])
        features = {}
        
        # If chromosome is missing from this BigWig, return zeros
        if chrom not in bw.chroms():
            for i in range(1, bins + 1):
                features[f'{bw_name}_bin{i}_mean'] = 0.0
                features[f'{bw_name}_bin{i}_max'] = 0.0
                features[f'{bw_name}_bin{i}_std'] = 0.0
            results[item['id']] = features
            continue
            
        # Absolute coordinate for the upstream edge of the 2.5kb window
        start_coord = max(0, int(item['tss']) - half_window)
        
        for i in range(bins):
            b_start = start_coord + (i * bin_size)
            b_end = b_start + bin_size
            
            try:
                signal = bw.values(chrom, b_start, b_end)
                signal_arr = np.nan_to_num(np.array(signal, dtype=float), nan=0.0)
                
                if len(signal_arr) == 0:
                    b_mean, b_max, b_std = 0.0, 0.0, 0.0
                else:
                    b_mean = np.mean(signal_arr)
                    b_max = np.max(signal_arr)
                    b_std = np.std(signal_arr)
            except RuntimeError:
                b_mean, b_max, b_std = 0.0, 0.0, 0.0
                
            # Geometry mapping: Reverse bin numbering if on the negative strand
            bin_idx = (bins - i) if item['strand'] == '-' else (i + 1)
            
            features[f'{bw_name}_bin{bin_idx}_mean'] = b_mean
            features[f'{bw_name}_bin{bin_idx}_max']  = b_max
            features[f'{bw_name}_bin{bin_idx}_std']  = b_std
            
        results[item['id']] = features
        
    bw.close()
    return bw_name, results


def main():
    parser = argparse.ArgumentParser(description='Parallel Spatial Binning for Epigenetic ML Features')
    parser.add_argument('-i', '--input', required=True, help='Path to your sgRNA .txt file')
    parser.add_argument('-o', '--out', default='Final_ML_Feature_Matrix.csv', help='Output CSV filename')
    parser.add_argument('-w', '--window', type=int, default=2500, help='Total window size in bp (default 2500)')
    parser.add_argument('-b', '--bins', type=int, default=10, help='Number of bins (default 10)')
    # Accept any number of BigWig files at the end of the command
    parser.add_argument('bigwigs', nargs='+', help='List of .bigwig files to process')
    
    args = parser.parse_args()

    print("="*60)
    print(f"[INFO] Initializing Parallel Feature Extraction")
    print(f"[INFO] Window: +/- {args.window//2}bp | Bins: {args.bins} ({args.window//args.bins}bp per bin)")
    print("="*60)

    # 1. Load Data
    print(f"[INFO] Loading input file: {args.input}")
    try:
        # Using sep='\t' since it's a .txt file, fallback to comma if needed
        df = pd.read_csv(args.input, sep=None, engine='python') 
    except Exception as e:
        print(f"CRITICAL ERROR loading TXT: {e}")
        sys.exit(1)

    # Clean headers (strip whitespace) just in case
    df.columns = [col.strip() for col in df.columns]

    # Map your specific columns (case-insensitive matching to be safe)
    col_map = {c.lower(): c for c in df.columns}
    
    req_keys = ['sgrna_id', 'chr', 'start', 'strand', 'distance from tss']
    actual_cols = {}
    for key in req_keys:
        if key not in col_map:
            # If standard 'chr' is missing, try 'chromosome'
            if key == 'chr' and 'chromosome' in col_map:
                actual_cols[key] = col_map['chromosome']
                continue
            print(f"CRITICAL ERROR: Missing column '{key}' in input file.")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        actual_cols[key] = col_map[key]

    # 2. Back-calculate the TSS
    print("[INFO] Calculating absolute TSS coordinates...")
    df['TSS_Coord'] = df[actual_cols['start']] - df[actual_cols['distance from tss']]
    df['TSS_Coord'] = df['TSS_Coord'].astype(int)

    # 3. Build Interval List for Workers
    intervals = []
    for _, row in df.iterrows():
        intervals.append({
            'id': row[actual_cols['sgrna_id']],
            'chrom': row[actual_cols['chr']],
            'tss': row[actual_cols['TSS_Coord']],
            'strand': row[actual_cols['strand']]
        })

    # 4. Multiprocessing BigWig Extraction
    print(f"[INFO] Launching extraction on {len(args.bigwigs)} BigWig files concurrently...")
    combined_features = {item['id']: {} for item in intervals}
    
    with ProcessPoolExecutor() as executor:
        future_to_bw = {
            executor.submit(process_single_bigwig, bw_file, intervals, args.window, args.bins): bw_file 
            for bw_file in args.bigwigs
        }
        
        for future in as_completed(future_to_bw):
            bw_file = future_to_bw[future]
            try:
                bw_name, results = future.result()
                if not results:
                    continue
                print(f"  -> [SUCCESS] Processed {bw_name}")
                # Merge this BigWig's features into the master dictionary
                for sgrna_id, features in results.items():
                    combined_features[sgrna_id].update(features)
            except Exception as exc:
                print(f"  -> [FAILED] {os.path.basename(bw_file)} generated an exception: {exc}")

    # 5. Merge Features back to original DataFrame
    print("[INFO] Merging parallel results into final matrix...")
    features_df = pd.DataFrame.from_dict(combined_features, orient='index')
    features_df.index.name = actual_cols['sgrna_id']
    features_df.reset_index(inplace=True)

    final_df = pd.merge(df, features_df, on=actual_cols['sgrna_id'], how='left')

    # 6. Save
    final_df.to_csv(args.out, index=False)
    print("="*60)
    print(f"[COMPLETED] Final ML matrix saved to: {args.out}")
    print(f"[COMPLETED] Added {features_df.shape[1] - 1} epigenetic features.")
    print("="*60)

if __name__ == "__main__":
    main()
