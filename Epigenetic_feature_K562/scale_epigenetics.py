import pandas as pd
import numpy as np
import sys
import os
import argparse

def scale_k562_epigenetics(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} does not exist.")
        sys.exit(1)
        
    # Read the tab-separated file
    df = pd.read_csv(input_path, sep='\t')
    
    # Isolate exactly the columns containing '_bin_' based on layout
    epigenetic_cols = [col for col in df.columns if '_bin_' in col]
    
    # Transform raw intensities to intra-cell line fractional percentiles [0.0, 1.0]
    for col in epigenetic_cols:
        if df[col].nunique() > 1:
            df[col] = df[col].rank(pct=True, method='average')
        else:
            df[col] = 0.0
            
    # Write to a new tab-separated file
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Processed K562: {input_path} -> Saved to {output_path} (Scaled {len(epigenetic_cols)} features)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scale K562 Epigenetic Features (Tab-Separated)')
    parser.add_argument('-i', '--input', required=True, help='Path to input K562 tab-separated file')
    parser.add_argument('-o', '--output', required=True, help='Path to output scaled K562 tab-separated file')
    
    args = parser.parse_args()
    scale_k562_epigenetics(args.input, args.output)
