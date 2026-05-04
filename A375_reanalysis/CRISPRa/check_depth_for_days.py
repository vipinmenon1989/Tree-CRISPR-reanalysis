import pandas as pd
import numpy as np
import argparse

def calculate_gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # All values must be non-negative
    array = np.array(array, dtype=np.float64)
    array = array.flatten() 
    if np.amin(array) < 0:
        array -= np.amin(array) 
    # Values must be sorted
    array += 0.0000001 # Prevent division by zero
    array = np.sort(array)
    index = np.arange(1,array.shape[0]+1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def main():
    parser = argparse.ArgumentParser(description='Calculate Library Collapse (Bottleneck)')
    parser.add_argument('--counts', default='A375.count.txt', help='Path to raw count file')
    args = parser.parse_args()

    # Load the raw counts
    df = pd.read_csv(args.counts, sep='\t')
    
    # We will analyze the first replicate of PLX for each timepoint
    timepoints = ['d30_T300_plx_R1', 'd41_T300_plx_R1', 'd50_T300_plx_R1']
    
    print("=" * 60)
    print(" T300 LIBRARY BOTTLENECK ANALYSIS")
    print("=" * 60)
    print(f"{'Timepoint':<20} | {'Sparsity (% Zeros)':<18} | {'Gini Coefficient'}")
    print("-" * 60)

    for tp in timepoints:
        if tp not in df.columns:
            print(f"Column {tp} not found!")
            continue
            
        counts = df[tp].values
        
        # Calculate Sparsity (% of guides with 0 reads)
        zero_count = np.sum(counts == 0)
        sparsity = (zero_count / len(counts)) * 100
        
        # Calculate Gini Coefficient
        gini = calculate_gini(counts)
        
        print(f"{tp:<20} | {sparsity:>15.2f}%   | {gini:>15.4f}")
        
    print("=" * 60)
    print("-> For ML training, you want the HIGHEST timepoint where Gini is STILL < 0.7")
    print("-> If Sparsity > 40%, the model will suffer from heavy class imbalance.")

if __name__ == "__main__":
    main()
