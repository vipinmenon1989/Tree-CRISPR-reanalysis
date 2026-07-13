import pandas as pd
import numpy as np
import RNA
from Bio.SeqUtils import MeltingTemp as mt
from scipy.stats import entropy
import os

def shannon_entropy(seq):
    counts = [seq.count(base) for base in ['A', 'T', 'G', 'C']]
    return round(entropy(counts, base=2), 1)

def main():
    input_file = "K_A375_epigenetic_scaled.txt"
    output_file = "CRISPR_ml_features_final.csv"
    
    # Load input dataset
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Error: {input_file} not found in the current working directory.")
        
    df = pd.read_csv(input_file, sep='\t')
    
    # --- DIMENSIONAL TRACKING: START ---
    print("\n==================================================")
    print(f"INITIAL DATASET SHAPE: {df.shape[0]} rows, {df.shape[1]} columns")
    print("==================================================\n")
    
    # --- LABEL GENERATION BLOCK ---
    if 'Sigmoid_Score' not in df.columns:
        raise ValueError("Error: Required column 'Sigmoid_Score' not found in the dataset.")
        
    print("Generating binary 'class' labels (1 if Sigmoid_Score > 0.25, else 0)...")
    df['class'] = (df['Sigmoid_Score'] > 0.25).astype(int)
    
    print(f"Shape after Label Generation: {df.shape[0]} rows, {df.shape[1]} columns\n")
    # ------------------------------
    
    # 1. Thermodynamics & Entropy (sgRNA Sequence - 20bp)
    print("Calculating thermodynamic features...")
    sgrna_col = 'sgRNA Sequence'
    if sgrna_col not in df.columns:
        raise ValueError(f"Error: Required column '{sgrna_col}' not found in the dataset.")
        
    df['MFE'] = df[sgrna_col].apply(lambda s: round(RNA.fold(s)[-1], 0))
    df['Tm'] = df[sgrna_col].apply(lambda s: mt.Tm_NN(s))
    df['entropy'] = df[sgrna_col].apply(shannon_entropy)
    
    # GC Logic
    df['GC_count'] = df[sgrna_col].apply(lambda s: s.count('G') + s.count('C'))
    df['GC_content'] = df.apply(lambda row: round((row['GC_count'] / len(row[sgrna_col])) * 100, 0), axis=1)
    df['gc_low'] = (df['GC_count'] < 10).astype(int)
    df['gc_high'] = (df['GC_count'] >= 10).astype(int)
    
    # 2. Sequence Geometry (extended_sequence - 30bp)
    print("Generating positional and sequence geometry features...")
    ext_col = 'extended_sequence'
    if ext_col not in df.columns:
        raise ValueError(f"Error: Required column '{ext_col}' not found in the dataset.")
        
    pos_data = {}
    
    # Positional One-Hot: 120 mononucleotide columns (pos1_A, etc. up to position 30)
    for i in range(30):
        for base in ['A', 'T', 'G', 'C']:
            col_name = f'pos{i+1}_{base}'
            pos_data[col_name] = df[ext_col].str[i] == base
            
    # Positional One-Hot: 464 dinucleotide columns (di1_AA, etc. up to position 29)
    dinucleotides = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
    for i in range(29):
        for dibase in dinucleotides:
            col_name = f'di{i+1}_{dibase}'
            pos_data[col_name] = df[ext_col].str[i:i+2] == dibase
            
    pos_df = pd.DataFrame(pos_data).astype(int)
    
    # Global Counts (extended_sequence)
    for base in ['A', 'T', 'G', 'C']:
        df[f'count_{base}'] = df[ext_col].str.count(base)
        
    for dibase in dinucleotides:
        di_cols = [f'di{i+1}_{dibase}' for i in range(29)]
        df[f'count_{dibase}'] = pos_df[di_cols].sum(axis=1)
        
    # 3. Data Merging and Output Formatting
    print("Merging feature matrices...")
    keep_cols = [
        'ID', 'sgRNA Sequence', 'Gene', 'Chromosome', 'Start', 'End', 'Strand',
        'start_30', 'end_30', 'extended_sequence', 'PAM', 'distance_to_TSS',
        'closest_TSS_coord', 'Gene_Strand', 'Sigmoid_Score', 'class'
    ]
    
    keep_existing = [c for c in keep_cols if c in df.columns]
    bin_cols = [c for c in df.columns if '_bin_' in c]
    
    orig_kept_df = df[keep_existing + bin_cols]
    
    new_features = [
        'MFE', 'Tm', 'entropy', 'GC_count', 'GC_content', 'gc_low', 'gc_high',
        'count_A', 'count_T', 'count_G', 'count_C'
    ] + [f'count_{db}' for db in dinucleotides]
    
    new_features_df = pd.concat([df[new_features], pos_df], axis=1)
    
    # Unify into a single flat dataframe
    final_df = pd.concat([orig_kept_df, new_features_df], axis=1)
    
    # --- DIMENSIONAL TRACKING: FINAL ---
    print("\n==================================================")
    print(f"FINAL DATASET SHAPE: {final_df.shape[0]} rows, {final_df.shape[1]} columns")
    print("==================================================\n")
    
    # Save output to CSV
    final_df.to_csv(output_file, index=False)
    print(f"Final feature matrix saved to {output_file}")

if __name__ == "__main__":
    main()
