import os
import sys
import pandas as pd
import numpy as np
import RNA
from Bio.SeqUtils import MeltingTemp as mt
from scipy.stats import entropy

def shannon_entropy(seq):
    counts = [seq.count(base) for base in ['A', 'T', 'G', 'C']]
    return round(entropy(counts, base=2), 1)

def main():
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Complete_dataset_sigmoid_score_K562/"
    input_file = os.path.join(working_dir, "ALKE_hits_complete_K562_Epigenetic_ID_merged.txt")
    output_file = os.path.join(working_dir, "CRISPR_ml_features_final.csv")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Error: {input_file} not found in the target directory.")

    print("[*] Loading master epigenomic hits dataset...")
    df = pd.read_csv(input_file, sep='\t')

    # 1. Thermodynamics & Entropy Calculation (sgRNA Sequence - 20bp)
    sgrna_col = 'sgRNA Sequence'
    if sgrna_col not in df.columns:
        raise ValueError(f"Error: Required column '{sgrna_col}' missing from your raw text file.")

    print("[*] Compiling biophysical thermodynamic features (MFE, Tm, Entropy)...")
    df['MFE'] = df[sgrna_col].apply(lambda s: round(RNA.fold(s)[-1], 0))
    df['Tm'] = df[sgrna_col].apply(lambda s: mt.Tm_NN(s))
    df['entropy'] = df[sgrna_col].apply(shannon_entropy)

    # GC Logic
    df['GC_count'] = df[sgrna_col].apply(lambda s: s.count('G') + s.count('C'))
    df['GC_content'] = df.apply(lambda row: round((row['GC_count'] / len(row[sgrna_col])) * 100, 0), axis=1)
    df['gc_low'] = (df['GC_count'] < 10).astype(int)
    df['gc_high'] = (df['GC_count'] >= 10).astype(int)

    # 2. Sequence Geometry Generation (extended_sequence - 30bp)
    ext_col = 'extended_sequence'
    if ext_col not in df.columns:
        raise ValueError(f"Error: Required context column '{ext_col}' missing from your raw text file.")

    print("[*] Vectorizing mononucleotide and dinucleotide geometry matrices...")
    pos_data = {}

    # Positional One-Hot: 120 mononucleotide features (pos1_A to pos30_C)
    for i in range(30):
        for base in ['A', 'T', 'G', 'C']:
            col_name = f'pos{i+1}_{base}'
            pos_data[col_name] = df[ext_col].str[i] == base

    # Positional One-Hot: 464 dinucleotide features (di1_AA to di29_CC)
    dinucleotides = ['AA', 'AT', 'AG', 'AC', 'TA', 'TT', 'TG', 'TC', 'GA', 'GT', 'GG', 'GC', 'CA', 'CT', 'CG', 'CC']
    for i in range(29):
        for dibase in dinucleotides:
            col_name = f'di{i+1}_{dibase}'
            pos_data[col_name] = df[ext_col].str[i:i+2] == dibase

    pos_df = pd.DataFrame(pos_data).astype(int)

    # Global Base Distributions
    for base in ['A', 'T', 'G', 'C']:
        df[f'count_{base}'] = df[ext_col].str.count(base)

    for dibase in dinucleotides:
        di_cols = [f'di{i+1}_{dibase}' for i in range(29)]
        df[f'count_{dibase}'] = pos_df[di_cols].sum(axis=1)

    # ==================================================================
    # 3. UNIFIED MATRIX CONCATENATION (RETAINING GENE & SGRNA IDENTIFIERS)
    # ==================================================================
    # This explicit track list guarantees your keys survive the feature generation slice
    keep_cols = [
        'ID', 'Gene', 'sgRNA Sequence', 'Chromosome', 'Strand', 'PAM', 
        'extended_sequence', 'closest_TSS_coord', 'Sigmoid_Score', 
        'distance_to_TSS', 'class'
    ]
    
    keep_existing = [c for c in keep_cols if c in df.columns]
    bin_cols = [c for c in df.columns if '_bin_' in c]

    # Isolate original metadata + epigenetic tracks
    orig_kept_df = df[keep_existing + bin_cols].copy()

    # Isolate engineered structural/thermodynamic tracking blocks
    new_features = [
        'MFE', 'Tm', 'entropy', 'GC_count', 'GC_content', 'gc_low', 'gc_high',
        'count_A', 'count_T', 'count_G', 'count_C'
    ] + [f'count_{db}' for db in dinucleotides]

    new_features_df = pd.concat([df[new_features], pos_df], axis=1)

    # Unify layers cleanly into the final structural CSV
    final_df = pd.concat([orig_kept_df, new_features_df], axis=1)

    # Save output to file matrix
    final_df.to_csv(output_file, index=False)
    print(f"[-->] Success. Final file containing ID, Gene, and sgRNA Sequence saved to: {output_file}")

if __name__ == "__main__":
    main()
