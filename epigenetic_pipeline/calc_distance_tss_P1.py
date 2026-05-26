import sys
import pandas as pd
import numpy as np
import os

def main():
    if len(sys.argv) != 4:
        print("Usage: python calc_tss_distance.py <library_file.txt> <tss_file.csv> <output_file.txt>")
        sys.exit(1)

    lib_path = sys.argv[1]
    tss_path = sys.argv[2]
    out_path = sys.argv[3]

    # Validate file existence
    if not os.path.exists(lib_path):
        print(f"Error: File '{lib_path}' not found.")
        sys.exit(1)
    if not os.path.exists(tss_path):
        print(f"Error: File '{tss_path}' not found.")
        sys.exit(1)

    try:
        # 1. Load the TSS Annotation File
        tss_df = pd.read_csv(tss_path)
        tss_df.columns = [str(c).strip() for c in tss_df.columns]

        # Explicitly filter for P1 transcripts
        t_col = 'transcript' if 'transcript' in tss_df.columns else 'Transcript'
        if t_col in tss_df.columns:
            tss_df = tss_df[tss_df[t_col] == 'P1'].copy()
            print("Filtered TSS annotations to primary transcripts (P1).")
        else:
            print("Warning: Transcript column not found. Defaulting to first occurrence per gene.")
            tss_df = tss_df.drop_duplicates(subset=['gene'])

        # Build a dictionary mapping each gene to its P1 TSS coordinate and strand.
        tss_dict = {}
        for _, row in tss_df.iterrows():
            gene = row['gene']
            tss_pos = row["Primary TSS, 5'"]
            strand = row['strand']
            
            if pd.isna(tss_pos):
                continue
                
            tss_dict[gene] = (tss_pos, strand)

        # 2. Load the Library File
        lib_df = pd.read_csv(lib_path, sep='\t')
        lib_df.columns = [str(c).strip() for c in lib_df.columns]

        # 3. Define Distance Calculation Logic (P1-specific)
        def get_p1_tss_distance(row):
            gene = row['Gene']
            
            # Return NaN if the gene is not present in the P1 TSS annotation
            if gene not in tss_dict:
                return np.nan
                
            # Define sgRNA coordinate as the midpoint of its target sequence
            sgrna_mid = (row['Start'] + row['End']) / 2.0
            
            tss_pos, tss_strand = tss_dict[gene]
            
            if tss_strand == '+':
                dist = sgrna_mid - tss_pos
            else: # '-' strand
                dist = tss_pos - sgrna_mid
                
            return int(dist)

        # 4. Apply Calculation
        lib_df['distance_to_TSS'] = lib_df.apply(get_p1_tss_distance, axis=1)

        # 5. Export to Output File
        lib_df.to_csv(out_path, sep='\t', index=False)
        print(f"Processing complete. P1-specific output saved to: {out_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
