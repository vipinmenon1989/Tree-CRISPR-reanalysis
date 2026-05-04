import sys
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) != 4:
        print("Usage: python calc_tss_distance.py <library_file.txt> <tss_file.csv> <output_file.txt>")
        sys.exit(1)

    lib_path = sys.argv[1]
    tss_path = sys.argv[2]
    out_path = sys.argv[3]

    # 1. Load the TSS Annotation File
    # Using 'Primary TSS, 5'' as the exact start site coordinate.
    tss_df = pd.read_csv(tss_path)
    
    # Build a dictionary mapping each gene to a list of its TSS coordinates and strands.
    # Format: {'GeneA': [(pos1, '+'), (pos2, '+')], ...}
    tss_dict = {}
    for _, row in tss_df.iterrows():
        gene = row['gene']
        tss_pos = row["Primary TSS, 5'"]
        strand = row['strand']
        
        if pd.isna(tss_pos):
            continue
            
        if gene not in tss_dict:
            tss_dict[gene] = []
        tss_dict[gene].append((tss_pos, strand))

    # 2. Load the TreeCRISPRi Library File
    lib_df = pd.read_csv(lib_path, sep='\t')

    # 3. Define Distance Calculation Logic
    def get_closest_tss_distance(row):
        gene = row['Gene']
        
        # Return NaN if the gene is not present in the TSS annotation
        if gene not in tss_dict:
            return np.nan
            
        # Define sgRNA coordinate as the midpoint of its target sequence
        sgrna_mid = (row['Start'] + row['End']) / 2.0
        distances = []
        
        # Calculate distance to all annotated TSSs for this gene
        for tss_pos, tss_strand in tss_dict[gene]:
            if tss_strand == '+':
                dist = sgrna_mid - tss_pos
            else: # '-' strand
                dist = tss_pos - sgrna_mid
            distances.append(dist)
            
        # Select the distance with the smallest absolute value (closest TSS)
        closest_dist = min(distances, key=abs)
        
        # Return as integer coordinate distance
        return int(closest_dist)

    # 4. Apply Calculation
    lib_df['distance_to_TSS'] = lib_df.apply(get_closest_tss_distance, axis=1)

    # 5. Export to Output File
    # Keeping all original columns and formatting as tab-separated text
    lib_df.to_csv(out_path, sep='\t', index=False)
    print(f"Processing complete. Output saved to: {out_path}")

if __name__ == "__main__":
    main()
