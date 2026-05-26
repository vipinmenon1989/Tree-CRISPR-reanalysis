import sys
import pandas as pd
import numpy as np

def main():
    if len(sys.argv) != 4:
        print("Usage: python step1_build_base.py <library_file.txt> <tss_file.csv> <output_base.txt>")
        sys.exit(1)

    lib_path = sys.argv[1]
    tss_path = sys.argv[2]
    out_path = sys.argv[3]

    # 1. Load the TSS Annotation File
    tss_df = pd.read_csv(tss_path)
    tss_df = tss_df.dropna(subset=["Primary TSS, 5'", "strand"])
    
    # Map each gene symbol to all its annotated alternative promoter positions
    tss_dict = {}
    for _, row in tss_df.iterrows():
        gene = row['gene']
        tss_pos = int(row["Primary TSS, 5'"])
        strand = row['strand']
        
        if gene not in tss_dict:
            tss_dict[gene] = []
        tss_dict[gene].append((tss_pos, strand))

    # 2. Load the sgRNA Library File
    lib_df = pd.read_csv(lib_path, sep='\t')

    # 3. Calculation Logic per Row
    def map_to_closest_promoter(row):
        gene = row['Gene']
        
        if gene not in tss_dict:
            return pd.Series({'distance_to_TSS': np.nan, 'closest_TSS_coord': np.nan, 'Gene_Strand': np.nan})
            
        sgrna_mid = (row['Start'] + row['End']) / 2.0
        candidate_results = []
        
        for tss_pos, tss_strand in tss_dict[gene]:
            if tss_strand == '+':
                dist = sgrna_mid - tss_pos
            else: # '-' strand
                dist = tss_pos - sgrna_mid
            
            candidate_results.append({
                'distance': int(dist),
                'coord': tss_pos,
                'strand': tss_strand
            })
            
        # Select the absolute closest promoter
        closest_match = min(candidate_results, key=lambda x: abs(x['distance']))
        
        return pd.Series({
            'distance_to_TSS': closest_match['distance'],
            'closest_TSS_coord': closest_match['coord'],
            'Gene_Strand': closest_match['strand']
        })

    # 4. Apply and append tracking columns
    print("Calculating closest TSS coordinates and distances...")
    calculated_cols = lib_df.apply(map_to_closest_promoter, axis=1)
    
    lib_df['distance_to_TSS'] = calculated_cols['distance_to_TSS'].astype('Int64')
    lib_df['closest_TSS_coord'] = calculated_cols['closest_TSS_coord'].astype('Int64')
    lib_df['Gene_Strand'] = calculated_cols['Gene_Strand']

    # 5. Export flat file
    lib_df.to_csv(out_path, sep='\t', index=False)
    print(f"Step 1 complete. Base matrix saved to: {out_path}")

if __name__ == "__main__":
    main()
