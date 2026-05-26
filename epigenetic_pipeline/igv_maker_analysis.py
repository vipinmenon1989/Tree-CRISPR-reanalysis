import sys
import pandas as pd

def main():
    if len(sys.argv) < 2:
        print("Error: Missing input matrix file.")
        print("Usage: python generate_igv_all_tracks.py <input_matrix.txt>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    # Dynamic output naming based on the input file, no hardcoded genes
    output_guides_bed = "genome_guides_target.bed"
    output_cov_bins_bed = "genome_bins_coverage.bed"
    output_pct_bins_bed = "genome_bins_percent.bed"
    
    print(f"Reading master matrix from: {input_file}")
    try:
        df = pd.read_csv(input_file, sep="\t")
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
        
    # Verify core columns exist
    required_cols = ["Chromosome", "Start", "End", "sgRNA Sequence", "distance_to_TSS", "Strand", "Gene", "closest_TSS_coord"]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}' in input file.")
            sys.exit(1)

    # --- 1. ALL SGRNA TARGET LOCATIONS (DYNAMICAL) ---
    print("Processing all guide RNA coordinates...")
    guides_bed = pd.DataFrame()
    guides_bed['chrom'] = df['Chromosome'].astype(str)
    guides_bed['chromStart'] = df['Start'].astype(int) - 1
    guides_bed['chromEnd'] = df['End'].astype(int)
    # Label each guide with both the Gene and its unique sequence for instant identification in IGV
    guides_bed['name'] = df['Gene'].astype(str) + "_" + df['sgRNA Sequence'].astype(str)
    guides_bed['score'] = df['distance_to_TSS'].fillna(0).astype(int)
    guides_bed['strand'] = df['Strand'].astype(str)
    
    guides_bed.to_csv(output_guides_bed, sep="\t", index=False, header=False)
    print(f"Generated Master Guide Targets Track: {output_guides_bed}")

    # --- 2. MULTI-GENE SEPARATE BINS TRACKS FOR COVERAGE AND PERCENT ---
    print("Extracting multi-locus regional epigenetic bins...")
    # Drop duplicates to extract exactly one set of 10 bins per unique gene promoter
    unique_tss = df.drop_duplicates(subset=['Gene', 'closest_TSS_coord'])
    
    cov_records = []
    pct_records = []
    bin_width = 200
    
    for _, row in unique_tss.iterrows():
        chrom = str(row['Chromosome'])
        tss = int(float(row['closest_TSS_coord']))
        gene_name = str(row['Gene'])
        
        # Center the 2kb macro window on the specific TSS of THIS current gene loop
        macro_start = tss - 1000 
        
        for i in range(1, 11):
            b_start = macro_start + (i - 1) * bin_width
            b_end = macro_start + i * bin_width
            
            cov_score = row[f'dna_methylation_coverage_bin_{i}']
            pct_score = row[f'dna_methylation_percent_bin_{i}']
            
            # Dynamic labels showing the specific Gene Name alongside the bin metrics
            cov_label = f"{gene_name}_Cov_Bin_{i}: {round(cov_score, 3)}"
            pct_label = f"{gene_name}_Pct_Bin_{i}: {round(pct_score, 3)}"
            
            # Format: Chrom, Start, End, Label, Score (int), Strand
            cov_records.append([chrom, b_start, b_end, cov_label, int(float(cov_score) * 10), "."])
            pct_records.append([chrom, b_start, b_end, pct_label, int(float(pct_score) * 100), "."])
            
    # Write out the complete multi-gene coverage matrix track
    pd.DataFrame(cov_records).to_csv(output_cov_bins_bed, sep="\t", index=False, header=False)
    print(f"Generated Master Coverage Bins Track: {output_cov_bins_bed}")
    
    # Write out the complete multi-gene percent matrix track
    pd.DataFrame(pct_records).to_csv(output_pct_bins_bed, sep="\t", index=False, header=False)
    print(f"Generated Master Percent Bins Track: {output_pct_bins_bed}")
    print(f"Successfully compiled visualization tracking for {len(unique_tss)} unique gene loci across the genome.")

if __name__ == "__main__":
    main()
