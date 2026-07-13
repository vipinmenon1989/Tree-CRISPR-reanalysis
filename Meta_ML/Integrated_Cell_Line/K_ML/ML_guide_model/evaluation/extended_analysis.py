import os
import sys
import subprocess
import pandas as pd

def main():
    # ======================================================================
    # 1. CLUSTER PATH CONFIGURATIONS
    # ======================================================================
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Integrated_Cell_Line/Isolated_equation/Essential_unfied_gene/unseen_guide_no_gene_epi_geatures/evaluation/"
    
    # Target file to be updated in-place
    prediction_file = os.path.join(working_dir, "prediction.txt")
    
    # Standard Cluster Genomic Reference Indexes
    bowtie_index = "/local/projects-t3/lilab/vmenon/CRISPRi/Genome/hg38_large/hg38_large" 
    ref_genome_fasta = "/local/projects-t3/lilab/vmenon/CRISPRi/Genome/hg38.fa"   
    
    # Temporary cluster scratch files
    tmp_fa = os.path.join(working_dir, "tmp_extract.fa")
    tmp_sam = os.path.join(working_dir, "tmp_extract.sam")
    tmp_bed = os.path.join(working_dir, "tmp_extract.bed")
    tmp_out_fa = os.path.join(working_dir, "tmp_extracted_30nt.fa")

    if not os.path.exists(prediction_file):
        print(f"CRITICAL ERROR: Input prediction target file missing: {prediction_file}")
        sys.exit(1)

    print("[*] Reading existing prediction matrix...")
    df = pd.read_csv(prediction_file, sep='\t')
    df.columns = [col.lower().strip() for col in df.columns]

    # ======================================================================
    # 2. GENERATE FASTA DISK BUFFER FOR BOWTIE INGESTION
    # ======================================================================
    print("[*] Generating temporary FASTA file for alignment...")
    with open(tmp_fa, 'w') as f:
        for idx, row in df.iterrows():
            f.write(f">{row['unique_sgrna_id']}\n{row['sgrna sequence'].upper()}\n")

    # ======================================================================
    # 3. RUN BOWTIE GENOMIC MAPPING (-m 1 -v 0)
    # ======================================================================
    print("[*] Aligning guides uniquely against hg38 reference genome...")
    bowtie_cmd = f"bowtie -m 1 -v 0 -S -f {bowtie_index} {tmp_fa} {tmp_sam}"
    subprocess.run(bowtie_cmd, shell=True, check=True)

    # ======================================================================
    # 4. RESOLVE CO-ORDINATES AND SHIFT FOR STRAND-AWARE 30NT BOUNDARY
    # ======================================================================
    print("[*] Parsing alignment coordinates and processing flanking margins...")
    sam_columns = ['qname', 'flag', 'rname', 'pos', 'mapq', 'cigar', 'rnext', 'pnext', 'tlen', 'seq', 'qual']
    
    alignments = []
    with open(tmp_sam, 'r') as f:
        for line in f:
            if line.startswith('@'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 11:
                alignments.append(parts[:11])
                
    df_sam = pd.DataFrame(alignments, columns=sam_columns)
    df_sam['pos'] = df_sam['pos'].astype(int)
    df_sam['flag'] = df_sam['flag'].astype(int)
    df_sam['strand'] = df_sam['flag'].apply(lambda x: '-' if (x & 16) else '+')
    
    bed_records = []
    coord_map = {}
    
    for idx, row in df_sam.iterrows():
        chrom = row['rname']
        if chrom == '*': 
            continue # Drop unmapped vectors safely
            
        guide_len = len(row['seq'])
        
        # 0-indexed adjustment for BED format mapping
        if row['strand'] == '+':
            # 4nt upstream (subtract from start) | 6nt downstream (add to end)
            start_30nt = (row['pos'] - 1) - 4
            end_30nt = (row['pos'] - 1) + guide_len + 6
        else:
            # Reverse-strand logic inversion
            start_30nt = (row['pos'] - 1) - 6
            end_30nt = (row['pos'] - 1) + guide_len + 4
            
        bed_records.append([chrom, max(0, start_30nt), end_30nt, row['qname'], '.', row['strand']])
        
        # Cache specific coordinate arrays mapped by query identifier
        coord_map[row['qname']] = {
            'chromosome': chrom,
            'start': max(0, start_30nt),
            'end': end_30nt,
            'strand': row['strand']
        }

    df_bed = pd.DataFrame(bed_records, columns=['chrom', 'start', 'end', 'name', 'score', 'strand'])
    df_bed.to_csv(tmp_bed, sep='\t', index=False, header=False)

    # ======================================================================
    # 5. FETCH FLANKING SEQUENCES VIA BEDTOOLS
    # ======================================================================
    print("[*] Extracting 30nt structural windows via bedtools getfasta...")
    bedtools_cmd = f"bedtools getfasta -s -fi {ref_genome_fasta} -bed {tmp_bed} -fo {tmp_out_fa} -name"
    subprocess.run(bedtools_cmd, shell=True, check=True)

    # ======================================================================
    # 6. MAP EXTRACTED GEOMETRIES AND FASTA BACK TO ORIGINAL DATAFRAME
    # ======================================================================
    print("[*] Parsing fasta map and updating local matrix...")
    fasta_dict = {}
    with open(tmp_out_fa, 'r') as f:
        curr_id = ""
        for line in f:
            if line.startswith('>'):
                # Extract clean unique identifier name, dropping bedtools strand tags
                curr_id = line.strip().replace('>', '').split('::')[0]
            else:
                fasta_dict[curr_id] = line.strip().upper()

    # Append coordinate arrays dynamically via key matching
    df['chromosome'] = df['unique_sgrna_id'].apply(lambda x: coord_map.get(x, {}).get('chromosome', 'unmapped'))
    df['start'] = df['unique_sgrna_id'].apply(lambda x: coord_map.get(x, {}).get('start', -1))
    df['end'] = df['unique_sgrna_id'].apply(lambda x: coord_map.get(x, {}).get('end', -1))
    df['strand'] = df['unique_sgrna_id'].apply(lambda x: coord_map.get(x, {}).get('strand', '*'))
    df['prediction_values'] = df['unique_sgrna_id'].map(fasta_dict)

    # ======================================================================
    # 7. WRITE BACK DIRECTLY TO PREDICTION.TXT
    # ======================================================================
    print(f"[*] Overwriting updated predictions frame back to target file path...")
    df.to_csv(prediction_file, sep='\t', index=False)

    # Clean local scratch arrays to preserve storage structure
    for path in [tmp_fa, tmp_sam, tmp_bed, tmp_out_fa]:
        if os.path.exists(path):
            os.remove(path)
            
    print(f"[-->] Completed successfully. 30nt sequences and coordinates integrated directly into: {prediction_file}")

if __name__ == "__main__":
    main()
