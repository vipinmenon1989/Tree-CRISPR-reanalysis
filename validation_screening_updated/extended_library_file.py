import pandas as pd
import subprocess
import pysam
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_CSV = "Horlbeck_CRISPRi_libaray.csv"
OUTPUT_CSV = "Horlbeck_CRISPRi_libaray_extended.csv"
BOWTIE_INDEX = "/local/projects-t3/lilab/vmenon/CRISPRi/Genome/hg38" # Prefix of your Bowtie 1 index
HG38_FASTA = "/local/projects-t3/lilab/vmenon/CRISPRi/Genome/hg38.fa" # Uncompressed hg38 reference genome

VALID_PAMS = {'AGG', 'CGG', 'GGG', 'TGG'}

def get_reverse_complement(seq):
    """Returns the reverse complement of a DNA sequence."""
    trans = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
    return seq.translate(trans)[::-1]

def process_crispr_data():
    print("1. Loading input data...")
    df = pd.read_csv(INPUT_CSV)
    
    # Check if necessary columns exist
    if 'protospacer sequence' not in df.columns:
        raise ValueError("Column 'protospacer sequence' missing from input CSV.")

    print("2. Generating temporary FASTA for Bowtie...")
    fasta_path = "temp_protospacers.fa"
    with open(fasta_path, "w") as f:
        for idx, row in df.iterrows():
            # Use dataframe index as the read name to merge back later
            f.write(f">{idx}\n{row['protospacer sequence']}\n")

    print("3. Aligning to hg38 using Bowtie...")
    sam_path = "temp_alignments.sam"
    # -v 0 (0 mismatches), -m 1 (unique alignments only), -S (SAM output), -f (FASTA input)
    bowtie_cmd = [
        "bowtie", "-f", "-v", "0", "-S", "-m", "1", 
        BOWTIE_INDEX, fasta_path
    ]
    
    with open(sam_path, "w") as sam_out:
        subprocess.run(bowtie_cmd, stdout=sam_out, check=True)

    print("4. Parsing alignments and extracting 30nt sequences...")
    hg38 = pysam.FastaFile(HG38_FASTA)
    samfile = pysam.AlignmentFile(sam_path, "r")
    
    alignment_data = []

    for read in samfile.fetch(until_eof=True):
        if read.is_unmapped:
            continue

        idx = int(read.query_name)
        chrom = read.reference_name
        pos = read.reference_start + 1  # Convert pysam 0-based to 1-based standard coordinate
        is_reverse = read.is_reverse

        if not is_reverse:
            # Plus strand mapping
            start = pos
            end = pos + 19
            start_30 = pos - 4
            end_30 = pos + 25
            
            try:
                # pysam fetch uses 0-based, half-open intervals [start, end)
                seq_30 = hg38.fetch(chrom, start_30 - 1, end_30).upper()
            except KeyError:
                continue
                
        else:
            # Minus strand mapping
            start = pos
            end = pos + 19
            start_30 = pos - 6
            end_30 = pos + 23
            
            try:
                ref_seq = hg38.fetch(chrom, start_30 - 1, end_30).upper()
                seq_30 = get_reverse_complement(ref_seq)
            except KeyError:
                continue

        if len(seq_30) != 30:
            continue

        # Extract PAM (positions 24, 25, 26 in the 30nt sequence)
        # Format: 4nt + 20nt + 3nt(PAM) + 3nt
        pam = seq_30[24:27]

        # Only retain if the PAM strictly matches the standard SpCas9 rules
        if pam in VALID_PAMS:
            alignment_data.append({
                'id': idx,
                'Chr': chrom,
                'Start': start,
                'End': end,
                'Strand': '-' if is_reverse else '+',
                'Start_30': start_30,
                'End_30': end_30,
                'Extended_sequence(30nt)': seq_30,
                'PAM': pam
            })

    # Close file handles
    samfile.close()
    hg38.close()

    print("5. Merging and saving final output...")
    # Convert results to DataFrame
    align_df = pd.DataFrame(alignment_data)
    
    if align_df.empty:
        print("No valid unique alignments with correct PAMs found.")
        return

    align_df.set_index('id', inplace=True)

    # Note: We replace the original 'Strand' column with the verified hg38 mapping strand,
    # and drop the obsolete hg19 'PAMcoordinate'.
    final_df = df.copy()
    if 'Strand' in final_df.columns:
        final_df = final_df.drop(columns=['Strand'])
    if 'PAMcoordinate' in final_df.columns:
        final_df = final_df.drop(columns=['PAMcoordinate'])

    # Join the alignment data back to the original dataframe using the index
    final_df = final_df.join(align_df, how='inner')

    # Reorder columns logically
    cols = ['Gene', 'transcript', 'Chr', 'Start', 'End', 'Strand', 
            'Start_30', 'End_30', 'Extended_sequence(30nt)', 'PAM', 
            'protospacer sequence', 'predicted score']
    
    final_df = final_df[cols]
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    # Cleanup temporary files
    os.remove(fasta_path)
    os.remove(sam_path)
    
    print(f"Complete. Processed data saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    process_crispr_data()
