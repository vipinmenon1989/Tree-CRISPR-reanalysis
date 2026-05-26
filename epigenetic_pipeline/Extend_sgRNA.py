import csv
import argparse
from pyfaidx import Fasta

def reverse_complement(seq):
    """Returns the reverse complement of a DNA sequence."""
    mapping = str.maketrans('ATCGNnatcg', 'TAGCNntagc')
    return seq.translate(mapping)[::-1]

def extend_coordinates_and_fetch_seq(input_tsv, genome_fasta, output_tsv):
    # Load the reference genome (pyfaidx creates an index automatically if missing)
    print(f"Loading reference genome: {genome_fasta}")
    fasta = Fasta(genome_fasta)
    
    with open(input_tsv, 'r', encoding='utf-8') as infile, \
         open(output_tsv, 'w', newline='', encoding='utf-8') as outfile:
         
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        
        # Read and modify headers
        headers = next(reader)
        
        # Find column indices based on standard naming
        chrom_idx = headers.index('Chromosome')
        start_idx = headers.index('Start')
        end_idx = headers.index('End')
        strand_idx = headers.index('Strand')
        
        # Append new headers
        new_headers = headers + ['start_30', 'end_30', 'extended_sequence', 'PAM']
        writer.writerow(new_headers)
        
        valid_pams = {'AGG', 'TGG', 'CGG', 'GGG'}
        
        for row in reader:
            if not row: continue
            
            chrom = row[chrom_idx]
            start = int(row[start_idx])
            end = int(row[end_idx])
            strand = row[strand_idx]
            
            # Ensure chromosome exists in FASTA
            if chrom not in fasta:
                print(f"Warning: {chrom} not found in genome fasta. Skipping row.")
                continue
            
            if strand == '+':
                start_30 = start - 4
                end_30 = end + 6
                # pyfaidx uses 0-based exclusive for slicing. 
                # Input Start is 1-based. So 1-based start_30 requires (start_30 - 1)
                ext_seq = fasta[chrom][start_30 - 1 : end_30].seq.upper()
                
            elif strand == '-':
                start_30 = start - 6
                end_30 = end + 4
                raw_seq = fasta[chrom][start_30 - 1 : end_30].seq.upper()
                ext_seq = reverse_complement(raw_seq)
            else:
                continue

            # Handle edge case where sequence hits end of chromosome and is < 30nt
            if len(ext_seq) < 30:
                ext_seq = ext_seq.ljust(30, 'N')
            
            # Extract PAM precisely at Python indices 24, 25, 26
            pam = ext_seq[24:27]
            
            row.extend([str(start_30), str(end_30), ext_seq, pam])
            writer.writerow(row)
            
    print(f"Extraction complete. Output saved to {output_tsv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extend mapped sgRNA coordinates and fetch 30nt sequences.")
    parser.add_argument("-i", "--input", required=True, help="Input TSV file from Bowtie mapping.")
    parser.add_argument("-f", "--fasta", required=True, help="Path to the reference genome FASTA file.")
    parser.add_argument("-o", "--output", required=True, help="Output TSV file path.")
    
    args = parser.parse_args()
    extend_coordinates_and_fetch_seq(args.input, args.fasta, args.output)
