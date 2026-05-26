import subprocess
import os
import csv
import argparse

def map_sgrna_from_csv(csv_path, bowtie_index_prefix, output_tsv_path):
    """
    Reads a CSV, builds a temporary FASTA, executes Bowtie 1 alignment,
    computes 1-based coordinates, and writes the results to a TSV file.
    """
    fasta_file = "temp_sgrna.fasta"
    bowtie_output = "temp_bowtie.out"
    
    # Dictionaries to retain original metadata during the Bowtie execution
    gene_mapping = {}
    seq_mapping = {}

    # 1. Parse CSV and build FASTA file
    try:
        with open(csv_path, mode='r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            headers = [h.strip() for h in reader.fieldnames]
            
            id_col = next((h for h in headers if 'id' in h.lower()), headers[0])
            seq_col = next((h for h in headers if 'seq' in h.lower()), headers[1])
            gene_col = next((h for h in headers if 'gene' in h.lower()), headers[2] if len(headers) > 2 else None)
            
            with open(fasta_file, 'w') as f:
                for row in reader:
                    if not row.get(id_col) or not row.get(seq_col):
                        continue
                        
                    seq_id = row[id_col].strip()
                    seq = row[seq_col].strip()
                    gene = row[gene_col].strip() if gene_col and row.get(gene_col) else "N/A"
                    
                    # Store original data for final TSV construction
                    seq_mapping[seq_id] = seq
                    gene_mapping[seq_id] = gene
                    
                    f.write(f">{seq_id}\n{seq}\n")
                    
    except FileNotFoundError:
        print(f"Error: Input file '{csv_path}' not found.")
        return
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return

    # 2. Construct and execute the Bowtie command
    command = [
        "bowtie",
        "-v", "0",
        "-m", "1",
        "-f",
        "--suppress", "6,7,8",
        bowtie_index_prefix,
        fasta_file,
        bowtie_output
    ]

    print(f"Executing alignment: {' '.join(command)}")
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Bowtie execution failed.")
        print(f"Error details: {e.stderr}")
        return

    # 3. Parse Bowtie output and compute coordinates
    results = []
    if os.path.exists(bowtie_output):
        with open(bowtie_output, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 5:
                    seq_id = parts[0]
                    strand = parts[1]
                    chromosome = parts[2]
                    start_0_based = int(parts[3])
                    length = len(parts[4])
                    
                    # Translate 0-based to 1-based inclusive coordinates
                    start_coord = start_0_based + 1
                    end_coord = start_0_based + length
                    
                    # Retrieve original mapped data
                    gene = gene_mapping.get(seq_id, "N/A")
                    orig_seq = seq_mapping.get(seq_id, "N/A")

                    results.append([seq_id, orig_seq, gene, chromosome, start_coord, end_coord, strand])

    # 4. Write results to the final TSV file
    try:
        with open(output_tsv_path, 'w', newline='', encoding='utf-8') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t')
            # Write the required headers
            writer.writerow(["ID", "sgRNA Sequence", "Gene", "Chromosome", "Start", "End", "Strand"])
            # Write the data payload
            writer.writerows(results)
        print(f"Success: Mapping complete. Results saved to {output_tsv_path}")
    except Exception as e:
        print(f"Error writing to TSV file: {e}")

    # 5. Clean up temporary operational files
    if os.path.exists(fasta_file): os.remove(fasta_file)
    if os.path.exists(bowtie_output): os.remove(bowtie_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Map sgRNA sequences to a genome using Bowtie 1 and output to TSV.")
    
    parser.add_argument("-i", "--input", required=True, help="Path to input CSV file.")
    parser.add_argument("-x", "--index", required=True, help="Path and prefix to the Bowtie index.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the resulting TSV file.")
    
    args = parser.parse_args()
    
    map_sgrna_from_csv(args.input, args.index, args.output)
