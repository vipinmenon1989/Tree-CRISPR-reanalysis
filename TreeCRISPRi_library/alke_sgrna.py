import sys
import pandas as pd
import os

def main():
    if len(sys.argv) < 3:
        print("Error: Missing input arguments.")
        print("Usage: python extract_final_sgrna.py <bed_sgnra_example_ALKE.csv> <second_file.txt>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    # Validate file existence
    if not os.path.exists(file1):
        print(f"Error: File '{file1}' not found.")
        sys.exit(1)
    if not os.path.exists(file2):
        print(f"Error: File '{file2}' not found.")
        sys.exit(1)

    try:
        # Load and process the first file
        df1 = pd.read_csv(file1)
        df1.columns = [str(c).strip() for c in df1.columns]

        # Locate identifiers in the first file
        f1_gene_col = 'Gene' if 'Gene' in df1.columns else 'gene'
        f1_id_col = 'ID' if 'ID' in df1.columns else 'id'

        # Load the second file based on its extension
        if file2.endswith('.txt') or file2.endswith('.tsv'):
            df2 = pd.read_csv(file2, sep='\t')
        else:
            df2 = pd.read_csv(file2)
        
        df2.columns = [str(c).strip() for c in df2.columns]

        # Locate identifiers in the second file
        f2_gene_col = 'Gene' if 'Gene' in df2.columns else 'gene'
        f2_id_col = 'ID' if 'ID' in df2.columns else 'id'

        # Merge on both columns
        merged_df = pd.merge(
            df2, 
            df1, 
            left_on=[f2_gene_col, f2_id_col], 
            right_on=[f1_gene_col, f1_id_col], 
            how='inner'
        )

        # Define output columns to extract from file2
        target_columns = [
            f2_id_col, 'sgRNA Sequence', f2_gene_col, 'Chromosome', 'Start', 'End', 
            'Strand', 'start_30', 'end_30', 'extended_sequence', 'PAM', 
            'distance_to_TSS'
        ]
        
        existing_columns = [c for c in target_columns if c in merged_df.columns]
        output_df = merged_df[existing_columns]

        # Standardize standard columns in output
        output_df = output_df.rename(columns={f2_id_col: 'ID', f2_gene_col: 'Gene'})

        # Save to output file
        output_filename = 'bed_sgnra_example.csv'
        output_df.to_csv(output_filename, index=False)
        
        print(f"Extraction complete. Saved {len(output_df)} rows to '{output_filename}'.")
        print(output_df.head())

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
