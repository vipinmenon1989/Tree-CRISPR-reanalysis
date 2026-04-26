import pandas as pd
import argparse
import sys

def create_specific_output_files(sgrna_file, gene_filtered_file, control_label):
    try:
        df_sgrna = pd.read_csv(sgrna_file)
        df_gene = pd.read_csv(gene_filtered_file)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    # Clean headers to remove hidden spaces/newlines
    df_sgrna.columns = df_sgrna.columns.str.strip()
    df_gene.columns = df_gene.columns.str.strip()

    # Force identify the gene column in both files
    # Check for 'Gene_ID' or 'gene' or 'Gene' in the filtered file
    if 'Gene_ID' in df_gene.columns:
        df_gene = df_gene.rename(columns={'Gene_ID': 'gene'})
    elif 'Gene' in df_gene.columns:
        df_gene = df_gene.rename(columns={'Gene': 'gene'})
    
    if 'Gene' in df_sgrna.columns:
        df_sgrna = df_sgrna.rename(columns={'Gene': 'gene'})

    # Check if 'gene' exists now in both; if not, exit with clear info
    if 'gene' not in df_gene.columns or 'gene' not in df_sgrna.columns:
        print(f"ERROR: 'gene' column not found.")
        print(f"Filtered headers: {list(df_gene.columns)}")
        print(f"sgRNA headers: {list(df_sgrna.columns)}")
        sys.exit(1)

    requested_cols = ['sgrna', 'gene', 'Sigmoid_Score']

    # File 1: Hits Only
    df_merged = pd.merge(df_sgrna, df_gene, on='gene', how='inner')
    df_file1 = df_merged[requested_cols].copy()

    # File 2: Hits + NTC
    ntc_mask = df_sgrna['gene'].str.contains(control_label, case=False, na=False)
    df_ntc = df_sgrna[ntc_mask][requested_cols].copy()
    
    df_file2 = pd.concat([df_file1, df_ntc], axis=0, ignore_index=True)

    return df_file1, df_file2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sgrna", required=True)
    parser.add_argument("-f", "--filtered", required=True)
    parser.add_argument("-c", "--control", default="NO-TARGET")
    parser.add_argument("-o1", "--out1", default="File_1.csv")
    parser.add_argument("-o2", "--out2", default="File_2.csv")

    args = parser.parse_args()
    f1, f2 = create_specific_output_files(args.sgrna, args.filtered, args.control)
    f1.to_csv(args.out1, index=False)
    f2.to_csv(args.out2, index=False)

