import pandas as pd

def extract_validation_genes(file1_path, file2_path, output_path):
    print("1. Loading File 1 (CSV)...")
    df1 = pd.read_csv(file1_path)
    
    print("2. Loading File 2 (TSV)...")
    df2 = pd.read_csv(file2_path, sep='\t')
    
    print("3. Filtering data...")
    # Extract rows matching the target genes and create an independent copy
    filtered_df = df1[df1['Gene'].isin(df2['gene'])].copy()
    
    print(f"Extraction complete. Found {len(filtered_df)} matching rows.")
    
    print("4. Generating unique IDs...")
    # Generate sequential identifiers: sgrna_1, sgrna_2, ...
    id_list = [f"sgrna_{i+1}" for i in range(len(filtered_df))]
    
    # Insert the 'ID' column at position 0 (the first column)
    filtered_df.insert(0, 'ID', id_list)
    
    print("5. Writing to output file...")
    # Save the result, maintaining comma-separated structure
    filtered_df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")

if __name__ == "__main__":
    # Define file paths
    FILE1 = "Horlbeck_Libarary_TSS.csv"
    FILE2 = "gene-to-test.txt"
    OUTPUT = "gene-test-validation-analysis.txt"
    
    extract_validation_genes(FILE1, FILE2, OUTPUT)
