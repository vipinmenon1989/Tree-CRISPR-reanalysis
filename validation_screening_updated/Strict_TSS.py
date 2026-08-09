import pandas as pd

def merge_and_filter_grna_tss(file1_path, file2_path, output_path):
    print("1. Loading datasets...")
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    print("2. Standardizing nomenclature in File 2...")
    df2 = df2.rename(columns={
        'gene': 'Gene',
        'chromosome': 'Chr',
        'strand': 'Strand'
    })
    
    print("3. Executing strict relational merge (Inner Join)...")
    # 'how=inner' discards any row in File 1 that does not have a matching key in File 2
    merged_df = pd.merge(
        df1, 
        df2, 
        on=['Gene', 'transcript', 'Chr', 'Strand'], 
        how='inner' 
    )
    
    print("4. Validating TSS data completeness...")
    # Drop rows where all TSS coordinate values are missing (NaN), 
    # ensuring only guides with actual TSS information remain.
    merged_df = merged_df.dropna(subset=[
        "Primary TSS, 5'", "Primary TSS, 3'", 
        "Secondary TSS, 5'", "Secondary TSS, 3'"
    ], how='all')
    
    print("5. Formatting output structure...")
    final_columns = [
        "Gene", "transcript", "Chr", "Start", "End", "Strand", 
        "Start_30", "End_30", "Extended_sequence(30nt)", "PAM", 
        "protospacer sequence", "predicted score", "TSS source", 
        "Primary TSS, 5'", "Primary TSS, 3'", 
        "Secondary TSS, 5'", "Secondary TSS, 3'"
    ]
    
    final_df = merged_df[final_columns]
    
    print("6. Writing to disk...")
    final_df.to_csv(output_path, index=False)
    print(f"Complete. Filtered and merged file saved to: {output_path}")

if __name__ == "__main__":
    # Replace these filenames with your actual file paths
    FILE1 = "Horlbeck_CRISPRi_libaray_extended.csv"
    FILE2 = "TSS_annotation_hg38.csv"
    OUTPUT = "merged_filtered_output.csv"
    
    merge_and_filter_grna_tss(FILE1, FILE2, OUTPUT)
