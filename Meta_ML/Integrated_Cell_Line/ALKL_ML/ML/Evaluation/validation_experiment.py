import pandas as pd

def integrate_sgrna_data(file1_path, file2_path, output_path):
    # 1. Load datasets
    df1 = pd.read_csv(file1_path, sep='\t')
    df2 = pd.read_csv(file2_path, sep='\t')

    # 2. Strip whitespace from headers to ensure clean matching
    df1.columns = df1.columns.str.strip()
    df2.columns = df2.columns.str.strip()

    # 3. Dynamically find all common columns to use as exact merge keys
    # This automatically finds ['unique_sgrna_id', 'id', 'sgrna sequence', etc.] 
    # based strictly on what actually exists in both files.
    common_keys = list(df1.columns.intersection(df2.columns))
    print(f"Merging dynamically on these shared columns:\n{common_keys}\n")

    # 4. Execute the outer join
    merged_df = pd.merge(df1, df2, on=common_keys, how='outer')

    # 5. Save the final integrated file
    merged_df.to_csv(output_path, sep='\t', index=False)
    
    # Validation output
    print(f"File 1 rows: {len(df1)}")
    print(f"File 2 rows: {len(df2)}")
    print(f"Final Integrated rows: {len(merged_df)}")

# THE CALL MUST BE OUTSIDE THE FUNCTION
integrate_sgrna_data('Unified_GenomeWide_3Cellline_meta.txt', 'merged_sgRNA_data.txt', 'final_integrated_data.tsv')
