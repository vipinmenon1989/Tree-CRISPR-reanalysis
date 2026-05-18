import os
import pandas as pd


def analyze_gene_csvs(folder_path):
    # Target column name
    gene_column = "Gene"

    # Verify folder existence
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # List all CSV files in the directory
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    print(f"Found {len(csv_files)} CSV file(s). Processing...\n")
    print(
        f"{'File Name':<30} | {'Total Rows (Data)':<18} | {'Unique Genes':<12}"
    )
    print("-" * 68)

    total_rows_all_files = 0
    all_unique_genes = set()

    for file_name in csv_files:
        file_path = os.path.join(folder_path, file_name)

        try:
            # Read only the 'Gene' column if you only care about genes to save memory,
            # or read the whole file if you need to ensure row counts represent all data.
            # We load the whole file to accurately count total data rows.
            df = pd.read_csv(file_path)

            # 1. Total rows in this file
            row_count = len(df)
            total_rows_all_files += row_count

            # 2. Unique genes in this file
            if gene_column in df.columns:
                # Drop NaN values and get unique values
                unique_genes_in_file = df[gene_column].dropna().unique()
                gene_count = len(unique_genes_in_file)

                # Update the global set for cross-file deduplication
                all_unique_genes.update(unique_genes_in_file)
            else:
                gene_count = 0
                print(
                    f"Warning: '{gene_column}' column not found in {file_name}"
                )

            print(f"{file_name:<30} | {row_count:<18} | {gene_count:<12}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    print("-" * 68)
    print(f"{'TOTAL (Combined)':<30} | {total_rows_all_files:<18} | {len(all_unique_genes):<12}")


# --- Execution ---
# Replace 'your_folder_path_here' with the actual path to your folder
# Example: r"C:\Users\Name\Documents\GeneData" or "./data"
folder_target = "./"
analyze_gene_csvs(folder_target)
