import os
import pandas as pd


def get_sort_key(file_name):
    """Extracts the prefix before the underscore to match the required order."""
    # Strip extension (e.g., 'SAM_hits.csv' -> 'SAM_hits')
    base_name = os.path.splitext(file_name)[0]

    # Extract the prefix before the first underscore (e.g., 'SAM_hits' -> 'SAM')
    prefix = base_name.split("_")[0]

    # Define your new required order of prefixes
    order_hierarchy = [
        "SAM",
        "T300",
        "TP",
    ]

    try:
        return order_hierarchy.index(prefix)
    except ValueError:
        # Fallback for any unexpected files to keep them at the bottom
        return len(order_hierarchy), file_name


def analyze_gene_csvs(folder_path):
    gene_column = "Gene"

    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    if not csv_files:
        print("No CSV files found in the specified folder.")
        return

    # Sort files using the updated SAM -> T300 -> TP hierarchy
    csv_files = sorted(csv_files, key=get_sort_key)

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
            df = pd.read_csv(file_path)
            row_count = len(df)
            total_rows_all_files += row_count

            if gene_column in df.columns:
                unique_genes_in_file = df[gene_column].dropna().unique()
                gene_count = len(unique_genes_in_file)
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
    print(
        f"{'TOTAL (Combined)':<30} | {total_rows_all_files:<18} | {len(all_unique_genes):<12}"
    )


# --- Execution ---
folder_target = "./"
analyze_gene_csvs(folder_target)
