import sys
import pandas as pd

def main():
    # Enforce argument validation
    if len(sys.argv) < 3:
        print("Error: Please provide two file paths.")
        print("Usage: python merge_genes.py <final_genes.csv> <ALKE_confidence.csv>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    except FileNotFoundError as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Perform an inner join on the 'gene' column
    # This keeps only the genes present in both files.
    merged_df = pd.merge(df1, df2, on='gene', how='inner')

    # Extract the requested columns ('gene', 'sgrna')
    try:
        output_df = merged_df[['gene', 'sgrna']]
    except KeyError as e:
        print(f"Missing column in the provided dataframes: {e}")
        sys.exit(1)

    # Save to the output file
    output_df.to_csv('bed_sgnra_example.csv', index=False)
    
    print("Process complete. Merged data saved to 'bed_sgnra_example.csv'.")
    print(output_df)

if __name__ == '__main__':
    main()
