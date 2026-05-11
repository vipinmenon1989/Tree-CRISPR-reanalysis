import sys
import pandas as pd

def process_file():
    # Ensure the input file argument is provided
    if len(sys.argv) < 2:
        print("Error: Please provide the input CSV file path.")
        print("Usage: python filter_genes.py <input_file.csv>")
        sys.exit(1)

    input_file = sys.argv[1]

    # Load data
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        sys.exit(1)

    # Determine columns to check
    other_cols = [c for c in df.columns if c not in ['gene', 'ALKE_confidence']]

    # Filter where ALKE_confidence is not NaN and all other columns are NaN
    condition = df['ALKE_confidence'].notnull() & (df[other_cols].isnull().all(axis=1))

    filtered_df = df[condition].copy()
    filtered_df = filtered_df[['gene', 'ALKE_confidence']].rename(columns={'ALKE_confidence': 'ALKE'})

    # Save output
    filtered_df.to_csv('final_genes.csv', index=False)
    
    print("Process Complete. Results saved to final_genes.csv")
    print(filtered_df)

if __name__ == "__main__":
    process_file()
