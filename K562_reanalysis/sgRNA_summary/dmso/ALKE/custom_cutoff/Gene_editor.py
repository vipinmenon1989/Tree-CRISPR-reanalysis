import sys
import pandas as pd

def main():
    # 1. Check if the input file argument is provided
    if len(sys.argv) < 2:
        print("Error: Missing input file path.", file=sys.stderr)
        print("Usage: python script.py <input_file.csv>", file=sys.stderr)
        sys.exit(1)
        
    input_file = sys.argv[1]

    try:
        # 2. Load the dataset from command line argument
        df = pd.read_csv(input_file)
        
        # Filter: Only select rows where Sigmoid_Score is greater than 0.25
        df = df[df['Sigmoid_Score'] > 0.25]
        
        # 3. Compute and print summary metrics to stdout
        unique_genes = df['Gene'].dropna().unique()
        global_mean = df['Sigmoid_Score'].mean()

        print("--- SUMMARY METRICS (Sigmoid_Score > 0.25) ---")
        print(f"Total Unique Genes: {len(unique_genes)}")
        # Check to avoid division by zero errors if no rows pass the filter
        if len(df) > 0:
            print(f"Global Mean Sigmoid Score: {global_mean:.6f}\n")
        else:
            print("Global Mean Sigmoid Score: N/A (No rows passed threshold)\n")

        # 4. Create the structured DataFrame for export
        output_df = pd.DataFrame({
            'Gene': unique_genes,
            'ALKE': 1
        })

        # 5. Save to the requested CSV file
        output_df.to_csv('Gene_Editor_ALKE.csv', index=False)
        print("File successfully saved as: Gene_Editor_ALKE.csv")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
