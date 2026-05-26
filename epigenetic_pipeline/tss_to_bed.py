import sys
import pandas as pd
import os

def main():
    # Enforce command-line arguments
    if len(sys.argv) < 3:
        print("Error: Missing input/output arguments.")
        print("Usage: python convert_tss_to_bed.py <input_file.csv> <output_file.bed>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Validate file existence
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    try:
        # Load the CSV file
        df = pd.read_csv(input_file)
        
        # Standardize column names (strip whitespace)
        df.columns = [str(c).strip() for c in df.columns]

        # Verify necessary columns exist
        required_cols = ['gene', 'transcript', 'chromosome', 'strand', "Primary TSS, 5'", "Primary TSS, 3'"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
            print(f"Error: Missing required columns in input file: {missing_cols}")
            sys.exit(1)

        # Construct the name column combining gene and transcript
        df['name'] = df['gene'].astype(str) + "_" + df['transcript'].astype(str)
        df['score'] = 1000

        # Extract only the columns required for the BED file
        bed_df = df[[
            'chromosome',
            "Primary TSS, 5'",
            "Primary TSS, 3'",
            'name',
            'score',
            'strand'
        ]].copy()

        # Save to a tab-delimited BED file without header
        bed_df.to_csv(output_file, sep='\t', index=False, header=False)
        
        print(f"Process complete. Converted {len(bed_df)} rows to BED format and saved to '{output_file}'.")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
