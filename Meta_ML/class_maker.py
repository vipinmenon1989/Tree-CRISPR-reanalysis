import pandas as pd
import argparse
import sys

def append_class_vector(input_file):
    # 1. Load data with flexible delimiter engine
    try:
        # Auto-detects if file is tab-separated (.txt) or comma-separated (.csv)
        delim = '\t' if input_file.endswith('.txt') else ','
        df = pd.read_csv(input_file, sep=delim)
        print(f"Loaded file: {input_file} ({df.shape[0]} rows, {df.shape[1]} columns)")
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # 2. Clean column name white-spaces
    df.columns = df.columns.str.strip()

    if 'Sigmoid_Score' not in df.columns:
        print("CRITICAL ERROR: 'Sigmoid_Score' column was not found in the header.")
        sys.exit(1)

    # 3. Apply Threshold logic (1 if > 0.25, else 0)
    df['class'] = (df['Sigmoid_Score'] > 0.25).astype(int)

    # 4. Overwrite input file inline preservation
    try:
        df.to_csv(input_file, sep=delim, index=False)
        
        # Verify counts
        ones = (df['class'] == 1).sum()
        zeros = (df['class'] == 0).sum()
        print(f"Successfully updated matrix inline.")
        print(f"  --> Total Class 1 (>0.25): {ones}")
        print(f"  --> Total Class 0 (<=0.25): {zeros}")
    except Exception as e:
        print(f"Write failure: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inject Binary Class Vector')
    parser.add_argument('-i', '--input', required=True, help='Path to target file')
    args = parser.parse_args()
    append_class_vector(args.input)
