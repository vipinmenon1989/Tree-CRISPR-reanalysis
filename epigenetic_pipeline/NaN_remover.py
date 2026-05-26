import sys
import os
import pandas as pd

def main():
    # Force strict checking of command-line arguments
    if len(sys.argv) < 2:
        print("Error: Missing input matrix file.")
        print("Usage: python clean_matrix_nans.py <matrix_file.txt>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    
    # Verify the file actually exists before processing
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' does not exist.")
        sys.exit(1)
        
    print(f"Loading matrix file: {input_file}")
    try:
        # Read the tab-separated file natively
        df = pd.read_csv(input_file, sep="\t")
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
        
    # Capture initial row/column dimensions for verification
    rows, cols = df.shape
    print(f"Matrix loaded successfully. Dimensions: {rows} rows, {cols} columns.")
    
    print("Sweeping matrix and substituting all NaN/blank entries with 0.0...")
    # Globally replace all floating-point NaNs, literal strings, and blank spaces
    df.fillna(0.0, inplace=True)
    
    print(f"Overwriting clean data directly back to: {input_file}")
    try:
        # Save back to the exact same input file path, stripping out pandas automatic indexing
        df.to_csv(input_file, sep="\t", index=False)
        print("Success: Matrix is now 100% insulated from missing values.")
    except Exception as e:
        print(f"Error writing updates back to file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
