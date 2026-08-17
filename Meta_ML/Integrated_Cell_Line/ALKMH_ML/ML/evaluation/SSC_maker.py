import pandas as pd
import os

def extract_sgrna_sequence(input_file, output_file="Test_SSC.csv"):
    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: Input file '{input_file}' not found.")
        return

    print(f"[*] Loading dataset: {input_file}")
    df = pd.read_csv(input_file, sep=',')
    
    # Create a lowercase mapping to handle case-sensitivity issues
    col_map = {col.lower().strip(): col for col in df.columns}
    target_col = 'sgrna sequence'
    
    if target_col in col_map:
        exact_col_name = col_map[target_col]
        print(f"[*] Locating target column: '{exact_col_name}'")
        
        # Extract the specific column
        sgrna_data = df[exact_col_name]
        
        # Save to CSV without the index and without the header
        sgrna_data.to_csv(output_file, index=False, header=False)
        print(f"[-->] Successfully exported {len(sgrna_data)} sequences to {output_file} (No Header).")
    else:
        print(f"CRITICAL ERROR: 'sgRNA sequence' column not found in {input_file}.")
        print(f"Available columns: {list(df.columns)}")

if __name__ == "__main__":
    # Point this to your independent test file
    INPUT_MATRIX = "CRISPRi_ML_Holdout_Test_20_unseen_guides.csv" 
    
    extract_sgrna_sequence(
        input_file=INPUT_MATRIX, 
        output_file="Test_SSC.csv"
    )

