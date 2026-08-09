import pandas as pd
import sys

def filter_and_format_data(input_path, output_path):
    print(f"[*] Loading data from {input_path}...")
    try:
        # Load the original CSV
        df = pd.read_csv(input_path, sep=',')
    except FileNotFoundError:
        print(f"CRITICAL ERROR: File '{input_path}' not found.")
        sys.exit(1)

    # 1. Map existing column names to your new desired names
    rename_mapping = {
        'unique_sgrna_id': 'id',
        'probability_score': 'prediction_probability'
    }
    df = df.rename(columns=rename_mapping)

    # 2. Define the exact order of the columns to retain
    target_columns = [
        'sgrna sequence', 
        'gene', 
        'id', 
        'class', 
        'prediction_probability', 
        'prediction_binary'
    ]

    # Verify all required columns exist before filtering to prevent KeyErrors
    missing_cols = [col for col in target_columns if col not in df.columns]
    if missing_cols:
        print(f"CRITICAL ERROR: Missing columns in the dataset after renaming: {missing_cols}")
        sys.exit(1)

    # 3. Isolate the required columns
    df_filtered = df[target_columns]

    # 4. Export the file. 
    # Using sep='\t' (tab-separated) based on the spacing in your requested output.
    # Change to sep=',' if you need a standard CSV.
    print(f"[*] Exporting filtered data to {output_path}...")
    df_filtered.to_csv(output_path, sep='\t', index=False)
    
    print(f"[-->] Operation complete. Shape of new file: {df_filtered.shape[0]} rows x {df_filtered.shape[1]} columns.")

if __name__ == "__main__":
    # Replace these variables with your actual file paths
    INPUT_FILE = "independent_predictions.csv"
    OUTPUT_FILE = "Combined_Prediction_Metadata_Result.txt" 
    
    filter_and_format_data(INPUT_FILE, OUTPUT_FILE)
