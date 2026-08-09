
import pandas as pd

def extract_ml_columns(input_file_path, output_file_path):
    print("1. Loading source file...")
    # Read the dataset containing all columns
    df = pd.read_csv(input_file_path)
    
    print("2. Extracting target columns...")
    # Define the exact headers to retain
    target_columns = ['ID', 'Extended_sequence(30nt)', 'protospacer sequence']
    
    # Slice the dataframe to keep only the target columns
    ml_df = df[target_columns]
    
    print("3. Writing to output file...")
    # Save to the new CSV file without the pandas integer index
    ml_df.to_csv(output_file_path, index=False)
    
    print(f"Extraction complete. File saved to: {output_file_path}")

if __name__ == "__main__":
    # Replace INPUT_FILE with the actual name of your file containing the full dataset
    INPUT_FILE = "gene-test-validation-analysis.txt" 
    OUTPUT_FILE = "processor-ml-file.csv"
    
    extract_ml_columns(INPUT_FILE, OUTPUT_FILE)
