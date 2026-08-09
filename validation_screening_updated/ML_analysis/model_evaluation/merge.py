import pandas as pd
import os
import sys

def main():
    # ==========================================
    # 1. DEFINE FILE PATHS
    # ==========================================
    # Update these strings to match the exact names of your files in the directory
    file1_path = "multimodel_predictions_effectiveness.csv"  # File with the probabilities
    file2_path = "gene-test-validation-analysis.csv"                                 # File with Gene, transcript, etc.
    output_path = "Xiaofeng_validation_list.csv"

    # ==========================================
    # 2. VALIDATE FILE EXISTENCE
    # ==========================================
    if not os.path.exists(file1_path):
        print(f"CRITICAL ERROR: File 1 '{file1_path}' not found.")
        sys.exit(1)
        
    if not os.path.exists(file2_path):
        print(f"CRITICAL ERROR: File 2 '{file2_path}' not found.")
        sys.exit(1)

    # ==========================================
    # 3. INGEST DATA
    # ==========================================
    print(f"[*] Loading prediction matrix from: {file1_path}")
    df1 = pd.read_csv(file1_path)
    
    print(f"[*] Loading metadata matrix from: {file2_path}")
    df2 = pd.read_csv(file2_path)

    # Ensure the common key 'ID' exists in both dataframes
    if 'ID' not in df1.columns:
        print(f"CRITICAL ERROR: 'ID' column missing in {file1_path}")
        sys.exit(1)
        
    if 'ID' not in df2.columns:
        print(f"CRITICAL ERROR: 'ID' column missing in {file2_path}")
        sys.exit(1)

    # ==========================================
    # 4. EXECUTE MERGE
    # ==========================================
    print("[*] Merging datasets on 'ID' key...")
    # Using 'inner' guarantees we only keep rows where the ID exists in both files.
    # We place df2 first so the metadata columns appear before the prediction columns.
    merged_df = pd.merge(df2, df1, on='ID', how='inner')

    # ==========================================
    # 5. EXPORT FINAL MATRIX
    # ==========================================
    print(f"[*] Exporting merged dataset to {output_path}...")
    merged_df.to_csv(output_path, index=False)
    
    print("\n" + "="*65)
    print("    MERGE COMPLETE")
    print("="*65)
    print(f" -> Input 1 shape: {df1.shape[0]} rows x {df1.shape[1]} columns")
    print(f" -> Input 2 shape: {df2.shape[0]} rows x {df2.shape[1]} columns")
    print(f" -> Final merged shape: {merged_df.shape[0]} rows x {merged_df.shape[1]} columns")
    print(f" -> Saved successfully to: {output_path}")
    print("="*65)

if __name__ == "__main__":
    main()
