import os
import sys
import pandas as pd

def check_shared_guides(file1_path, file2_path):
    print("======================================================================")
    print("SHARED GUIDE AUDIT: INTERSECTING ON SGRNA + GENE + CLASS")
    print("======================================================================\n")
    
    # 1. Verification
    if not os.path.exists(file1_path) or not os.path.exists(file2_path):
        print("CRITICAL ERROR: One or both input files do not exist.")
        sys.exit(1)
        
    # 2. Load Datasets (Handles both true CSVs or Tab-separated files automatically)
    def load_and_standardize(file_path):
        # Determine delimiter based on extension
        sep = '\t' if file_path.endswith('.txt') else ','
        df = pd.read_csv(file_path, sep=sep)
        
        # Standardize headers to handle casing anomalies
        rename_map = {}
        for col in df.columns:
            col_lower = col.lower()
            if col_lower == 'sgrna' or col_lower == 'id':
                rename_map[col] = 'sgrna'
            elif col_lower == 'gene':
                rename_map[col] = 'Gene'
            elif col_lower == 'class':
                rename_map[col] = 'class'
                
        df = df.rename(columns=rename_map)
        
        # Verify required keys exist
        required = ['sgrna', 'Gene', 'class']
        if not all(k in df.columns for k in required):
            print(f"CRITICAL ERROR: File {file_path} is missing required columns. Found: {list(df.columns)}")
            sys.exit(1)
            
        return df[['sgrna', 'Gene', 'class']]

    df1 = load_and_standardize(file1_path)
    df2 = load_and_standardize(file2_path)
    
    print(f"[*] File 1 unique rows: {len(df1)}")
    print(f"[*] File 2 unique rows: {len(df2)}")
    print("----------------------------------------------------------------------")
    
    # 3. Inner Join Intersection
    shared_df = pd.merge(df1, df2, on=['sgrna', 'Gene', 'class'], how='inner')
    
    total_shared = len(shared_df)
    shared_class_1 = (shared_df['class'] == 1).sum()
    shared_class_0 = (shared_df['class'] == 0).sum()
    
    # 4. Results Breakdown
    print("AUDIT RESULTS:")
    print(f"--> Total Rows Perfectly Shared: {total_shared}")
    print(f"    |-- Shared Functional Hits (Class 1): {shared_class_1}")
    print(f"    |-- Shared Structural Failures (Class 0): {shared_class_0}")
    print("======================================================================")

if __name__ == "__main__":
    # Update these filenames to your actual file names on the cluster
    file_one = "Output_File3_Balanced_ML_Matrix_K562.csv"
    file_two = "Output_File3_Balanced_ML_Matrix.csv"
    
    check_shared_guides(file_one, file_two)
