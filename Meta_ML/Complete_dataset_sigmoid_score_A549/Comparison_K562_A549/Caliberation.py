import os
import sys
import pandas as pd

def run_diagnostic_consensus(k562_path, a549_path, output_path):
    print("======================================================================")
    print("DIAGNOSTIC CONSENSUS ENGINE: VERIFYING ROW COUNT PARADOX")
    print("======================================================================\n")
    
    # 1. Verify files exist
    if not os.path.exists(k562_path) or not os.path.exists(a549_path):
        print(f"CRITICAL ERROR: Check paths.\nK562: {k562_path}\nA549: {a549_path}")
        sys.exit(1)
        
    # 2. Ingest with explicit Tab-Separation for .txt files
    # Change sep=',' if you switch back to true comma-separated files
    print("[*] Loading raw dataframes using tab separator...")
    k562_raw = pd.read_csv(k562_path, sep='\t')
    a549_raw = pd.read_csv(a549_path, sep='\t')
    
    # CRITICAL AUDIT 1: Raw Python Row Counts
    print(f"[AUDIT] Python-read rows in K562 raw matrix: {len(k562_raw)}")
    print(f"[AUDIT] Python-read rows in A549 raw matrix: {len(a549_raw)}")
    print("----------------------------------------------------------------------")

    def standardize_and_audit(df, label):
        df_clean = df.copy()
        
        id_cols = [c for c in df_clean.columns if c.lower() == 'id']
        gene_cols = [c for c in df_clean.columns if c.lower() == 'gene']
        score_cols = [c for c in df_clean.columns if c.lower() == 'sigmoid_score']
        
        if not id_cols or not gene_cols or not score_cols:
            print(f"CRITICAL ERROR: Missing columns in {label}. Found columns: {list(df_clean.columns)[:5]}...")
            sys.exit(1)
            
        df_clean = df_clean.rename(columns={
            id_cols[0]: 'ID',
            gene_cols[0]: 'Gene',
            score_cols[0]: 'Sigmoid_Score'
        })
        
        # Calculate class assignment
        df_clean['class'] = (df_clean['Sigmoid_Score'] > 0.25).astype(int)
        
        # CRITICAL AUDIT 2: Inside-Python Duplicate Key Check
        dups = df_clean.duplicated(subset=['ID']).sum()
        print(f"[AUDIT] Duplicate 'ID' keys found inside Python for {label}: {dups}")
        
        return df_clean

    k562_df = standardize_and_audit(k562_raw, "K562")
    a549_df = standardize_and_audit(a549_raw, "A549")
    print("----------------------------------------------------------------------")

    # 3. Multi-Key Inner Merge
    print("[*] Executing inner join intersection...")
    consensus_df = pd.merge(
        a549_df, 
        k562_df[['ID', 'Gene', 'class']], 
        on=['ID', 'Gene', 'class'], 
        how='inner'
    )
    
    # 4. Save results
    consensus_df.drop(columns=['class']).to_csv(output_path, sep='\t', index=False)
    
    # 5. Final Metrics Output
    total_matches = len(consensus_df)
    print(f"\n[RESULTS] Final Invariant Rows Extracted: {total_matches}")
    print(f"[RESULTS] Rows dropped relative to original A549: {len(a549_raw) - total_matches}")
    print("======================================================================")

if __name__ == "__main__":
    # Point these EXACTLY to the text files you audited on your terminal
    k562_file = "ALKE_hits_complete_K562_Epigenetic_ID_merged.txt"
    a549_file = "ALKE_hits_complete_A549_Epigenetic_ID_merged.txt"
    output_file = "A549_validated_consensus_matrix.txt"
    
    run_diagnostic_consensus(k562_file, a549_file, output_file)
