import os
import pandas as pd

def merge_ssc_data():
    working_dir = "./"
    holdout_path = os.path.join(working_dir, "CRISPRi_ML_Holdout_Test_20_unseen_guides.csv")
    ssc_path = os.path.join(working_dir, "Test_SSC.out")
    output_path = os.path.join(working_dir, "Holdout_with_SSC_Scores.csv")

    # 1. Load Datasets
    df_holdout = pd.read_csv(holdout_path)
    df_ssc = pd.read_csv(ssc_path, sep='\s+') # Handles space or tab delimiters

    # 2. Force total normalization on the sequence columns
    # This strips whitespace and forces all characters to uppercase
    df_holdout['sgrna sequence'] = df_holdout['sgrna sequence'].astype(str).str.strip().str.upper()
    
    # Identify sequence and score columns in the .out file dynamically
    ssc_seq_col = [c for c in df_ssc.columns if 'seq' in c.lower()][0]
    ssc_score_col = [c for c in df_ssc.columns if 'ssc' in c.lower()][0]
    
    df_ssc[ssc_seq_col] = df_ssc[ssc_seq_col].astype(str).str.strip().str.upper()

    # 3. Merge
    df_merged = pd.merge(
        df_holdout,
        df_ssc[[ssc_seq_col, ssc_score_col]],
        left_on='sgrna sequence',
        right_on=ssc_seq_col,
        how='left'
    )

    # 4. Check for Empty Matches
    matched_count = df_merged[ssc_score_col].notna().sum()
    print(f"[*] Total rows in Holdout: {len(df_holdout)}")
    print(f"[*] Total successful matches found: {matched_count}")
    
    if matched_count == 0:
        print("CRITICAL: 0 matches found. Printing samples for manual inspection:")
        print(f"Holdout Seq Sample: '{df_holdout['sgrna sequence'].iloc[0]}'")
        print(f"SSC Seq Sample: '{df_ssc[ssc_seq_col].iloc[0]}'")
        return

    # 5. Select and Rename
    df_final = df_merged[['unique_sgrna_id', 'sgrna sequence', 'gene', 'class', ssc_score_col]].copy()
    df_final.columns = ['unique_sgrna_id', 'sgRNA sequence', 'gene', 'class', 'SSC']
    
    df_final.to_csv(output_path, index=False)
    print(f"[✔] Success! Saved {matched_count} matches to {output_path}")

if __name__ == "__main__":
    merge_ssc_data()

