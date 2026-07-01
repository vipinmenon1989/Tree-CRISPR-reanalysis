import os
import sys
import pandas as pd

def merge_ssc_scores():
    # ======================================================================
    # 1. CLUSTER PATH CONFIGURATIONS
    # ======================================================================
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Integrated_Cell_Line/Isolated_equation/Essential_unfied_gene/unseen_guide_no_gene_epi_geatures/evaluation/"
    
    # Target files
    prediction_file = os.path.join(working_dir, "independent_predictions.txt")
    ssc_file = os.path.join(working_dir, "SSC_score.out") # Adjust if your output file name differs
    
    if not os.path.exists(prediction_file):
        print(f"CRITICAL ERROR: Master prediction file missing: {prediction_file}")
        sys.exit(1)
    if not os.path.exists(ssc_file):
        print(f"CRITICAL ERROR: SSC scores file missing: {ssc_file}")
        sys.exit(1)

    # ======================================================================
    # 2. INGEST AND NORMALIZE TABLES
    # ======================================================================
    print("[*] Ingesting master prediction data...")
    # Read prediction file supporting flexible whitespace configurations
    df_pred = pd.read_csv(prediction_file, sep='\t')
    df_pred.columns = [col.strip() for col in df_pred.columns]

    print("[*] Ingesting SSC score vectors...")
    # Catch both standard CSV or tab/whitespace separated formats for SSC outputs
    try:
        df_ssc = pd.read_csv(ssc_file, sep=None, engine='python')
    except Exception:
        df_ssc = pd.read_csv(ssc_file, delim_whitespace=True)
        
    df_ssc.columns = [col.strip() for col in df_ssc.columns]

    # Verify tracking keys exist in both sets
    join_key = 'sgrna sequence'
    if join_key not in df_pred.columns or join_key not in df_ssc.columns:
        print(f"CRITICAL ERROR: Merge column '{join_key}' missing from headers!")
        print(f"Prediction headers: {list(df_pred.columns)}")
        print(f"SSC headers: {list(df_ssc.columns)}")
        sys.exit(1)

    # Convert sequences to uppercase to ensure a clean match map
    df_pred[join_key] = df_pred[join_key].astype(str).str.upper()
    df_ssc[join_key] = df_ssc[join_key].astype(str).str.upper()

    # Drop duplicate guide entries in SSC file if any exist to prevent data blowing up
    df_ssc = df_ssc.drop_duplicates(subset=[join_key])

    # ======================================================================
    # 3. EXECUTE STRUCTURAL MERGE
    # ======================================================================
    print("[*] Merging datasets on guide sequences...")
    # Left join ensures every single row in your original prediction file is strictly retained
    df_final = pd.merge(df_pred, df_ssc[[join_key, 'SSC']], on=join_key, how='left')

    # ======================================================================
    # 4. OVERWRITE DIRECTLY BACK TO PREDICTION.TXT
    # ======================================================================
    print(f"[*] Overwriting updated frame back to: {prediction_file}")
    df_final.to_csv(prediction_file, sep='\t', index=False)
    
    print(f"[-->] Completed successfully. 'SSC' column integrated into master roster.")

if __name__ == "__main__":
    merge_ssc_scores()
