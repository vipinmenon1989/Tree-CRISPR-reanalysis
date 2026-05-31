import pandas as pd

def process_and_diagnose_crispri_merge(hits_file, epi_file, merged_output, diagnostic_output):
    # 1. Load the datasets with correct delimiters
    df_hits = pd.read_csv(hits_file, sep=',')
    df_epi = pd.read_csv(epi_file, sep='\t')

    # Define the common identifiers for mapping
    common_columns = ['ID', 'Gene']

    # ---------------------------------------------------------
    # PART A: Generate the Final Merged File (Common Hits Only)
    # ---------------------------------------------------------
    # The inner join acts as a strict filter, dropping incomplete data
    merged_df = pd.merge(df_hits, df_epi, on=common_columns, how='inner')
    merged_df.to_csv(merged_output, sep='\t', index=False)
    
    # ---------------------------------------------------------
    # PART B: Isolate and Diagnose the Purged Targets
    # ---------------------------------------------------------
    # Use a left join with an indicator to spot hits that failed the inner join
    test_merge = pd.merge(df_hits, df_epi[['ID', 'Gene']], on=common_columns, how='left', indicator=True)
    purged_df = test_merge[test_merge['_merge'] == 'left_only'].drop(columns=['_merge']).copy()

    # Define diagnostic logic for dropped rows
    def diagnose_drop(row):
        matching_id = df_epi[df_epi['ID'] == row['ID']]
        if matching_id.empty:
            return "ID missing entirely from epigenetic file (Filtered during reference generation)."
        else:
            actual_epi_gene = matching_id['Gene'].iloc[0]
            return f"Gene mismatch. Hits file = '{row['Gene']}', Epigenetic file = '{actual_epi_gene}'."

    # Apply diagnostics and save the secondary file
    purged_df['Drop_Reason'] = purged_df.apply(diagnose_drop, axis=1)
    purged_df.to_csv(diagnostic_output, index=False)

    # ---------------------------------------------------------
    # PART C: Output Analytical Metrics
    # ---------------------------------------------------------
    print(f"--- Process Complete ---")
    print(f"Input Hits: {len(df_hits)} rows")
    print(f"Input Epigenetics: {len(df_epi)} rows")
    print(f"-> SUCCESS: Saved {len(merged_df)} validated targets to {merged_output}")
    print(f"-> DIAGNOSTIC: Saved {len(purged_df)} purged targets to {diagnostic_output}\n")
    
    print("Purge Breakdown:")
    if not purged_df.empty:
        print(purged_df['Drop_Reason'].value_counts().to_string())
    else:
        print("No data dropped. 100% overlap achieved.")

# --- Execution ---
hits_csv = 'ALKE_hits.csv'
epi_txt = 'TreeCRISPRi_lib_merged_A549.txt'

# Output configurations
merged_out = 'ALKE_epigenetic_merged_common_only.txt'
diag_out = 'Purged_ALKE_sgRNAs_Diagnosed.csv'

process_and_diagnose_crispri_merge(hits_csv, epi_txt, merged_out, diag_out)
