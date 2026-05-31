import pandas as pd

def extract_and_diagnose_purged_targets(hits_file, epi_file, output_path="Purged_ALKE_sgRNAs_Diagnosed.csv"):
    # 1. Load the datasets
    df_hits = pd.read_csv(hits_file, sep=',')
    df_epi = pd.read_csv(epi_file, sep='\t')

    # 2. Identify the dropped rows by looking for IDs/Genes in Hits that aren't in the Inner Join
    # We use a left join with the indicator flag to spot the orphans
    merged_test = pd.merge(df_hits, df_epi[['ID', 'Gene']], on=['ID', 'Gene'], how='left', indicator=True)
    purged_df = merged_test[merged_test['_merge'] == 'left_only'].drop(columns=['_merge']).copy()

    # 3. Define the diagnostic logic for the drop
    def diagnose_drop(row):
        # Search for the isolated ID in the epigenetic database
        matching_id_in_epi = df_epi[df_epi['ID'] == row['ID']]
        
        if matching_id_in_epi.empty:
            # The ID was physically removed from the reference file
            return "ID missing entirely from epigenetic file (Filtered during reference generation)."
        else:
            # The ID exists, but the Gene string is different
            actual_epi_gene = matching_id_in_epi['Gene'].iloc[0]
            return f"Gene mismatch. Hits file = '{row['Gene']}', but Epigenetic file = '{actual_epi_gene}'."

    # 4. Apply the diagnostic logic to generate the 'Drop_Reason' column
    purged_df['Drop_Reason'] = purged_df.apply(diagnose_drop, axis=1)

    # 5. Export to CSV for inspection
    purged_df.to_csv(output_path, index=False)
    
    # 6. Console Output for immediate analytical review
    print(f"Total purged sgRNAs diagnosed: {len(purged_df)}")
    print("\nSummary of Drop Reasons:")
    print(purged_df['Drop_Reason'].value_counts().to_string())

# Execution
file_1 = 'ALKE_hits.csv'
file_2 = 'TreeCRISPRi_lib_merged_K562_epigenetic_feature.txt'
extract_and_diagnose_purged_targets(file_1, file_2)
