import pandas as pd

# 1. Load the data
df1 = pd.read_csv("Comprehensive_Validation_Set.txt", sep='\t')
df2 = pd.read_csv("TreeCRISPR_lib_TSS.txt", sep='\t')

# 2. Standardize keys for consistent merging
key_cols = ['id', 'sgrna sequence', 'gene']
for col in key_cols:
    df1[col] = df1[col].astype(str).str.strip()
    df2[col] = df2[col].astype(str).str.strip()

# 3. Perform the Merge
# Using 'left' ensures we keep all model predictions from file1
# Joining on the three-part composite key
merged_df = pd.merge(df1, df2, on=key_cols, how='left')

# 4. Filter for desired output columns
target_cols = [
    'sgrna sequence', 'gene', 'id', 'prediction_probability', 
    'prediction_binary', 'Chromosome', 'Start', 'End', 'Strand', 
    'extended_sequence', 'PAM', 'closest_TSS_coord', 'gene_status','total_guides_for_gene'
]

# Keep only columns that exist in the result to avoid KeyErrors
final_cols = [col for col in target_cols if col in merged_df.columns]
final_df = merged_df[final_cols]

# 5. Save the final curated dataset
final_df.to_csv("Final_Validated_Dataset.txt", sep='\t', index=False)

print(f"--- DATA MERGE SUCCESSFUL ---")
print(f"Rows preserved: {len(final_df)}")
print(f"Columns in final set: {list(final_df.columns)}")
