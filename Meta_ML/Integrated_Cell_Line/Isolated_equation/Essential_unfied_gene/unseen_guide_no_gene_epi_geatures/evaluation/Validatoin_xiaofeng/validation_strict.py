import pandas as pd

# 1. Load the merged file
df = pd.read_csv("Combined_Prediction_Metadata_Result.txt", sep='\t')
df['gene'] = df['gene'].str.strip()

# 2. Identify Consistent guides (Class == Prediction_Binary)
df['is_consistent'] = (df['class'] == df['prediction_binary'])
consistent_df = df[df['is_consistent'] == True].copy()

# 3. Filter for genes with >= 2 distinct guides
# First, ensure we only count distinct sequences as guides
distinct_df = consistent_df.drop_duplicates(subset=['sgrna sequence', 'gene'])

# 4. Strict Filter:
# A gene is kept ONLY IF:
#   a) It has >= 2 distinct guides
#   b) ALL IDs within that gene are unique (no ID repeats for that gene)
def is_gene_valid(group):
    # Check if more than one distinct guide exists
    has_two_guides = len(group) >= 2
    # Check if there are NO duplicate IDs in this gene group
    has_unique_ids = group['id'].is_unique
    return has_two_guides and has_unique_ids

# Apply the strict filter
validated_df = distinct_df.groupby(['gene', 'class']).filter(is_gene_valid)

# 5. Save the final file
output_path = "Validated_Strict_UniqueID_Genes.txt"
validated_df.sort_values(by=['gene', 'id']).to_csv(output_path, sep='\t', index=False)

# 6. Reporting
print(f"--- FILTERING COMPLETE ---")
print(f"Total rows remaining: {len(validated_df)}")
print(f"Unique genes remaining: {validated_df['gene'].nunique()}")
print(f"File saved to: {output_path}")

# Visualization of the grouping logic used to clean your data
