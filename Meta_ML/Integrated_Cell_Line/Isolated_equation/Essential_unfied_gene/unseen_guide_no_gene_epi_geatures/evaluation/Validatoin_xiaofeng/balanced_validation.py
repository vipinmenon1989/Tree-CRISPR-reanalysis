import pandas as pd

# 1. Load the data
df = pd.read_csv("Combined_Prediction_Metadata_Result.txt", sep='\t')
df['gene'] = df['gene'].str.strip()

# 2. Identify all consistent guides
df['is_consistent'] = (df['class'] == df['prediction_binary'])
consistent_df = df[df['is_consistent'] == True].copy()

# 3. Find genes that have at least 2 consistent guides TOTAL
# This will capture genes that have two '1's, two '0's, OR a mix (one '1' and one '0')
def is_gene_balanced(group):
    # Ensure they are distinct sgRNAs
    unique_guides = group.drop_duplicates(subset=['sgrna sequence'])
    return len(unique_guides) >= 2

balanced_genes_df = consistent_df.groupby('gene').filter(is_gene_balanced)

# 4. Save and analyze distribution
balanced_genes_df.sort_values(['gene', 'class']).to_csv("Balanced_Experimental_Validation_Set.txt", sep='\t', index=False)

# 5. Check your new negative control count
print("--- VALIDATION SET DISTRIBUTION ---")
print(balanced_genes_df['class'].value_counts())
