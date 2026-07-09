import pandas as pd

# 1. Load the data
df = pd.read_csv("Combined_Prediction_Metadata_Result.txt", sep='\t')
df['gene'] = df['gene'].str.strip()

# 2. Identify "Consistent" guides
df['is_consistent'] = (df['class'] == df['prediction_binary'])

# 3. Filter for genes with >= 2 consistent guides of the SAME class
# Grouping by [gene, class] ensures consistency and type matching
validated_df = df[df['is_consistent'] == True].groupby(['gene', 'class']).filter(lambda x: len(x) >= 2)

# 4. SORTING: Sort by 'gene' (alphabetical) and then by 'id' (numerical)
validated_df = validated_df.sort_values(by=['gene', 'id']).drop(columns=['is_consistent'])

# 5. Save the sorted validation file
output_path = "Validated_Strict_Genes_Sorted.txt"
validated_df.to_csv(output_path, sep='\t', index=False)

# 6. Reporting
print(f"--- SORTING & FILTERING COMPLETE ---")
print(f"Total rows: {len(validated_df)}")
print(f"Unique genes: {validated_df['gene'].nunique()}")
print(f"File saved to: {output_path}")

# Visualization check for the structure
print("\n--- PREVIEW OF SORTED DATA ---")
print(validated_df.head(10))
