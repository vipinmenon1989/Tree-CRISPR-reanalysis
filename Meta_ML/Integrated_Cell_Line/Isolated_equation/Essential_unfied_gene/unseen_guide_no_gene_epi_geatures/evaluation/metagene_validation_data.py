import pandas as pd

# 1. Load the two files
# Replace filenames with your actual file paths
file_predictions = pd.read_csv("independent_predictions.txt", sep='\t')
file_metadata = pd.read_csv("Unified_target_genes_metadata.txt", sep='\t')

# 2. Merge the dataframes
# We merge on both identifiers to ensure strict matching
merged_df = pd.merge(
    file_predictions, 
    file_metadata, 
    on=['unique_sgrna_id', 'sgrna sequence'], 
    how='inner'
)

# 3. Select and order the required columns
final_columns = [
    'sgrna sequence', 
    'gene', 
    'id', 
    'class', 
    'prediction_probability', 
    'prediction_binary'
]

# Ensure all requested columns exist before filtering
missing_cols = [col for col in final_columns if col not in merged_df.columns]
if not missing_cols:
    output_df = merged_df[final_columns]
else:
    print(f"ERROR: Missing columns in merged file: {missing_cols}")
    exit(1)

# 4. Save the third file
output_path = "Combined_Prediction_Metadata_Result.txt"
output_df.to_csv(output_path, sep='\t', index=False)

print(f"--- MERGE COMPLETE ---")
print(f"Total rows successfully matched: {output_df.shape[0]}")
print(f"File saved to: {output_path}")
