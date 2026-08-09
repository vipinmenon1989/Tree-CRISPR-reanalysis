import pandas as pd

# 1. Load data
df = pd.read_csv("Combined_Prediction_Metadata_Result.txt", sep='\t')
df['gene'] = df['gene'].str.strip()

# 2. Keep the deduplicated unique observations
unique_df = df.drop_duplicates(subset=['sgrna sequence', 'gene', 'id'])

# 3. Filter for genes with 2+ guides (This ensures biological robustness)
valid_genes = unique_df.groupby('gene').filter(lambda x: len(x) >= 2).copy()

# 4. Calculate Status and Guide Count per Gene
def analyze_gene(group):
    total_guides = len(group)
    
    # Check agreement between ground truth and model prediction
    if (group['class'] == group['prediction_binary']).all(): 
        status = 'Perfect_Agreement'
    elif not (group['class'] == group['prediction_binary']).any(): 
        status = 'Systematic_Mismatch'
    else: 
        status = 'Mixed_Performance'
        
    return pd.Series({
        'gene_status': status, 
        'total_guides_for_gene': total_guides
    })

# Apply the analysis to get a summary table
gene_metrics = valid_genes.groupby('gene').apply(analyze_gene).reset_index()

# 5. Merge the new metadata back into the main dataframe
final_set = pd.merge(valid_genes, gene_metrics, on='gene')

# Sort for easy reading
final_set = final_set.sort_values(by=['gene_status', 'gene', 'id'])

# 6. Save the comprehensive file with ALL columns intact
final_set.to_csv("Comprehensive_Validation_Set.txt", sep='\t', index=False)

print(f"--- COMPREHENSIVE DATASET EXTRACTED ---")
print(f"Total Genes: {final_set['gene'].nunique()}")
print(f"Total Guides: {len(final_set)}")
print(f"Columns exported: {list(final_set.columns)}")
print("\nBreakdown by Gene Status:")
print(final_set.groupby('gene_status')['gene'].nunique())

