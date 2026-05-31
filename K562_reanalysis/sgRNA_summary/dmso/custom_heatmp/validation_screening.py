import pandas as pd
import os
import glob

def extract_exact_validation_targets(input_directory, output_suffix="_validation_screening.csv"):
    """
    Extracts exactly 5 Top and 5 Bottom genes using cascading thresholds to guarantee output size.
    """
    csv_files = glob.glob(os.path.join(input_directory, "*.csv"))
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        
        if not {'sgrna', 'Gene', 'Sigmoid_Score'}.issubset(df.columns):
            print(f"Skipping {os.path.basename(file_path)}: Missing required columns.")
            continue
            
        # Group by Gene
        gene_stats = df.groupby('Gene')['Sigmoid_Score'].agg(['mean', 'min', 'max']).reset_index()
        
        # --- TOP GENES EXTRACTION ---
        def get_top_5(stats):
            # Attempt 1: Strict 0.75 limit
            t1 = stats[stats['min'] >= 0.75]
            if len(t1) >= 5: 
                return t1.sort_values('mean', ascending=False).head(5)
            
            # Attempt 2: Relaxed 0.60 limit
            t2 = stats[stats['min'] >= 0.60]
            if len(t2) >= 5: 
                return t2.sort_values('mean', ascending=False).head(5)
            
            # Attempt 3: Hard limit fallback (Absolute top 5 available)
            return stats.sort_values('mean', ascending=False).head(5)
        
        # --- BOTTOM GENES EXTRACTION ---
        def get_bottom_5(stats):
            # Attempt 1: Strict 0.25 - 0.35 bracket
            b1 = stats[(stats['min'] >= 0.25) & (stats['max'] <= 0.35)]
            if len(b1) >= 5: 
                return b1.sort_values('mean', ascending=True).head(5)
            
            # Attempt 2: Hard limit fallback (5 closest to target center of 0.30)
            stats_copy = stats.copy()
            stats_copy['dist_to_030'] = abs(stats_copy['mean'] - 0.30)
            return stats_copy.sort_values('dist_to_030', ascending=True).head(5).drop(columns=['dist_to_030'])

        # Execute extraction
        top_5_genes = get_top_5(gene_stats)
        bottom_5_genes = get_bottom_5(gene_stats)
        
        # Combine and filter original dataset
        selected_genes = pd.concat([top_5_genes, bottom_5_genes])['Gene']
        final_df = df[df['Gene'].isin(selected_genes)].copy()
        
        # Apply labels
        final_df['Category'] = final_df['Gene'].apply(
            lambda x: 'Top' if x in top_5_genes['Gene'].values else 'Bottom'
        )
        
        # Sort output
        final_df = final_df.sort_values(by=['Category', 'Gene', 'Sigmoid_Score'], ascending=[False, True, False])
        
        # Export
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{base_name}{output_suffix}"
        output_path = os.path.join(os.path.dirname(file_path), output_filename)
        
        final_df.to_csv(output_path, index=False)
        print(f"Processed: {os.path.basename(file_path)} -> Saved as: {output_filename}")

# --- Execution ---
target_folder = '.' 
extract_exact_validation_targets(target_folder)
