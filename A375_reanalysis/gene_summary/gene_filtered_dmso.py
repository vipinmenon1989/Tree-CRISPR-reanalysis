import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_a375_essentiality_plots():
    # 1. Path to your A375 specific filtered data
    base_dir = "TreeCRISPR_A375_Analysis/Filtered"
    
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist. Ensure the A375 pipeline has run.")
        return

    # 2. Identify editor folders
    editors = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    summary_list = []

    for editor in editors:
        editor_path = os.path.join(base_dir, editor)
        
        # Matches the filename pattern from the A375 pipeline code
        filename = f"A375_{editor}_filtered.csv"
        file_path = os.path.join(editor_path, filename)
        
        if not os.path.exists(file_path):
            continue

        try:
            # 4. Load Data
            df = pd.read_csv(file_path)

            # 5. Directional Filtering
            # In A375 dropout, we focus on negative LFC (essential genes that died)
            essential_hits = df[df['LFC'] < 0].copy()
            
            # Record counts for summary
            summary_list.append({
                'Editor': editor,
                'Total_Essential_Hits': len(essential_hits),
                'Mean_Dropout_LFC': essential_hits['LFC'].mean() if not essential_hits.empty else 0,
                'Median_PValue': essential_hits['P_Value'].median() if not essential_hits.empty else 1.0
            })

            # Get Top 15 Essential Genes (most negative LFC)
            plot_df = essential_hits.sort_values('LFC').head(15)

            # 6. Plotting Logic
            plt.figure(figsize=(10, 8))

            if not plot_df.empty:
                sns.barplot(
                    data=plot_df, 
                    x='LFC', 
                    y='Gene_ID', 
                    hue='Gene_ID', 
                    palette='Reds_r', 
                    legend=False
                )
                plt.title(f'A375: Top 15 Essential Genes ({editor})', fontsize=14, fontweight='bold')
                plt.xlabel('Log Fold Change (Negative = Stronger Dropout)')
                plt.ylabel('Gene Symbol')
            else:
                plt.text(0.5, 0.5, f'No Hits with LFC < 0 for {editor}', ha='center', va='center')

            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            # Save plot in the specific editor folder
            plt.savefig(os.path.join(editor_path, f"A375_{editor}_top_hits.png"))
            plt.close()
            
            print(f"Finalized A375 Analysis for {editor}")

        except Exception as e:
            print(f"Failure in {editor}: {e}")

    # 7. Save the Final Summary CSV
    df_summary = pd.DataFrame(summary_list)
    summary_path = os.path.join(base_dir, "A375_Essentiality_Filtering_Summary.csv")
    df_summary.to_csv(summary_path, index=False)
    
    print("-" * 30)
    print(f"A375 SUMMARY SAVED TO: {summary_path}")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    generate_a375_essentiality_plots()
