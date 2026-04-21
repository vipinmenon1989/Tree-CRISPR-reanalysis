import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_internal_plots_with_summary():
    # 1. Path to the root of your filtered data
    base_dir = "TreeCRISPR_Analysis/Filtered"
    
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist. Check your directory structure.")
        return

    # 2. Iterate through each editor folder
    editors = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # List to store summary counts
    summary_list = []

    for editor in editors:
        editor_path = os.path.join(base_dir, editor)
        
        neg_filename = f"{editor}_neg_filtered.csv"
        pos_filename = f"{editor}_pos_filtered.csv"
        
        neg_path = os.path.join(editor_path, neg_filename)
        pos_path = os.path.join(editor_path, pos_filename)
        
        if not os.path.exists(neg_path) or not os.path.exists(pos_path):
            continue

        try:
            # 4. Load Data
            df_neg = pd.read_csv(neg_path)
            df_pos = pd.read_csv(pos_path)

            # 5. Directional Filtering (THE TRUTH)
            # Only count genes that actually moved in the intended direction
            final_neg_hits = df_neg[df_neg['Delta_Neg_LFC'] < 0]
            final_pos_hits = df_pos[df_pos['Delta_Pos_LFC'] > 0]
            
            # Record counts for summary
            summary_list.append({
                'Editor': editor,
                'Total_Sensitizers_LFC_Neg': len(final_neg_hits),
                'Total_Resistance_LFC_Pos': len(final_pos_hits),
                'Total_High_Confidence_Genes': len(final_neg_hits) + len(final_pos_hits)
            })

            # Get Top 10 for Plotting
            plot_neg = final_neg_hits.sort_values('Delta_Neg_LFC').head(10)
            plot_pos = final_pos_hits.sort_values('Delta_Pos_LFC', ascending=False).head(10)

            # 6. Plotting Logic
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Left: Sensitizers
            if not plot_neg.empty:
                sns.barplot(data=plot_neg, x='Delta_Neg_LFC', y='Gene_ID', hue='Gene_ID', palette='Reds_r', legend=False, ax=ax1)
                ax1.set_title(f'Top 10 Sensitizers (LFC < 0)', fontsize=12)
            else:
                ax1.text(0.5, 0.5, 'No Hits with LFC < 0', ha='center')

            # Right: Resistance
            if not plot_pos.empty:
                sns.barplot(data=plot_pos, x='Delta_Pos_LFC', y='Gene_ID', hue='Gene_ID', palette='Blues_r', legend=False, ax=ax2)
                ax2.set_title(f'Top 10 Resistance Genes (LFC > 0)', fontsize=12)
            else:
                ax2.text(0.5, 0.5, 'No Hits with LFC > 0', ha='center')

            plt.suptitle(f"Directional Analysis: {editor}", fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            plt.savefig(os.path.join(editor_path, f"{editor}_top_directional_hits.png"))
            plt.close()
            
            print(f"Finalized {editor}")

        except Exception as e:
            print(f"Failure in {editor}: {e}")

    # 7. Create and Save the Final Summary CSV
    df_summary = pd.DataFrame(summary_list)
    summary_path = os.path.join(base_dir, "Directional_Filtering_Summary.csv")
    df_summary.to_csv(summary_path, index=False)
    
    print("-" * 30)
    print(f"SUMMARY SAVED TO: {summary_path}")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    generate_internal_plots_with_summary()