import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_crispra_filtered_analysis():
    # 1. Path to the FILTERED CRISPRa data
    base_dir = "TreeCRISPRa_Analysis/Filtered"
    
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist. Check naming: TreeCRISPRa_Analysis vs TreeCRISPR_Analysis.")
        return

    # 2. Iterate through SAM, T300, WT folders
    editors = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    summary_list = []

    for editor in editors:
        editor_path = os.path.join(base_dir, editor)
        
        # Files created by your a_ prefix pipeline
        neg_filename = f"a_{editor}_neg_filtered.csv"
        pos_filename = f"a_{editor}_pos_filtered.csv"
        
        neg_path = os.path.join(editor_path, neg_filename)
        pos_path = os.path.join(editor_path, pos_filename)
        
        if not os.path.exists(neg_path) or not os.path.exists(pos_path):
            continue

    # [Logic for directional count on high-confidence hits]
        try:
            df_neg = pd.read_csv(neg_path)
            df_pos = pd.read_csv(pos_path)

            # High Confidence Directional Hits (P-value already <= 0.05 from folder name)
            final_neg_hits = df_neg[df_neg['Delta_Neg_LFC'] < 0]
            final_pos_hits = df_pos[df_pos['Delta_Pos_LFC'] > 0]
            
            summary_list.append({
                'Editor': f"a_{editor}",
                'Conf_Sensitizers': len(final_neg_hits),
                'Conf_Resistance': len(final_pos_hits),
                'Total_High_Conf_a': len(final_neg_hits) + len(final_pos_hits)
            })

            # Plotting Top 10 High Confidence
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            if not final_neg_hits.empty:
                plot_neg = final_neg_hits.sort_values('Delta_Neg_LFC').head(10)
                sns.barplot(data=plot_neg, x='Delta_Neg_LFC', y='Gene_ID', hue='Gene_ID', palette='Reds_r', legend=False, ax=ax1)
                ax1.set_title(f'Top Confident CRISPRa Sensitizers (P <= 0.05)', fontsize=12)
            
            if not final_pos_hits.empty:
                plot_pos = final_pos_hits.sort_values('Delta_Pos_LFC', ascending=False).head(10)
                sns.barplot(data=plot_pos, x='Delta_Pos_LFC', y='Gene_ID', hue='Gene_ID', palette='Blues_r', legend=False, ax=ax2)
                ax2.set_title(f'Top Confident CRISPRa Resistance (P <= 0.05)', fontsize=12)

            plt.suptitle(f"CRISPRa Confidence Hits: {editor}", fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            plt.savefig(os.path.join(editor_path, f"a_{editor}_filtered_confidence_plots.png"))
            plt.close()

        except Exception as e:
            print(f"Error processing {editor}: {e}")

    # Save summary
    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(os.path.join(base_dir, "CRISPRa_Filtered_Directional_Summary.csv"), index=False)
    print("-" * 30)
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    generate_crispra_filtered_analysis()
