import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_unfiltered_analysis():
    # 1. Path to the root of your UNFILTERED data
    base_dir = "TreeCRISPR_Analysis/Unfiltered"
    
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist.")
        return

    # 2. Iterate through each editor folder
    editors = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    summary_list = []

    for editor in editors:
        editor_path = os.path.join(base_dir, editor)
        
        # Note: filenames here should match your Unfiltered naming convention
        # Based on previous steps, they likely end in _unfiltered.csv
        neg_filename = f"{editor}_neg_unfiltered.csv"
        pos_filename = f"{editor}_pos_unfiltered.csv"
        
        neg_path = os.path.join(editor_path, neg_filename)
        pos_path = os.path.join(editor_path, pos_filename)
        
        if not os.path.exists(neg_path) or not os.path.exists(pos_path):
            continue

        try:
            # 4. Load Data
            df_neg = pd.read_csv(neg_path)
            df_pos = pd.read_csv(pos_path)

            # 5. Directional Filtering (The ML Training Set)
            # We use < 0 and > 0 to capture the full range of biological behavior
            final_neg_hits = df_neg[df_neg['Delta_Neg_LFC'] <= -0.5]
            final_pos_hits = df_pos[df_pos['Delta_Pos_LFC'] >= 0.5]
            
            # Record counts
            summary_list.append({
                'Editor': editor,
                'Unfiltered_Sensitizers': len(final_neg_hits),
                'Unfiltered_Resistance': len(final_pos_hits),
                'Total_ML_Samples': len(final_neg_hits) + len(final_pos_hits)
            })

            # 6. Plotting (Top 10 for visual sanity check)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            if not final_neg_hits.empty:
                plot_neg = final_neg_hits.sort_values('Delta_Neg_LFC').head(10)
                sns.barplot(data=plot_neg, x='Delta_Neg_LFC', y='Gene_ID', hue='Gene_ID', palette='Reds_r', legend=False, ax=ax1)
                ax1.set_title(f'{editor}: Top Unfiltered Sensitizers', fontsize=12)
            
            if not final_pos_hits.empty:
                plot_pos = final_pos_hits.sort_values('Delta_Pos_LFC', ascending=False).head(10)
                sns.barplot(data=plot_pos, x='Delta_Pos_LFC', y='Gene_ID', hue='Gene_ID', palette='Blues_r', legend=False, ax=ax2)
                ax2.set_title(f'{editor}: Top Unfiltered Resistance', fontsize=12)

            plt.suptitle(f"Unfiltered Directional Analysis: {editor}", fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save PNG inside the Unfiltered editor folder
            plt.savefig(os.path.join(editor_path, f"{editor}_unfiltered_top_hits.png"))
            plt.close()
            
            print(f"Processed Unfiltered {editor}")

        except Exception as e:
            print(f"Failure in {editor}: {e}")

    # 7. Save the Unfiltered Summary
    df_summary = pd.DataFrame(summary_list)
    summary_path = os.path.join(base_dir, "Unfiltered_Directional_Summary.csv")
    df_summary.to_csv(summary_path, index=False)
    
    print("-" * 30)
    print(f"UNFILTERED SUMMARY SAVED TO: {summary_path}")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    generate_unfiltered_analysis()
