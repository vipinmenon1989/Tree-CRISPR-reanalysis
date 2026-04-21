import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_unfiltered_crispra_analysis():
    # 1. Path to the root of your UNFILTERED CRISPRa data
    base_dir = "TreeCRISPRa_Analysis/Unfiltered"
    
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist. Check your CRISPRa pipeline output.")
        return

    # 2. Iterate through each activation editor folder (SAM, T300, etc.)
    editors = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    summary_list = []

    for editor in editors:
        editor_path = os.path.join(base_dir, editor)
        
        # Consistent with your a_{editor}_... naming convention
        neg_filename = f"a_{editor}_neg_unfiltered.csv"
        pos_filename = f"a_{editor}_pos_unfiltered.csv"
        
        neg_path = os.path.join(editor_path, neg_filename)
        pos_path = os.path.join(editor_path, pos_filename)
        
        if not os.path.exists(neg_path) or not os.path.exists(pos_path):
            print(f"Skipping {editor}: Missing expected CSV files.")
            continue

        try:
            # 4. Load Data
            df_neg = pd.read_csv(neg_path)
            df_pos = pd.read_csv(pos_path)

            # 5. Directional + Magnitude Filtering (|LFC| >= 0.5)
            # Sensitizers: Activation makes drug MORE lethal
            final_neg_hits = df_neg[df_neg['Delta_Neg_LFC'] <= -0.5]
            # Resistance: Activation makes drug LESS lethal (Rescue)
            final_pos_hits = df_pos[df_pos['Delta_Pos_LFC'] >= 0.5]
            
            # Record counts
            summary_list.append({
                'Editor': f"a_{editor}",
                'a_Sensitizers_LFC_le_-0.5': len(final_neg_hits),
                'a_Resistance_LFC_ge_0.5': len(final_pos_hits),
                'Total_ML_Samples_a': len(final_neg_hits) + len(final_pos_hits)
            })

            # 6. Plotting (Top 10 for visual sanity check)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            if not final_neg_hits.empty:
                plot_neg = final_neg_hits.sort_values('Delta_Neg_LFC').head(10)
                sns.barplot(data=plot_neg, x='Delta_Neg_LFC', y='Gene_ID', hue='Gene_ID', palette='Reds_r', legend=False, ax=ax1)
                ax1.set_title(f'CRISPRa {editor}: Top Overexpression Sensitizers', fontsize=12)
            
            if not final_pos_hits.empty:
                plot_pos = final_pos_hits.sort_values('Delta_Pos_LFC', ascending=False).head(10)
                sns.barplot(data=plot_pos, x='Delta_Pos_LFC', y='Gene_ID', hue='Gene_ID', palette='Blues_r', legend=False, ax=ax2)
                ax2.set_title(f'CRISPRa {editor}: Top Overexpression Resistance', fontsize=12)

            plt.suptitle(f"CRISPRa Unfiltered Directional Analysis: {editor}", fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save PNG inside the Unfiltered CRISPRa editor folder
            plt.savefig(os.path.join(editor_path, f"a_{editor}_unfiltered_top_hits.png"))
            plt.close()
            
            print(f"Processed CRISPRa {editor}")

        except Exception as e:
            print(f"Failure in CRISPRa {editor}: {e}")

    # 7. Save the Unfiltered CRISPRa Summary
    df_summary = pd.DataFrame(summary_list)
    summary_path = os.path.join(base_dir, "CRISPRa_Unfiltered_Directional_Summary.csv")
    df_summary.to_csv(summary_path, index=False)
    
    print("-" * 30)
    print(f"CRISPRa SUMMARY SAVED TO: {summary_path}")
    print(df_summary.to_string(index=False))

if __name__ == "__main__":
    generate_unfiltered_crispra_analysis()
