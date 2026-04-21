import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_balanced_unfiltered_dmso_selection():
    base_dir = "TreeCRISPR_DMSO_Essentiality/Unfiltered"
    
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} does not exist.")
        return

    editors = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    summary_list = []

    for editor in editors:
        editor_path = os.path.join(base_dir, editor)
        filename = f"{editor}_dmso_unfiltered.csv"
        file_path = os.path.join(editor_path, filename)
        
        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path)

            # 1. SELECT THE ESSENTIAL HITS (LFC <= -0.5)
            # These are your true biological signals for the model to learn.
            essential_hits = df[df['DMSO_LFC'] <= -0.5].copy()

            # 2. SELECT THE NEGATIVE CONTROLS (0.0 to 0.2)
            # These represent the genes that the editor failed to repress.
            # We use the 0.0 to 0.2 range specifically as you requested.
            negative_controls_all = df[(df['DMSO_LFC'] >= 0.0) & (df['DMSO_LFC'] <= 0.2)].copy()

            # 3. BALANCING LOGIC
            # To prevent the model from becoming biased towards 'No Effect', 
            # we cap the controls at 1.5x the number of hits.
            num_hits = len(essential_hits)
            num_controls_needed = int(num_hits * 1.5)

            if len(negative_controls_all) > num_controls_needed:
                negative_controls = negative_controls_all.sample(n=num_controls_needed, random_state=42)
            else:
                negative_controls = negative_controls_all

            # 4. CONSOLIDATE DATA
            # This DataFrame contains the contrast needed for Regression.
            df_balanced = pd.concat([essential_hits, negative_controls]).sort_values('DMSO_LFC')
            
            # Record counts for your audit
            summary_list.append({
                'Editor': editor,
                'Essential_Hits_Count': len(essential_hits),
                'Neutral_Controls_Count': len(negative_controls),
                'Total_Selected_Genes': len(df_balanced)
            })

            # 5. SAVE DATA
            # This is the file you will eventually merge with BigWig features.
            output_path = os.path.join(editor_path, f"{editor}_dmso_ml_ready_set.csv")
            df_balanced.to_csv(output_path, index=False)

            # 6. VISUALIZE SELECTION
            plt.figure(figsize=(10, 6))
            sns.histplot(df_balanced['DMSO_LFC'], bins=30, color='purple', kde=True)
            plt.axvline(-0.5, color='red', linestyle='--', label='Hit Threshold')
            plt.axvspan(0.0, 0.2, color='gray', alpha=0.3, label='Control Zone')
            plt.title(f'{editor}: DMSO Selection for ML Pipeline')
            plt.xlabel('Log Fold Change (Dropout)')
            plt.legend()
            
            plt.savefig(os.path.join(editor_path, f"{editor}_ml_selection_check.png"))
            plt.close()

        except Exception as e:
            print(f"Failure in {editor}: {e}")

    # 7. Save Summary CSV
    pd.DataFrame(summary_list).to_csv(os.path.join(base_dir, "DMSO_ML_Selection_Summary.csv"), index=False)
    print("Balanced Selection Complete.")

if __name__ == "__main__":
    generate_balanced_unfiltered_dmso_selection()
