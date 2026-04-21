import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_dmso_essentiality_pipeline():
    input_dir = "./" 
    base_output_dir = "TreeCRISPR_DMSO_Essentiality"
    # Added your specific editors
    editors = ['ALKM_E', 'ALKM_L', 'ALKM_H', 'ALK_E', 'ALK_L', 'ALK_H', 'WT', 'K']
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.gene_summary.txt')]
    
    summary_data = []

    for editor in editors:
        # Regex to find the editor in the filename
        editor_pattern = rf"[._]{editor}_" 
        editor_files = [f for f in all_files if re.search(editor_pattern, f)]
        
        # We ONLY care about DMSO for the Universal Essentiality Model
        dmso_file = [f for f in editor_files if "_dmso_" in f.lower()]
        
        if not dmso_file:
            print(f"Warning: Could not find DMSO summary for {editor}")
            continue

        try:
            # Load the MAGeCK gene summary
            df_dmso = pd.read_csv(os.path.join(input_dir, dmso_file[0]), sep='\t')
            
            # For Universal Essentiality, we focus strictly on Negative Selection (Dropout)
            # The 'neg|lfc' here is the raw dropout compared to Day 0
            df_final = df_dmso[['id', 'neg|lfc', 'neg|p-value', 'neg|goodsgrna']].copy()
            df_final.columns = ['Gene_ID', 'DMSO_LFC', 'P_Value', 'Good_sgRNAs']
            df_final['Editor'] = editor
            df_final['Cell_Line'] = 'K562'
            df_final['Screen_Type'] = 'Essentiality' # Hardcoded for the Master Model

            # --- HIERARCHICAL FILTERING ---
            # 1. Unfiltered: Genes with enough sgRNA coverage (the "Universe")
            unf_mask = (df_final['Good_sgRNAs'] >= 3) & (df_final['Good_sgRNAs'] <= 6)
            df_unf = df_final[unf_mask].copy()

            # 2. Filtered: Genes that are statistically significant hits
            df_fil = df_unf[df_unf['P_Value'] <= 0.05].copy()

            # --- SAVE DATA ---
            for status in ['Filtered', 'Unfiltered']:
                os.makedirs(os.path.join(base_output_dir, status, editor), exist_ok=True)

            df_unf.to_csv(os.path.join(base_output_dir, 'Unfiltered', editor, f'{editor}_dmso_unfiltered.csv'), index=False)
            df_fil.to_csv(os.path.join(base_output_dir, 'Filtered', editor, f'{editor}_dmso_filtered.csv'), index=False)

            # --- LOG SUMMARY ---
            summary_data.append({
                'Editor': editor,
                'Total_Genes': len(df_final),
                'Unfiltered_Essentiality': len(df_unf),
                'Filtered_Essentiality': len(df_fil)
            })

            # --- PLOT DROPOUT DISTRIBUTION ---
            plt.figure(figsize=(8, 6))
            sns.kdeplot(df_unf['DMSO_LFC'], label='All Guides (Coverage ok)', fill=True, color='gray')
            if not df_fil.empty:
                sns.kdeplot(df_fil['DMSO_LFC'], label='Significant Dropout (P<=0.05)', fill=True, color='green')
            
            plt.axvline(0, color='black', linestyle='--')
            plt.title(f'{editor} K562 DMSO: Essentiality Distribution')
            plt.xlabel('Log Fold Change (Dropout)')
            plt.legend()
            
            plot_dir = os.path.join(base_output_dir, "Plots")
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"{editor}_dmso_essentiality.png"))
            plt.close()

        except Exception as e:
            print(f"Error on {editor}: {e}")

    # Save final Processing Summary for merging with A375/A549 later
    pd.DataFrame(summary_data).to_csv(os.path.join(base_output_dir, "DMSO_Processing_Summary.csv"), index=False)
    print("DMSO Essentiality Pipeline Complete.")

if __name__ == "__main__":
    run_dmso_essentiality_pipeline()
