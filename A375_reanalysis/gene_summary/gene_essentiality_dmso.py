import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_a375_dropout_pipeline():
    input_dir = "./" 
    base_output_dir = "TreeCRISPR_A375_Analysis"
    # Adjusted list: removed WT based on your file listing
    editors = ['ALKM_E', 'ALKM_L', 'ALKM_H', 'ALK_E', 'ALK_L', 'ALK_H', 'K']
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.gene_summary.txt')]
    
    summary_data = []

    for editor in editors:
        # Regex to find the editor specifically in the d15 vs d0 file naming
        editor_pattern = rf"d15_{editor}_" 
        editor_files = [f for f in all_files if re.search(editor_pattern, f)]
        
        if not editor_files:
            print(f"Warning: No A375 summary file found for {editor}")
            continue

        try:
            # Load the MAGeCK gene summary
            df = pd.read_csv(os.path.join(input_dir, editor_files[0]), sep='\t')
            
            # Map standard columns for the Master Matrix
            # neg|lfc represents the essentiality/dropout signal we need
            df_processed = df[['id', 'neg|lfc', 'neg|p-value', 'neg|goodsgrna']].copy()
            df_processed.columns = ['Gene_ID', 'LFC', 'P_Value', 'Good_sgRNAs']
            df_processed['Editor'] = editor
            df_processed['Cell_Line'] = 'A375'

            # --- HIERARCHICAL FILTRATION ---
            # 1. Unfiltered Universe (sgRNA coverage check)
            unf_mask = (df_processed['Good_sgRNAs'] >= 3) & (df_processed['Good_sgRNAs'] <= 6)
            df_unf = df_processed[unf_mask].copy()

            # 2. Filtered Hits (Statistical Significance)
            df_fil = df_unf[df_unf['P_Value'] <= 0.05].copy()

            # --- SAVE DATA ---
            for status in ['Filtered', 'Unfiltered']:
                os.makedirs(os.path.join(base_output_dir, status, editor), exist_ok=True)

            df_unf.to_csv(os.path.join(base_output_dir, 'Unfiltered', editor, f'A375_{editor}_unfiltered.csv'), index=False)
            df_fil.to_csv(os.path.join(base_output_dir, 'Filtered', editor, f'A375_{editor}_filtered.csv'), index=False)

            # --- LOG SUMMARY ---
            summary_data.append({
                'Editor': editor,
                'Total_Genes': len(df_processed),
                'Unfiltered_Essentiality': len(df_unf),
                'Filtered_Essentiality': len(df_fil)
            })

            # --- DROPOUT PLOT ---
            plt.figure(figsize=(8, 6))
            sns.kdeplot(df_unf['LFC'], label='All Guides (Coverage OK)', fill=True, color='gray')
            if not df_fil.empty:
                sns.kdeplot(df_fil['LFC'], label='Significant Hits (P<=0.05)', fill=True, color='red')
            
            plt.axvline(0, color='black', linestyle='--')
            plt.title(f'A375 {editor}: Dropout Essentiality')
            plt.xlabel('Log Fold Change (d15 vs d0)')
            plt.legend()
            
            plot_dir = os.path.join(base_output_dir, "Plots")
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"A375_{editor}_dropout_dist.png"))
            plt.close()

        except Exception as e:
            print(f"Error on {editor}: {e}")

    # Save Summary CSV
    pd.DataFrame(summary_data).to_csv(os.path.join(base_output_dir, "A375_Processing_Summary.csv"), index=False)
    print("A375 Pipeline Complete.")

if __name__ == "__main__":
    run_a375_dropout_pipeline()
