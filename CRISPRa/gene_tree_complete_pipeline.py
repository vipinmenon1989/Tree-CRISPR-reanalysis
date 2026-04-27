import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_crispra_pipeline():
    input_dir = "./" 
    # Differentiating the output folder for CRISPRa
    base_output_dir = "TreeCRISPRa_Analysis" 
    # SAM and T300 are the standard activators
    editors = ['WT', 'SAM', 'T300', 'TP'] 
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.gene_summary.txt')]
    
    summary_data = []

    for editor in editors:
        # Robust pattern matching for CRISPRa filenames
        editor_pattern = rf"[._]{editor}_" 
        editor_files = [f for f in all_files if re.search(editor_pattern, f)]
        
        dmso_file = [f for f in editor_files if "_dmso_" in f.lower()]
        rig_file = [f for f in editor_files if "_rig_" in f.lower()]
        
        if not dmso_file or not rig_file:
            print(f"Warning: Could not find complete pair for {editor}")
            continue

        try:
            df_dmso = pd.read_csv(os.path.join(input_dir, dmso_file[0]), sep='\t')
            df_rig = pd.read_csv(os.path.join(input_dir, rig_file[0]), sep='\t')
            
            cols = ['id', 'neg|lfc', 'neg|p-value', 'pos|lfc', 'pos|p-value', 'neg|goodsgrna', 'pos|goodsgrna']
            
            df_merged = pd.merge(
                df_rig[cols].add_prefix('RIG_'), 
                df_dmso[cols].add_prefix('DMSO_'), 
                left_on='RIG_id', right_on='DMSO_id'
            ).rename(columns={'RIG_id': 'Gene_ID'})

            # Logic check: Delta LFC remains (Drug - Control)
            df_merged['Delta_Neg_LFC'] = df_merged['RIG_neg|lfc'] - df_merged['DMSO_neg|lfc']
            df_merged['Delta_Pos_LFC'] = df_merged['RIG_pos|lfc'] - df_merged['DMSO_pos|lfc']
            df_merged['Editor'] = editor

            # --- HIERARCHICAL FILTERING ---
            unf_neg_mask = (df_merged['RIG_neg|goodsgrna'] >= 3) & (df_merged['RIG_neg|goodsgrna'] <= 6)
            unf_pos_mask = (df_merged['RIG_pos|goodsgrna'] >= 3) & (df_merged['RIG_pos|goodsgrna'] <= 6)
            
            df_unf_neg = df_merged[unf_neg_mask].copy()
            df_unf_pos = df_merged[unf_pos_mask].copy()

            # Filtered subset (Statistical Significance)
            df_fil_neg = df_unf_neg[df_unf_neg['RIG_neg|p-value'] <= 0.05].copy()
            df_fil_pos = df_unf_pos[df_unf_pos['RIG_pos|p-value'] <= 0.05].copy()

            # --- SAVE DATA ---
            for status in ['Filtered', 'Unfiltered']:
                os.makedirs(os.path.join(base_output_dir, status, editor), exist_ok=True)

            # Saving with CRISPRa naming convention
            df_unf_neg.to_csv(os.path.join(base_output_dir, 'Unfiltered', editor, f'a_{editor}_neg_unfiltered.csv'), index=False)
            df_fil_neg.to_csv(os.path.join(base_output_dir, 'Filtered', editor, f'a_{editor}_neg_filtered.csv'), index=False)
            df_unf_pos.to_csv(os.path.join(base_output_dir, 'Unfiltered', editor, f'a_{editor}_pos_unfiltered.csv'), index=False)
            df_fil_pos.to_csv(os.path.join(base_output_dir, 'Filtered', editor, f'a_{editor}_pos_filtered.csv'), index=False)

            # --- LOG SUMMARY ---
            summary_data.append({
                'Editor': f"a_{editor}",
                'Unf_Sensitizers': len(df_unf_neg),
                'Fil_Sensitizers': len(df_fil_neg),
                'Unf_Resistance': len(df_unf_pos),
                'Fil_Resistance': len(df_fil_pos)
            })

            # --- PLOTTING ---
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            sns.kdeplot(df_unf_neg['Delta_Neg_LFC'], label='Unfiltered', fill=True, color='gray', ax=ax1)
            if not df_fil_neg.empty:
                sns.kdeplot(df_fil_neg['Delta_Neg_LFC'], label='Filtered (P<=0.05)', fill=True, color='green', ax=ax1)
            ax1.set_title(f'CRISPRa {editor}: Sensitization (Overexpression -> Death)')
            ax1.legend()

            sns.kdeplot(df_unf_pos['Delta_Pos_LFC'], label='Unfiltered', fill=True, color='gray', ax=ax2)
            if not df_fil_pos.empty:
                sns.kdeplot(df_fil_pos['Delta_Pos_LFC'], label='Filtered (P<=0.05)', fill=True, color='orange', ax=ax2)
            ax2.set_title(f'CRISPRa {editor}: Resistance (Overexpression -> Survival)')
            ax2.legend()

            plt.tight_layout()
            plot_dir = os.path.join(base_output_dir, "Plots")
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"a_{editor}_side_by_side.png"))
            plt.close()

        except Exception as e:
            print(f"Error on {editor}: {e}")

    pd.DataFrame(summary_data).to_csv(os.path.join(base_output_dir, "CRISPRa_Processing_Summary.csv"), index=False)

if __name__ == "__main__":
    run_crispra_pipeline()