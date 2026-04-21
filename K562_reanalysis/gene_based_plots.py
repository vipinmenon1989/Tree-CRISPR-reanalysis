import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_clean_barplots():
    # Define the editors and treatments
    editors = ['ALKM_E', 'ALKM_L', 'ALKM_H', 'ALK_E', 'ALK_L', 'ALK_H', 'WT', 'K']
    treatments = ['dmso', 'rig']

    for editor in editors:
        for treatment in treatments:
            # Construct folder path
            folder_path = os.path.join(editor, treatment)
            
            # Skip if the directory doesn't even exist
            if not os.path.exists(folder_path):
                continue

            # --- CLEANUP STEP ---
            # Remove all previous png and pdf files in this specific folder
            old_files = glob.glob(os.path.join(folder_path, "*.png")) + \
                        glob.glob(os.path.join(folder_path, "*.pdf"))
            for f in old_files:
                try:
                    os.remove(f)
                except OSError as e:
                    print(f"Error deleting {f}: {e}")

            # Define filtered data file path
            file_name = f"{editor}_{treatment}_filtered.txt"
            file_path = os.path.join(folder_path, file_name)

            if not os.path.exists(file_path):
                continue

            try:
                # Load the data
                df = pd.read_csv(file_path, sep='\t')

                # Filter: Only plot significant data
                plot_df = df[df['p-value significant'] == True].copy()

                if plot_df.empty:
                    print(f"Skipping {editor}/{treatment}: No significant data.")
                    continue

                # Initialize Plot
                plt.figure(figsize=(14, 8))
                sns.set_style("whitegrid")

                # Create Bar Plot
                # x = id (Gene Name), y = neg|lfc (Log Fold Change)
                sns.barplot(
                    data=plot_df, 
                    x='id', 
                    y='neg|lfc', 
                    palette="viridis",
                    capsize=.1,
                    errorbar='se'
                )

                # Formatting
                plt.title(f"Mean Significant LFC: {editor} - {treatment}", fontsize=16)
                plt.ylabel("Mean Log Fold Change (neg|lfc)", fontsize=12)
                plt.xlabel("Gene ID", fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.axhline(0, color='black', linewidth=1)
                plt.tight_layout()

                # Save new files
                base_path = os.path.join(folder_path, f"{editor}_{treatment}_barplot")
                plt.savefig(f"{base_path}.png", dpi=300)
                plt.savefig(f"{base_path}.pdf")
                
                plt.close()
                print(f"Cleaned and updated: {editor}/{treatment}")

            except Exception as e:
                print(f"Error processing {editor}/{treatment}: {e}")

if __name__ == "__main__":
    generate_clean_barplots()
