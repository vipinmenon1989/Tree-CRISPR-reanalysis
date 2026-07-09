import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os

# 1. Load data
df = pd.read_csv("merged_alke_dataset.csv")
drop_cols = ['ID', 'gene', 'mean_sigmoid_score', 'class']
features = [c for c in df.columns if c not in drop_cols]

# 2. Setup output directory
output_dir = "publication_plots"
os.makedirs(output_dir, exist_ok=True)

# 3. Process each significant feature
for feat in features:
    clean_data = df[[feat, 'class']].dropna()
    class_0 = clean_data[clean_data['class'] == 0][feat]
    class_1 = clean_data[clean_data['class'] == 1][feat]
    
    stat, p_val = ks_2samp(class_0, class_1)
    
    # Only plot if significant
    if p_val <= 0.05:
        plt.figure(figsize=(6, 8))
        
        # Prepare plotting data
        plot_df = clean_data.copy()
        plot_df['Label'] = plot_df['class'].map({1: 'Efficient\n(Class 1)', 0: 'Inefficient\n(Class 0)'})
        
        # Create boxplot
        ax = sns.boxplot(x='Label', y=feat, data=plot_df, palette={"Efficient\n(Class 1)": "#3498db", "Inefficient\n(Class 0)": "#e74c3c"}, width=0.5)
        
        # Add P-value annotation
        y_max = plot_df[feat].max()
        plt.plot([0, 1], [y_max * 1.05, y_max * 1.05], color='black', lw=1.5)
        plt.text(0.5, y_max * 1.08, f'p = {p_val:.4e}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.title(f"Gatekeeper Analysis: {feat}", fontsize=14, fontweight='bold')
        plt.ylabel("Normalized Signal Intensity")
        plt.xlabel("")
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/{feat}_boxplot.png", dpi=300)
        plt.close()

print(f"[+] Publication-ready plots saved to ./{output_dir}/")
