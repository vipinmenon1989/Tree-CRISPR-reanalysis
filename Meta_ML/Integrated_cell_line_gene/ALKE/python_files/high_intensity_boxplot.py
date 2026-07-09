import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. Load data
df = pd.read_csv("merged_alke_dataset.csv")

# 2. Define High-Stringency Threshold
THRESHOLD = 0.9
output_dir = "stringent_threshold_plots"
os.makedirs(output_dir, exist_ok=True)

# 3. Identify features (excluding meta-columns)
drop_cols = ['ID', 'gene', 'mean_sigmoid_score', 'class']
features = [c for c in df.columns if c not in drop_cols]

# 4. Scan for significant differences at the 0.9 threshold
for feat in features:
    subset = df[[feat, 'mean_sigmoid_score']].dropna().copy()
    
    # Categorize using the 0.9 threshold
    subset['Intensity_Category'] = subset[feat].apply(
        lambda x: f'Low (<= {THRESHOLD})' if x <= THRESHOLD else f'High (> {THRESHOLD})'
    )
    
    # Only visualize if there is enough data in both groups
    if subset['Intensity_Category'].nunique() == 2:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Intensity_Category', y='mean_sigmoid_score', data=subset, palette="magma")
        
        plt.title(f"Impact of {feat} at High-Stringency Threshold (>{THRESHOLD})", weight='bold')
        plt.ylabel("Repression Efficiency (Sigmoid Score)")
        plt.xlabel("Chromatin Intensity")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{feat}_stringent_plot.png", dpi=300)
        plt.close()

print(f"[+] Scan complete. Plots using {THRESHOLD} threshold saved in ./{output_dir}/")
