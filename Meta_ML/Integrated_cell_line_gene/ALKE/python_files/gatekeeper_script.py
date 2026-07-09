import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os

# 1. Load data
df = pd.read_csv("merged_alke_dataset.csv")

# 2. Setup directory
output_dir = "significant_gatekeepers"
os.makedirs(output_dir, exist_ok=True)

# 3. Identify features
# We explicitly exclude non-feature columns
drop_cols = ['ID', 'gene', 'mean_sigmoid_score', 'class']
features = [c for c in df.columns if c not in drop_cols]

# 4. Loop through ALL features
for feat in features:
    # We need the feature, the class (for KS test), AND the score (for plotting)
    # We drop any rows where ANY of these are missing
    subset = df[[feat, 'class', 'mean_sigmoid_score']].dropna()
    
    class_0 = subset[subset['class'] == 0][feat]
    class_1 = subset[subset['class'] == 1][feat]
    
    # Perform KS-test
    stat, p_val = ks_2samp(class_0, class_1)
    
    # Only proceed if significant
    if p_val <= 0.05:
        # Create categories for threshold analysis on the subset
        subset = subset.copy()
        subset['Intensity_Category'] = subset[feat].apply(
            lambda x: 'Low (<= 0.5)' if x <= 0.5 else 'High (> 0.5)'
        )
        
        # Plotting
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='Intensity_Category', y='mean_sigmoid_score', data=subset, palette="viridis")
        
        plt.title(f"Impact of {feat} (KS p={p_val:.2e})", weight='bold')
        plt.ylabel("Repression Efficiency (Sigmoid Score)")
        plt.xlabel("Chromatin Intensity Threshold")
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{feat}_impact_plot.png", dpi=300)
        plt.close()

print(f"[+] Scan complete. {len(features)} features checked.")
print(f"[+] Box plots for significant gatekeepers saved in ./{output_dir}/")