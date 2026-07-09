import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv("merged_alke_dataset.csv")

# 2. Sanitize: Force numeric and drop rows with missing values
# This ensures the density estimator doesn't crash on 'NaN's
cols_to_use = ['h3k4me2_bin_8', 'mean_sigmoid_score', 'class']
df_clean = df[cols_to_use].apply(pd.to_numeric, errors='coerce').dropna()

# 3. Setup the figure
plt.figure(figsize=(9, 7))
sns.set_style("whitegrid") # Adds professional grid lines

# 4. Generate the Density Map
# Use 'common_norm=False' so the distributions are normalized independently,
# ensuring the 'Inefficient' cloud is visible even if it's smaller.
sns.kdeplot(data=df_clean, 
            x='h3k4me2_bin_8', 
            y='mean_sigmoid_score', 
            hue='class', 
            fill=True, 
            palette={0: "#e74c3c", 1: "#3498db"}, # Professional Red/Blue
            alpha=0.4, 
            levels=10,
            common_norm=False)

# 5. Professional Formatting
plt.title("Competitive Exclusion Mapping: H3K4me2 vs. Repression", weight='bold', fontsize=14)
plt.xlabel("H3K4me2 Signal Intensity (Normalized)", fontsize=12)
plt.ylabel("Repression Efficiency (Sigmoid Score)", fontsize=12)

# Legend adjustment
plt.legend(title="Class", labels=["Inefficient (Class 0)", "Efficient (Class 1)"])

plt.tight_layout()
plt.savefig("final_density_heatmap.png", dpi=300)
print("[+] Publication-quality density heatmap saved as final_density_heatmap.png")
