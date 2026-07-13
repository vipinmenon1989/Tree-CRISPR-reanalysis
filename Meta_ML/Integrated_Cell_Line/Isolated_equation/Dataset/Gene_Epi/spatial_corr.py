import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("plots", exist_ok=True)

# 1. Load your saved correlation data
# (Assumes you ran the previous script and saved the report)
df_corr = pd.read_csv("sigmoid_high_score_correlations.txt", sep="\t")

# 2. Filter strictly for a specific histone mark to show a clean spatial trend
# Example: Isolating H3K4me3 bins
df_h3k4 = df_corr[df_corr["Feature"].str.contains("h3k27ac")].copy()

# 3. Extract the numeric bin number from the feature string names
df_h3k4["Bin_Number"] = df_h3k4["Feature"].str.extract(r"bin_(\d+)").astype(int)
df_h3k4 = df_h3k4.sort_values(by="Bin_Number").reset_index(drop=True)

# 4. Generate the Vector Visualization
plt.figure(figsize=(7, 4.5))

# Plot the continuous correlation line
plt.plot(df_h3k4["Bin_Number"], df_h3k4["Spearman_Rho"], marker="o", 
         color="#1967d2", linewidth=2, markersize=6, label="H3K4me3")

# Add a critical reference line at Rho = 0 (The Inversion Boundary)
plt.axhline(0, color="black", linestyle="--", alpha=0.6)

# Highlight significance (p <= 0.05) with distinct colored fill or markers
sig_mask = df_h3k4["P_Value"] <= 0.05
plt.scatter(df_h3k4[sig_mask]["Bin_Number"], df_h3k4[sig_mask]["Spearman_Rho"], 
            color="#d93025", s=120, zorder=3, label="Statistically Significant (p <= 0.05)")

# Formatting structural labels
plt.title("Spatial Epigenetic Gradient Relative to TSS\n(Genes with Repression Score > 0.25)", fontsize=10, fontweight="bold")
plt.xlabel("Spatial Coordinates (Bin Number radiating from TSS Core)", fontsize=9)
plt.ylabel("Spearman Correlation ($\rho$) with Sigmoid Score", fontsize=9)
plt.xticks(df_h3k4["Bin_Number"])
plt.grid(True, linestyle=":", alpha=0.5)
plt.legend(fontsize=8, loc="lower left")

plt.tight_layout()
plt.savefig("plots/h3k4me3_spatial_gradient.png", dpi=300)
plt.savefig("plots/h3k4me3_spatial_gradient.pdf")
plt.close()
