import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Ensure the plots directory exists on the cluster
os.makedirs("plots", exist_ok=True)

# 1. Load the classified dataset
file_path = "gene_aggregated_classified.txt"
df = pd.read_csv(file_path, sep="\t")

# 2. Drop intermediate class (-1) to isolate strict binary conditions
df_model = df[df["class"] != -1].reset_index(drop=True)

# 3. Dynamically identify all target epigenetic features
feature_cols = [col for col in df_model.columns if col.startswith("mean_guide_")]

# 4. Compute Mann-Whitney U Test and Rank-Biserial Correlation (Effect Size)
mw_results = []
for col in feature_cols:
    group_0 = df_model[df_model["class"] == 0][col]
    group_1 = df_model[df_model["class"] == 1][col]

    stat, p_val = mannwhitneyu(group_0, group_1, alternative="two-sided")
    
    n0, n1 = len(group_0), len(group_1)
    effect_size = 1 - (2 * stat) / (n0 * n1)

    mw_results.append({
        "Feature": col,
        "U_Statistic": stat,
        "P_Value": p_val,
        "Effect_Size_r": effect_size,
        "Mean_Class_0": group_0.mean(),
        "Mean_Class_1": group_1.mean()
    })

mw_df = pd.DataFrame(mw_results)

# 5. Compute FDR (Benjamini-Hochberg) globally across all 70 features
mw_df["FDR_Corrected_P_Value"] = multipletests(mw_df["P_Value"], method="fdr_bh")[1]

# 6. CRITICAL: Sort strictly by RAW P_Value to guarantee we pull the strongest signals
mw_df = mw_df.sort_values(by="P_Value").reset_index(drop=True)

# Save full report
mw_df.to_csv("mann_whitney_raw_vs_fdr_report.txt", sep="\t", index=False)

# 7. Generate Grid Boxplots for the Top 12 Features based on Raw Significance
top_n = min(12, len(mw_df))
cols = 3
rows = (top_n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(14, 3.8 * rows), sharex=False)
axes = axes.flatten()

for i in range(top_n):
    row_data = mw_df.iloc[i]
    col = row_data["Feature"]
    raw_p = row_data["P_Value"]
    fdr_p = row_data["FDR_Corrected_P_Value"]

    data_to_plot = [
        df_model[df_model["class"] == 0][col],
        df_model[df_model["class"] == 1][col]
    ]

    axes[i].boxplot(
        data_to_plot,
        labels=["Class 0", "Class 1"],
        patch_artist=True,
        boxprops=dict(facecolor="#d0e1f9", color="#1e3d59"),
        medianprops=dict(color="#ff6e40", linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=3, alpha=0.5)
    )

    clean_title = col.replace("mean_guide_", "")

    # Format Raw P-Value
    raw_p_text = f"raw p = {raw_p:.2e}" if raw_p < 0.001 else f"raw p = {raw_p:.4f}"
    # Format FDR P-Value
    fdr_p_text = f"FDR p = {fdr_p:.2e}" if fdr_p < 0.001 else f"FDR p = {fdr_p:.4f}"

    # Print both directly on the subplot title for comparison
    axes[i].set_title(f"{clean_title}\n({raw_p_text})\n({fdr_p_text})", fontsize=9, fontweight="bold")
    axes[i].grid(axis="y", linestyle="--", alpha=0.5)

# Prune empty axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

# Save in both formats for the HPC plots directory
png_path = os.path.join("plots", "features_raw_vs_fdr_boxplots.png")
pdf_path = os.path.join("plots", "features_raw_vs_fdr_boxplots.pdf")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print("Execution complete. Active differentiators mapped.")
print(f"  -> View full text stats: 'mann_whitney_raw_vs_fdr_report.txt'")
print(f"  -> View PNG grid: '{png_path}'")
print(f"  -> View PDF grid: '{pdf_path}'")
