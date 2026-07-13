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

# 5. Compute FDR (Benjamini-Hochberg) globally across all features
# Note: Cleaning out potential NaNs prevents global vector poisoning
mw_df = mw_df.dropna(subset=["P_Value"]).reset_index(drop=True)
mw_df["FDR_Corrected_P_Value"] = multipletests(mw_df["P_Value"], method="fdr_bh")[1]

# 6. Sort strictly by RAW P_Value to order by highest initial significance
mw_df = mw_df.sort_values(by="P_Value").reset_index(drop=True)

# Save the full comprehensive report before filtering for plotting
mw_df.to_csv("mann_whitney_raw_vs_fdr_report.txt", sep="\t", index=False)

# 7. CRITICAL FILTER: Isolate ONLY features where raw p-value <= 0.05
mw_sig = mw_df[mw_df["P_Value"] <= 0.05].reset_index(drop=True)
num_sig_features = len(mw_sig)

print("==================================================")
print("             STATISTICAL FILTER SUMMARY           ")
print("==================================================")
print(f"Total Features Analyzed         : {len(mw_df)}")
print(f"Significant Features (p <= 0.05): {num_sig_features}")
print("==================================================")

# 8. Conditional Plotting Logic
if num_sig_features == 0:
    print("WARNING: Zero features met the raw p-value <= 0.05 threshold.")
    print("Skipping plot generation to avoid empty file artifacts.")
else:
    # Dynamically compute rows required for a fixed 3-column layout grid
    cols = 3
    rows = (num_sig_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.8 * rows), sharex=False)
    axes = axes.flatten()

    # Loop strictly through the significant subset
    for i in range(num_sig_features):
        row_data = mw_sig.iloc[i]
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

        # Format Raw P-Value string
        raw_p_text = f"raw p = {raw_p:.2e}" if raw_p < 0.001 else f"raw p = {raw_p:.4f}"
        # Format FDR P-Value string
        fdr_p_text = f"FDR p = {fdr_p:.2e}" if fdr_p < 0.001 else f"FDR p = {fdr_p:.4f}"

        # Display both parameters directly in the title
        axes[i].set_title(f"{clean_title}\n({raw_p_text})\n({fdr_p_text})", fontsize=9, fontweight="bold")
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)

    # Prune any unused empty axis panels from the layout grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save outputs in dual formats for the HPC cluster
    png_path = os.path.join("plots", "significant_features_only_boxplots.png")
    pdf_path = os.path.join("plots", "significant_features_only_boxplots.pdf")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    print("Execution complete. Significant differentiators mapped.")
    print(f"  -> View PNG grid: '{png_path}'")
    print(f"  -> View PDF grid: '{pdf_path}'")
