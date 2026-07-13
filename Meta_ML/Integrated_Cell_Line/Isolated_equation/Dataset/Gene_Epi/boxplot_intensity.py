import os
import pandas as pd

# 1. Force headless Agg backend BEFORE importing pyplot for the cluster
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

# Ensure the plots directory exists on the cluster storage
os.makedirs("plots", exist_ok=True)

# 2. Load the aggregated dataset
file_path = "gene_aggregated_classified.txt"
df = pd.read_csv(file_path, sep="\t")

# 3. CRITICAL FILTER: Restrict strictly to genes with mean_sigmoid_score > 0.25 (Active Repression)
df_high_subset = df[df["mean_sigmoid_score"] > 0.25].reset_index(drop=True)

# 4. Dynamically identify all target epigenetic features
feature_cols = [col for col in df_high_subset.columns if col.startswith("mean_guide_")]

intensity_results = []

# 5. Compute Mann-Whitney U Test across the 0.5 intensity split
for col in feature_cols:
    group_low = df_high_subset[df_high_subset[col] < 0.75]["mean_sigmoid_score"]
    group_high = df_high_subset[df_high_subset[col] >= 0.75]["mean_sigmoid_score"]
    
    # Statistical Safeguard: Ensure both intensity brackets have viable data density
    if len(group_low) < 5 or len(group_high) < 5:
        continue  # Skip highly skewed features to prevent numerical errors
        
    stat, p_val = mannwhitneyu(group_low, group_high, alternative="two-sided")
    
    mean_low = group_low.mean()
    mean_high = group_high.mean()
    delta_sigmoid = mean_high - mean_low  # Positive means high intensity increases repression score
    
    intensity_results.append({
        "Feature": col,
        "Low_Count (<0.5)": len(group_low),
        "High_Count (>=0.5)": len(group_high),
        "P_Value": p_val,
        "Delta_Sigmoid_Score": delta_sigmoid
    })

results_df = pd.DataFrame(intensity_results)

# Clean out any potential NaNs and sort by significance
results_df = results_df.dropna(subset=["P_Value"]).sort_values(by="P_Value").reset_index(drop=True)
results_df.to_csv("high_score_intensity_split_report.txt", sep="\t", index=False)

# 6. CRITICAL FILTER FOR VISUALIZATION: Isolate features with p-value <= 0.05
sig_plots_df = results_df[results_df["P_Value"] <= 0.05].reset_index(drop=True)
num_sig_plots = len(sig_plots_df)

print("==================================================")
print("         ACTIVE SUBSET INTENSITY ANALYSIS         ")
print("==================================================")
print(f"Total Active Genes Checked (>0.25) : {len(df_high_subset)}")
print(f"Features Differentiating Repression: {num_sig_plots} / {len(feature_cols)}")
print("==================================================\n")

if num_sig_plots == 0:
    print("No features met the raw p-value <= 0.05 threshold within this active subset.")
    print("Skipping boxplot generation to prevent empty file errors.")
else:
    # 7. Generate Grid Boxplots for Significant Features Only
    cols = 3
    rows = (num_sig_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.8 * rows), sharex=False)
    axes = axes.flatten()
    
    for i in range(num_sig_plots):
        row_data = sig_plots_df.iloc[i]
        col = row_data["Feature"]
        p_val = row_data["P_Value"]
        delta = row_data["Delta_Sigmoid_Score"]
        
        plot_data_low = df_high_subset[df_high_subset[col] < 0.5]["mean_sigmoid_score"]
        plot_data_high = df_high_subset[df_high_subset[col] >= 0.5]["mean_sigmoid_score"]
        
        # Render the boxplot tracking continuous sigmoid score
        axes[i].boxplot(
            [plot_data_low, plot_data_high],
            labels=["Low (<0.5)", "High (>=0.5)"],
            patch_artist=True,
            boxprops=dict(facecolor="#e8f0fe", color="#1967d2"),
            medianprops=dict(color="#d93025", linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor="gray", markersize=3, alpha=0.4)
        )
        
        clean_title = col.replace("mean_guide_", "")
        p_text = f"p = {p_val:.2e}" if p_val < 0.001 else f"p = {p_val:.4f}"
        
        axes[i].set_title(f"{clean_title}\n({p_text})\nDelta = {delta:.4f}", fontsize=9, fontweight="bold")
        axes[i].set_ylabel("Mean Sigmoid Score", fontsize=8)
        axes[i].grid(axis="y", linestyle="--", alpha=0.5)
        
    # Delete empty trailing subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    
    # Export plots to cluster storage
    png_path = os.path.join("plots", "active_subset_intensity_boxplots.png")
    pdf_path = os.path.join("plots", "active_subset_intensity_boxplots.pdf")
    
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    
    print("HPC plotting task finalized successfully.")
    print(f"  -> Data Report Table: 'high_score_intensity_split_report.txt'")
    print(f"  -> Significant PNG Grid: '{png_path}'")
    print(f"  -> Significant PDF Grid: '{pdf_path}'")
