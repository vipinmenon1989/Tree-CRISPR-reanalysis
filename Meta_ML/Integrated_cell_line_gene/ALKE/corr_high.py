import os
import pandas as pd

# 1. Force headless Agg backend BEFORE importing pyplot for the cluster
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

# Ensure output directory paths exist on the cluster file system
os.makedirs("plots", exist_ok=True)

# 2. Load the essential gene feature matrix
file_path = "essential_gene.txt"
print("Reading essential gene feature matrix...")
df = pd.read_csv(file_path, sep="\t")

# 3. CRITICAL SUBSETTING: Isolate highly active genes (Score > 0.25)
print("Filtering rows where mean_sigmoid_score > 0.25...")
df_high = df[df["mean_sigmoid_score"] > 0.25].reset_index(drop=True)

# 4. EXPLICIT FEATURE EXTRACTION: Dynamically capture all 70 spatial bins
# Selects columns that contain '_bin_' but excludes baseline ID, gene, score, or class columns
feature_cols = [col for col in df_high.columns if "_bin_" in col and col not in ["ID", "gene", "mean_sigmoid_score", "class"]]

print("==================================================")
print("         CORRELATION RUN INITIALIZED              ")
print("==================================================")
print(f"Total Active Genes Checked (>0.25) : {len(df_high)}")
print(f"Total Epigenetic Bins Mapped     : {len(feature_cols)}")
print("==================================================\n")

# 5. Compute Pairwise Spearman Correlations against the Target Score
correlation_results = []
target_variable = df_high["mean_sigmoid_score"]

for col in feature_cols:
    feature_variable = df_high[col]
    
    # Statistical Safeguard: Skip any dead invariant columns to prevent NaN generation
    if feature_variable.nunique() <= 1:
        print(f"Skipping invariant column with zero variance: {col}")
        continue
        
    rho, p_val = spearmanr(feature_variable, target_variable)
    
    correlation_results.append({
        "Feature": col,
        "Spearman_Rho": rho,
        "P_Value": p_val
    })

# Convert to DataFrame and IMMEDIATELY write out raw results table
corr_df = pd.DataFrame(correlation_results)
report_output = "essential_genes_high_score_correlations.txt"
corr_df.to_csv(report_output, sep="\t", index=False)
print(f"CRITICAL STEP COMPLETE: Raw calculations saved to -> '{report_output}'\n")

# 6. Parse String Column Components for Matrix Alignment
# Extract numeric bin number suffix (1 to 10)
corr_df["Bin_Number"] = corr_df["Feature"].str.extract(r"bin_(\d+)").astype(int)

# Extract base epigenetic feature name by dropping the trailing '_bin_X'
def extract_base_name(feature_name):
    return feature_name.split("_bin_")[0]

corr_df["Base_Feature"] = corr_df["Feature"].apply(extract_base_name)

# 7. Pivot Table Arrays into Standardized 10-Column Spatial Grids
# Rows = Distinct Epigenetic Modalities, Columns = Spatial Coordinates Bin 1 to 10
rho_pivot = corr_df.pivot(index="Base_Feature", columns="Bin_Number", values="Spearman_Rho")
p_pivot = corr_df.pivot(index="Base_Feature", columns="Bin_Number", values="P_Value")

# Sort row features alphabetically so families (e.g., h3k4) group together cleanly
rho_pivot = rho_pivot.sort_index()
p_pivot = p_pivot.sort_index()

total_rows = len(rho_pivot)

# 8. Generate the Expanded 70-Feature Row Landscape Heatmap
fig, ax = plt.subplots(figsize=(13.5, 2 + (0.45 * total_rows)))

# Render continuous trajectories using a divergent color map (Red=Positive, Blue=Negative)
cax = ax.imshow(rho_pivot.values, cmap="bwr", vmin=-0.25, vmax=0.25, aspect="auto")

# Horizontal bottom colorbar layout
cbar = fig.colorbar(cax, orientation="horizontal", pad=0.08, shrink=0.4)
cbar.set_label("Spearman Correlation Coefficient ($\\rho$) with Repression Level", fontsize=9, fontweight="bold")

# Align axis coordinates to look like a genome browser layout
ax.set_xticks(np.arange(rho_pivot.shape[1]))
ax.set_xticklabels([f"Bin {c}" for c in rho_pivot.columns], fontsize=9, fontweight="bold")
ax.set_yticks(np.arange(rho_pivot.shape[0]))

# Clean up raw feature string names for presentation row labels
clean_row_labels = [r.replace("_", " ").upper() for r in rho_pivot.index]
ax.set_yticklabels(clean_row_labels, fontsize=8, fontweight="bold")

# 9. OVERLAY TEXT VALUES AND SIGNIFICANCE ASTERISKS NATIVELY
for (i, j), p_val in np.ndenumerate(p_pivot.values):
    rho_val = rho_pivot.values[i, j]
    
    if np.isnan(rho_val):
        continue
        
    # Contrast color adjustment based on cell density background shade
    text_color = "white" if abs(rho_val) > 0.12 else "black"
    
    if p_val <= 0.05:
        # Boldly flag statistically verified functional coordinates with an asterisk (*)
        # Employs yellow text on blue cells (negative) and purple text on red cells (positive) for visibility
        highlight_color = "yellow" if rho_val < 0 else "purple"
        ax.text(j, i, f"{rho_val:.2f}*", ha="center", va="center", 
                color=highlight_color, fontsize=8.5, fontweight="black")
    else:
        # Fades away weak background features to preserve image contrast on true signals
        ax.text(j, i, f"{rho_val:.2f}", ha="center", va="center", 
                color=text_color, fontsize=7.5, alpha=0.35)

plt.title("Essential Genes Landscape Profile (All Spatial Feature Tracks Checked | Score > 0.25)\n* Bolded Cell Entry Highlights Statistically Significant Window (p <= 0.05)", 
          fontsize=11, fontweight="bold", pad=20)

plt.tight_layout()

# 10. Save plots in dual cluster-ready graphics formats
png_path = os.path.join("plots", "essential_70_row_landscape_heatmap.png")
pdf_path = os.path.join("plots", "essential_70_row_landscape_heatmap.pdf")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print("==================================================")
print("         PIPELINE TASK EXECUTED SUCCESSFULLY      ")
print("==================================================")
print(f"Calculated Data Matrix Output  : '{report_output}'")
print(f"Saved Complete PNG Visual Map  : '{png_path}'")
print(f"Saved Complete PDF Vector Map  : '{pdf_path}'")
print("==================================================")
