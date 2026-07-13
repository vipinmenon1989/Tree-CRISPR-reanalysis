import os
import pandas as pd

# 1. Force headless Agg backend BEFORE importing pyplot for the cluster
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

# Ensure the plots directory exists on the cluster storage
os.makedirs("plots", exist_ok=True)

# 2. Load your saved correlation data
df_corr = pd.read_csv("sigmoid_high_score_correlations.txt", sep="\t")

# 3. Filter strictly for the core histone modifications to keep the track clean
target_marks = ["h3k4me3", "h3k27ac", "h3k4me2", "h3k4me1"]
df_tracks = df_corr[df_corr["Feature"].str.contains("|".join(target_marks))].copy()

# 4. Clean and parse spatial coordinates and mark names
df_tracks["Bin_Number"] = df_tracks["Feature"].str.extract(r"bin_(\d+)").astype(int)
df_tracks["Mark_Type"] = df_tracks["Feature"].str.replace("mean_guide_", "").str.split("_bin_").str[0]

# 5. Pivot data into explicit spatial grids (Rows = Marks, Columns = Bin 1 to 10)
rho_pivot = df_tracks.pivot(index="Mark_Type", columns="Bin_Number", values="Spearman_Rho")
p_pivot = df_tracks.pivot(index="Mark_Type", columns="Bin_Number", values="P_Value")

# Reorder rows to match biological proximity/logic
rho_pivot = rho_pivot.reindex(target_marks)
p_pivot = p_pivot.reindex(target_marks)

# 6. Generate the Genomic Heatmap Track
fig, ax = plt.subplots(figsize=(10, 3.5))

# Plot continuous correlation scales using a divergent color map (Red=Positive, Blue=Negative)
cax = ax.imshow(rho_pivot.values, cmap="bwr", vmin=-0.20, vmax=0.20, aspect="auto")

# Add standard colorbar scaling
cbar = fig.colorbar(cax, orientation="horizontal", pad=0.2, shrink=0.6)
cbar.set_label("Spearman Correlation Coefficient ($\\rho$) with Repression Score", fontsize=9, fontweight="bold")

# Align axis labels to reflect standard genomic tracks
ax.set_xticks(np.arange(rho_pivot.shape[1]))
ax.set_xticklabels([f"Bin {c}" for c in rho_pivot.columns], fontsize=9, fontweight="bold")
ax.set_yticks(np.arange(rho_pivot.shape[0]))
ax.set_yticklabels(rho_pivot.index, fontsize=9, fontweight="bold")

# 7. CRITICAL ADDITION: Overlay significance indicators directly onto the spatial map
for (i, j), p_val in np.ndenumerate(p_pivot.values):
    rho_val = rho_pivot.values[i, j]
    
    # Text color optimization based on color background density
    text_color = "white" if abs(rho_val) > 0.12 else "black"
    
    if p_val <= 0.05:
        # Boldly label statistically real modulatory windows with an asterisk or text
        ax.text(j, i, f"{rho_val:.2f}*", ha="center", va="center", 
                color="yellow" if rho_val < 0 else "purple", 
                fontsize=10, fontweight="black")
    else:
        # Display weak background trends transparently
        ax.text(j, i, f"{rho_val:.2f}", ha="center", va="center", 
                color=text_color, fontsize=8, alpha=0.4)

plt.title("Spatial Epigenetic Landscape Map (Effective Repression Subset > 0.25)\n* Indicates Statistically Significant Modification Window (p <= 0.05)", 
          fontsize=10, fontweight="bold", pad=15)

plt.tight_layout()

# 8. Dual-Format HPC Export
png_path = os.path.join("plots", "spatial_genomic_tracks_heatmap.png")
pdf_path = os.path.join("plots", "spatial_genomic_tracks_heatmap.pdf")

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print("Genomic Track Visualization Task Finalized.")
print(f"  -> View Spatial Track PNG: '{png_path}'")
print(f"  -> View Spatial Track PDF: '{pdf_path}'")
