import os
import sys

# ======================================================================
# FORCE HEADLESS HPC RENDER BACKEND (MUST EXECUTE BEFORE PYPLOT INGEST)
# ======================================================================
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
# ======================================================================

import numpy as np
import pandas as pd

def main():
    # ======================================================================
    # 1. CLUSTER WORKSPACE PATH CONFIGURATIONS
    # ======================================================================
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/K562_reanalysis/sgRNA_summary/dmso/ALKE/custom_cutoff/gene_editor/"
    
    # Read the output file generated from the previous processing run
    input_summary_file = os.path.join(working_dir, "gene_editor_summary_matrix.txt")
    
    density_plot_png = os.path.join(working_dir, "gene_editor_overlap_density.png")
    density_plot_pdf = os.path.join(working_dir, "gene_editor_overlap_density.pdf")

    if not os.path.exists(input_summary_file):
        print(f"CRITICAL ERROR: Input summary matrix missing at {input_summary_file}")
        sys.exit(1)

    print(f"[*] Ingesting summary matrix: {input_summary_file}")
    df = pd.read_csv(input_summary_file, sep='\t')
    df.columns = [col.strip() for col in df.columns]

    # Verify column existence
    if 'mean_sigmoid_score' not in df.columns:
        print("CRITICAL ERROR: 'mean_sigmoid_score' column not found in dataset!")
        sys.exit(1)

    # ======================================================================
    # 2. RECONSTRUCT THE THREE SOURCE REGIONS FOR DIAGNOSTIC OVERLAP CHECK
    # ======================================================================
    print("[*] Reconstructing class distributions for overlap diagnosis...")
    
    def reconstruct_groups(score):
        if score > 0.25:
            return 'Class 1 (Score > 0.25)'
        elif score <= 0.05:
            return 'Class 0 (Score <= 0.05)'
        else:
            return 'Intermediate Zone (-1)'

    df['overlap_group'] = df['mean_sigmoid_score'].apply(reconstruct_groups)

    # Calculate metrics for the cluster log output
    counts = df['overlap_group'].value_counts()
    print("\n=== Group Volume Metrics ===")
    for grp, cnt in counts.items():
        print(f" -> {grp:<25}: {cnt} genes")
    print("============================\n")

    # ======================================================================
    # 3. GENERATE HIGH-RESOLUTION DENSITY PLOT
    # ======================================================================
    print("[*] Generating Kernel Density Estimates...")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Definitive palette matching your strict data profiles
    color_map = {
        'Class 0 (Score <= 0.05)': '#2c3e50',  # Dark Midnight Blue
        'Intermediate Zone (-1)': '#e74c3c',   # Deep Crimson
        'Class 1 (Score > 0.25)': '#27ae60'    # Dark Emerald Green
    }

    # Plot continuous density curves
    for label in ['Class 0 (Score <= 0.05)', 'Intermediate Zone (-1)', 'Class 1 (Score > 0.25)']:
        sub_group = df[df['overlap_group'] == label]
        if not sub_group.empty:
            sns.kdeplot(
                data=sub_group['mean_sigmoid_score'],
                fill=True,
                alpha=0.20,
                linewidth=2.5,
                color=color_map[label],
                label=f"{label} (n={len(sub_group)})",
                ax=ax,
                warn_singular=False
            )

    # Add threshold markers to explicitly show the transition margins
    ax.axvline(x=0.05, color='#7f8c8d', linestyle='--', alpha=0.8, lw=1.5, label='Lower Split Threshold (0.05)')
    ax.axvline(x=0.25, color='#7f8c8d', linestyle='--', alpha=0.8, lw=1.5, label='Upper Split Threshold (0.25)')

    # Chart formatting and styling details
    ax.set_title("Overlap Density Verification: Mean Sigmoid Score Distribution", fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel("Mean Sigmoid Score", fontsize=11, fontweight='bold')
    ax.set_ylabel("Density (KDE)", fontsize=11, fontweight='bold')
    ax.set_xlim([-0.02, df['mean_sigmoid_score'].max() + 0.05])
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(loc="upper right", frameon=True, facecolor='white', edgecolor='none')

    # ======================================================================
    # 4. SAVE PLOTS TO DISK
    # ======================================================================
    plt.tight_layout()
    plt.savefig(density_plot_png, dpi=300, bbox_inches='tight')
    plt.savefig(density_plot_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig)
    
    print(f"[✔] Overlap density plots saved successfully:")
    print(f"    -> PNG: {density_plot_png}")
    print(f"    -> PDF: {density_plot_pdf}")

if __name__ == "__main__":
    main()
