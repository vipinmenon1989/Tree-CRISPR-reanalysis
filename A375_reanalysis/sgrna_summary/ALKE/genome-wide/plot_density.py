import pandas as pd
import numpy as np

# Force a non-interactive backend for headless HPC execution
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import sys
import os

def generate_hpc_density_plots(file_path, output_base_name):
    print("======================================================================")
    print("CRISPRi VISUALIZATION ENGINE: HPC HEADLESS PLOT GENERATION")
    print("======================================================================\n")
    
    print(f"[*] Loading visualization matrix from: {file_path}")
    if not os.path.exists(file_path):
        print(f"CRITICAL ERROR: Input file '{file_path}' does not exist in this directory.")
        sys.exit(1)
        
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to read file: {e}")
        sys.exit(1)

    # 1. Set aesthetic theme
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))

    # 2. Define custom high-contrast colors
    custom_palette = {
        'NTC': '#4A6984',          # Slate Blue for baseline noise
        'Genome_Wide_Hit': '#D9383A'  # Crimson for active dropouts
    }

    print("[*] Generating kernel density estimate (KDE) plot overlay...")
    
    # 3. Construct the KDE overlay
    # common_norm=False is mandatory to independently scale NTCs and Hits
    sns.kdeplot(
        data=df, 
        x="Sigmoid_Score", 
        hue="Visualization_Group", 
        palette=custom_palette,
        fill=True, 
        alpha=0.4, 
        linewidth=2.5,
        common_norm=False
    )

    # 4. Fine-tune plot layout and limits
    plt.title("Calibrated Score Distribution: NTC Baseline vs. High-Confidence Essential Hits", fontsize=14, pad=15, fontweight='bold')
    plt.xlabel("Calibrated Sigmoid Score", fontsize=12, labelpad=10)
    plt.ylabel("Density Probability", fontsize=12, labelpad=10)
    
    # Bounded tightly by sigmoid limits
    plt.xlim(-0.05, 1.05) 

    # Clean up legend layout
    plt.legend(
        title="Biological Group", 
        labels=["Verified Essential Hit (sgRNA >= 2)", "Non-Targeting Control (NTC)"],
        loc="upper right", 
        frameon=True, 
        facecolor='white', 
        edgecolor='none'
    )

    plt.tight_layout()
    
    # 5. Export clean files to the cluster
    png_output = f"{output_base_name}.png"
    pdf_output = f"{output_base_name}.pdf"
    
    print("[*] Exporting graphics buffers...")
    try:
        # Save standard high-resolution image for presentations/slides
        plt.savefig(png_output, dpi=300, format='png')
        print(f"--> SUCCESS: Saved high-resolution image: {png_output}")
        
        # Save lossless vector format for publication/figures
        plt.savefig(pdf_output, format='pdf')
        print(f"--> SUCCESS: Saved publication-ready vector file: {pdf_output}")
        
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to write output files to disk: {e}")
        sys.exit(1)
        
    # Clean up the memory figure footprint explicitly
    plt.close(fig)
    print("\n======================================================================")
    print("GRAPHICS COMPLETION SUCCESSFUL. PLOT MEMORY BUFFER FLUSHED.")
    print("======================================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HPC CRISPRi Plotting Utility")
    parser.add_argument("-i", "--input", default="Output_File2_Density_Distribution.csv", help="Input path for file 2")
    parser.add_argument("-o", "--output_prefix", default="CRISPRi_Sigmoid_Density_Plot", help="Base filename for plots (omitting suffix)")
    args = parser.parse_args()
    
    generate_hpc_density_plots(args.input, args.output_prefix)

