import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Generate Density Plot for ML Target Variable (RIG Score)')
    parser.add_argument('--file3', required=True, help='Path to File 3 (e.g., ALKE_sgRNA_with_NTC_RIG_Score.txt)')
    parser.add_argument('--prefix', default='ALKE', help='Prefix for the plot title')
    parser.add_argument('--out', default='density_plot_rig_score.png', help='Output image filename')
    args = parser.parse_args()

    # 1. Load Data
    try:
        df = pd.read_csv(args.file3, sep='\t')
    except Exception as e:
        print(f"CRITICAL ERROR loading file: {e}")
        sys.exit(1)

    if 'Score_RIG' not in df.columns:
        print("CRITICAL ERROR: 'Score_RIG' column missing. Feed this script File 3 from the new pipeline.")
        sys.exit(1)

    # 2. Label Groups explicitly
    df['Group'] = df['Gene'].apply(lambda x: 'NTC' if 'NO-TARGET' in str(x).upper() else 'Hits')

    # 3. Create the Density Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    palette_map = {'Hits': '#e74c3c', 'NTC': '#7f8c8d'}
    
    # We turn off Seaborn's automatic legend to prevent conflicts
    sns.kdeplot(
        data=df, 
        x='Score_RIG', 
        hue='Group', 
        hue_order=['Hits', 'NTC'], 
        fill=True, 
        common_norm=False, 
        palette=palette_map, 
        alpha=0.5, 
        linewidth=2.5,
        legend=False 
    )

    # 4. Add Baseline Reference
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7)
    
    # 5. EXPLICIT CUSTOM LEGEND
    legend_elements = [
        Patch(facecolor='#e74c3c', edgecolor='#e74c3c', alpha=0.4, label='Hits (Targeting sgRNAs)'),
        Patch(facecolor='#7f8c8d', edgecolor='#7f8c8d', alpha=0.4, label='NTC (Non-Targeting Controls)'),
        Line2D([0], [0], color='black', linestyle='--', alpha=0.7, label='NTC Baseline (0.5)')
    ]
    # Moved to 'upper left' to avoid the high-scoring data peaks
    plt.legend(handles=legend_elements, loc='upper left', fontsize=11, frameon=True, shadow=True)
    # 6. Formatting
    plt.title(f'Distribution of ML Target Variable: {args.prefix} Editor', fontsize=16, fontweight='bold')
    plt.xlabel('Sigmoid Score (Raw RIG Survival)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(-0.1, 1.1) 

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print("-" * 50)
    print(f"Fixed plot saved to: {args.out}")
    print("-" * 50)

if __name__ == "__main__":
    main()


