import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate Density Plot for sgRNA Hits + NTC')
    parser.add_argument('--file3', required=True, help='Path to File3 (sgRNA_with_NTC_Delta.txt)')
    parser.add_argument('--prefix', default='ALKH', help='Prefix for the plot title')
    parser.add_argument('--out', default='density_plot.png', help='Output image filename')
    args = parser.parse_args()

    # 1. Load File 3
    df = pd.read_csv(args.file3, sep='\t')

    # 2. Label the groups for plotting
    # NTCs are identified by the 'NO-TARGET' string, everything else is a 'Hit'
    df['Group'] = df['Gene'].apply(lambda x: 'NTC' if 'NO-TARGET' in str(x).upper() else 'Hit')

    # 3. Create the Density Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # KDE (Kernel Density Estimate) plot
    sns.kdeplot(data=df, x='Delta_Score', hue='Group', fill=True, common_norm=False, 
                palette={'NTC': '#7f8c8d', 'Hit': '#e74c3c'}, alpha=0.5, linewidth=2)

    # 4. Add Statistical Reference Lines
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.6, label='Baseline (0)')
    
    # 5. Formatting
    plt.title(f'Distribution of Delta Scores: {args.prefix} Editor', fontsize=15)
    plt.xlabel('Sigmoid Delta Score (RIG - DMSO)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(-0.5, 1.0)  # Standard range for Delta Scores
    
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Plot successfully saved to: {args.out}")

if __name__ == "__main__":
    main()
