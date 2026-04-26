import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Generate Density Plot with Fixed Legend Colors')
    parser.add_argument('--file3', required=True, help='Path to File3 (sgRNA_with_NTC_Delta.txt)')
    parser.add_argument('--prefix', default='ALKH', help='Prefix for the plot title')
    parser.add_argument('--out', default='density_plot_fixed.png', help='Output image filename')
    args = parser.parse_args()

    # 1. Load Data
    df = pd.read_csv(args.file3, sep='\t')

    # 2. Label Groups explicitly
    df['Group'] = df['Gene'].apply(lambda x: 'NTC' if 'NO-TARGET' in str(x).upper() else 'Hits')

    # 3. Create the Density Plot
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # ---------------------------------------------------------
    # FIXED LEGEND LOGIC:
    # hue_order ensures the palette mapping is consistent.
    # Hits will get the first color (Red), NTC will get the second (Grey).
    # ---------------------------------------------------------
    palette_map = {'Hits': '#e74c3c', 'NTC': '#7f8c8d'}
    
    sns.kdeplot(
        data=df, 
        x='Delta_Score', 
        hue='Group', 
        hue_order=['Hits', 'NTC'], # This fixes the legend mapping
        fill=True, 
        common_norm=False, 
        palette=palette_map, 
        alpha=0.5, 
        linewidth=2.5
    )

    # 4. Add Baseline Reference
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.7)
    
    # 5. Formatting
    plt.title(f'Distribution of Delta Scores: {args.prefix} Editor', fontsize=16, fontweight='bold')
    plt.xlabel('Sigmoid Delta Score (RIG - DMSO)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    # Set limits to show negative tails as requested
    plt.xlim(-0.4, 1.0) 

    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Fixed plot saved to: {args.out}")

if __name__ == "__main__":
    main()