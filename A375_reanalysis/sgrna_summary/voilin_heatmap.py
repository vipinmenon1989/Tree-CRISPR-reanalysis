import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys

def generate_advanced_plots(matrix_file, prefix):
    print(f"Loading matrix: {matrix_file}")
    df = pd.read_csv(matrix_file, index_col='gene')
    df = df.apply(pd.to_numeric, errors='coerce')

    num_genes, num_editors = df.shape
    print(f"Loaded {num_genes} genes across {num_editors} editors.")

    # ==========================================
    # PLOT 1: The Violin Plot (Distribution)
    # ==========================================
    print("Generating Violin Plot...")
    plt.figure(figsize=(12, 6))
    
    # We must 'melt' the dataframe from Wide to Long format for Seaborn violins
    df_long = df.reset_index().melt(id_vars='gene', var_name='Editor', value_name='Sigmoid_Score')
    
    # Drop NaNs just for the violin calculation
    df_long = df_long.dropna(subset=['Sigmoid_Score'])

    sns.violinplot(data=df_long, x='Editor', y='Sigmoid_Score', 
                   palette='coolwarm', inner='quartile', scale='width')
    
    plt.title(f'Editor Efficiency Distribution Across {num_genes} Genes', fontsize=16, fontweight='bold')
    plt.ylabel('Sigmoid Score', fontsize=12)
    plt.xlabel('Editor Type', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig(f"{prefix}_Violin.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{prefix}_Violin.png", dpi=300, bbox_inches='tight')

    # ==========================================
    # PLOT 2: The Clustered Heatmap
    # ==========================================
    print("Generating Clustered Heatmap...")
    
    # Clustermap cannot handle NaNs during the clustering math. 
    # Standard practice is to fill NaNs with 0 (assuming missing = no activity) 
    # OR drop rows with NaNs. Here we drop rows that have NO data, and fill the rest with 0.
    df_clust = df.dropna(how='all').fillna(0)
    
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    # Generate the clustered heatmap
    cg = sns.clustermap(df_clust, 
                        cmap=cmap, 
                        vmin=0, vmax=1,
                        metric='euclidean', 
                        method='ward',      # Ward variance minimization creates clean blocks
                        col_cluster=True,   # Clusters the editors based on similarity
                        row_cluster=True,   # Clusters the 610 genes into blocks
                        yticklabels=False,  # Force hide the 610 gene names to prevent clutter
                        figsize=(10, 12))
    
    cg.ax_heatmap.set_title('Hierarchically Clustered Gene Editing Profiles', fontweight='bold', pad=20)
    
    plt.savefig(f"{prefix}_Clustered_Heatmap.pdf", format='pdf', bbox_inches='tight')
    plt.savefig(f"{prefix}_Clustered_Heatmap.png", dpi=300, bbox_inches='tight')
    
    print("\nSuccess! Generated:")
    print(f" 1. {prefix}_Violin.png / .pdf")
    print(f" 2. {prefix}_Clustered_Heatmap.png / .pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to Traceable_Gene_Matrix.csv")
    parser.add_argument("-p", "--prefix", default="Editor_Comparison", help="Output prefix")
    args = parser.parse_args()
    
    generate_advanced_plots(args.input, args.prefix)

