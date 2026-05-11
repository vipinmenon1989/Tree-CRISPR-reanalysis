import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import argparse
import sys
import os

def generate_kmeans_trajectory(matrix_file, num_clusters, output_prefix):
    if not os.path.exists(matrix_file):
        print(f"CRITICAL ERROR: File not found: {matrix_file}")
        sys.exit(1)

    print(f"Loading matrix: {matrix_file}")
    
    try:
        # Load data
        df = pd.read_csv(matrix_file)
        
        # Normalize columns to lowercase to handle 'Gene' vs 'gene'
        df.columns = [c.lower() for c in df.columns]
        
        if 'gene' not in df.columns:
            print("CRITICAL ERROR: 'gene' column not found in matrix.")
            sys.exit(1)
            
        df = df.set_index('gene')
        df = df.apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    # 1. Preprocess for K-Means
    # Drop rows that are entirely NaN, then fill remaining NaNs with 0 for clustering math
    df_clean = df.dropna(how='all').fillna(0)
    num_genes = len(df_clean)
    
    print(f"Running K-Means clustering (k={num_clusters}) on {num_genes} genes...")
    
    # 2. Apply K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(df_clean)
    
    # Sort by Cluster so the heatmap and assignments are organized
    df_clean = df_clean.sort_values('Cluster')

    # --- 3. HEATMAP GENERATION (CLUBBING GENES BY CLUSTER) ---
    print("Generating Clustered Heatmap...")
    
    # Create a color sidebar to "club" the K-Means groups
    cluster_palette = sns.color_palette("tab10", n_colors=num_clusters)
    cluster_colors = df_clean['Cluster'].map(lambda x: cluster_palette[x])

    heatmap_data = df_clean.drop(columns=['Cluster'])
    
    # Using clustermap but disabling row_cluster to preserve K-Means order
    g = sns.clustermap(
        heatmap_data,
        row_cluster=False,         # Keep K-Means assignments as the primary order
        col_cluster=True,          # Hierarchically group similar Editors
        row_colors=cluster_colors, # Visual sidebar grouping the genes
        cmap="RdYlBu_r",
        vmin=0, vmax=1,
        figsize=(12, 14),
        cbar_kws={'label': 'Mean Sigmoid Score'},
        xticklabels=True,
        yticklabels=False          # Hide gene names for clarity; set to True if needed
    )

    g.fig.suptitle(f'K-Means Gene Grouping (k={num_clusters})', fontsize=16, fontweight='bold', y=1.02)
    
    # Save Heatmap
    g.savefig(f"{output_prefix}_Heatmap_k{num_clusters}.png", dpi=300, bbox_inches='tight')
    g.savefig(f"{output_prefix}_Heatmap_k{num_clusters}.pdf", format='pdf', bbox_inches='tight')

    # --- 4. TRAJECTORY PLOT GENERATION ---
    print("Generating Trajectory Plot...")
    
    # Melt for Seaborn lineplot
    df_long = df_clean.reset_index().melt(
        id_vars=['gene', 'Cluster'], 
        var_name='Editor', 
        value_name='Sigmoid_Score'
    )
    
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    ax = sns.lineplot(
        data=df_long, 
        x='Editor', 
        y='Sigmoid_Score', 
        hue='Cluster', 
        palette='tab10', 
        linewidth=2.5,
        marker='o',
        markersize=8
    )
    
    plt.title(f'K-Means Editor Trajectory Analysis (k={num_clusters})', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Mean Sigmoid Score', fontsize=12)
    plt.xlabel('Editor Type', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.ylim(-0.05, 1.05)
    
    # Cleanup legend to show N counts per cluster
    handles, labels = ax.get_legend_handles_labels()
    cluster_counts = df_clean['Cluster'].value_counts().sort_index()
    new_labels = [f"Cluster {int(l)} (n={cluster_counts[int(l)]})" for l in labels]
    plt.legend(handles, new_labels, title="Behavioral Clusters", loc='upper left', bbox_to_anchor=(1, 1))
    
    # Save Trajectory
    plt.savefig(f"{output_prefix}_Trajectory_k{num_clusters}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Trajectory_k{num_clusters}.pdf", format='pdf', bbox_inches='tight')
    plt.close()

    # 5. Save cluster assignments
    df_clean.to_csv(f"{output_prefix}_KMeans_Assigned.csv")
    
    print(f"\nSuccess! Final files saved with prefix: {output_prefix}")
    print(f"Cluster Assignments: {output_prefix}_KMeans_Assigned.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K-Means Trajectory and Clubbed Heatmap.")
    parser.add_argument("-i", "--input", required=True, help="Traceable_Gene_Matrix.csv")
    parser.add_argument("-k", "--clusters", type=int, default=5, help="Number of clusters (default: 5)")
    parser.add_argument("-p", "--prefix", default="Editor_Analysis", help="Output prefix")
    
    args = parser.parse_args()
    generate_kmeans_trajectory(args.input, args.clusters, args.prefix)

