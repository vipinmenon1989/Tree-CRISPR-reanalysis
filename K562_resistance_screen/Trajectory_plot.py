import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import argparse
import sys

def generate_kmeans_trajectory(matrix_file, num_clusters, output_prefix):
    if not os.path.exists(matrix_file):
        print(f"CRITICAL ERROR: File not found: {matrix_file}")
        sys.exit(1)

    print(f"Loading matrix: {matrix_file}")
    
    try:
        # 1. FIXED HEADERS: Using 'Gene' as the index column
        df = pd.read_csv(matrix_file, index_col='Gene')
        df = df.apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        print(f"CRITICAL ERROR reading matrix: {e}")
        sys.exit(1)

    # 2. Preprocess for K-Means
    # K-Means cannot handle NaNs. We fill NaNs with 0.
    # Since we dropped ALKML/ALKME, this will cluster ALKH and ALKE behaviors.
    df_clean = df.dropna(how='all').fillna(0)
    num_genes = len(df_clean)
    
    if num_genes < num_clusters:
        print(f"ERROR: Number of genes ({num_genes}) is less than requested clusters ({num_clusters}).")
        sys.exit(1)
    
    print(f"Running K-Means clustering (k={num_clusters}) on {num_genes} genes...")
    
    # 3. Apply K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(df_clean)
    
    # 4. Melt dataframe for Seaborn lineplot (Long format)
    # Using 'Delta_Score' as the generic value name
    df_long = df_clean.reset_index().melt(
        id_vars=['Gene', 'Cluster'], 
        var_name='Editor', 
        value_name='Delta_Score'
    )
    
    # 5. Plotting Setup
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # The lineplot shows the mean behavior of genes in each cluster across editors
    ax = sns.lineplot(
        data=df_long, 
        x='Editor', 
        y='Delta_Score', 
        hue='Cluster', 
        palette='viridis', 
        linewidth=3,
        marker='o',
        markersize=10
    )
    
    # 6. Formatting & Legend
    plt.title(f'K-Means Behavior Trajectory (k={num_clusters})', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Mean Delta Score', fontsize=12)
    plt.xlabel('Editor Condition', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    
    # Update legend with counts
    handles, labels = ax.get_legend_handles_labels()
    cluster_counts = df_clean['Cluster'].value_counts().sort_index()
    
    new_labels = [f"Cluster {int(l)} (n={cluster_counts[int(l)]})" for l in labels]
    plt.legend(handles, new_labels, title="Gene Response Clusters", loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    # 7. Save Outputs
    png_out = f"{output_prefix}_KMeans_k{num_clusters}.png"
    csv_out = f"{output_prefix}_KMeans_Assignments.csv"
    
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    df_clean.to_csv(csv_out)
    
    print(f"\nSuccess! Plot saved to: {png_out}")
    print(f"Cluster assignments saved to: {csv_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K-Means Trajectory plot from Gene Matrix.")
    parser.add_argument("-i", "--input", required=True, help="Path to Master_Editor_Comparison_Matrix.csv")
    parser.add_argument("-k", "--clusters", type=int, default=3, help="Number of clusters (suggested: 3-5)")
    parser.add_argument("-p", "--prefix", default="Editor_Behavior", help="Output prefix")
    
    args = parser.parse_args()
    generate_kmeans_trajectory(args.input, args.clusters, args.prefix)
