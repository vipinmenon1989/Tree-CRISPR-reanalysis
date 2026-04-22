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
        df = pd.read_csv(matrix_file, index_col='gene')
        df = df.apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    # 1. Preprocess for K-Means
    # K-Means cannot handle NaNs. 
    # We drop genes with absolutely no data, and fill remaining NaNs with 0 (assuming missing = no editing).
    df_clean = df.dropna(how='all').fillna(0)
    num_genes = len(df_clean)
    
    print(f"Running K-Means clustering (k={num_clusters}) on {num_genes} genes...")
    
    # 2. Apply K-Means
    # n_init=10 runs the algorithm 10 times with different centroid seeds and picks the best one
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_clean['Cluster'] = kmeans.fit_predict(df_clean)
    
    # 3. Melt dataframe for Seaborn lineplot (Long format)
    df_long = df_clean.reset_index().melt(
        id_vars=['gene', 'Cluster'], 
        var_name='Editor', 
        value_name='Sigmoid_Score'
    )
    
    # 4. Plotting Setup
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    
    # sns.lineplot automatically calculates the mean and 95% Confidence Interval for each group
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
    
    # 5. Formatting & Legend Cleanup
    plt.title(f'K-Means Editor Trajectory Analysis (k={num_clusters})', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Mean Sigmoid Score', fontsize=12)
    plt.xlabel('Editor Type', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.ylim(-0.05, 1.05)
    
    # Update the legend to show how many genes are in each cluster
    handles, labels = ax.get_legend_handles_labels()
    cluster_counts = df_clean['Cluster'].value_counts().sort_index()
    
    new_labels = [f"Cluster {int(l)} (n={cluster_counts[int(l)]})" for l in labels]
    plt.legend(handles, new_labels, title="Behavioral Clusters", loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    # 6. Save Outputs
    png_out = f"{output_prefix}_KMeans_k{num_clusters}.png"
    pdf_out = f"{output_prefix}_KMeans_k{num_clusters}.pdf"
    csv_out = f"{output_prefix}_KMeans_Assigned.csv"
    
    plt.savefig(png_out, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_out, format='pdf', bbox_inches='tight')
    
    # Save the dataframe with the Cluster labels attached
    df_clean.to_csv(csv_out)
    
    print("\nSuccess! Outputs generated:")
    print(f" -> [Plot - PNG]: {png_out}")
    print(f" -> [Plot - PDF]: {pdf_out}")
    print(f" -> [Data Matrix]: {csv_out} (Use this to see which gene belongs to which cluster)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K-Means Trajectory plot and cluster assignments.")
    parser.add_argument("-i", "--input", required=True, help="Traceable_Gene_Matrix.csv")
    parser.add_argument("-k", "--clusters", type=int, default=5, help="Number of clusters (default: 5)")
    parser.add_argument("-p", "--prefix", default="Editor_Trajectory", help="Output prefix")
    
    args = parser.parse_args()
    generate_kmeans_trajectory(args.input, args.clusters, args.prefix)
