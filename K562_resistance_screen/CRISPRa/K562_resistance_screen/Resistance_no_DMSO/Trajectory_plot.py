import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import argparse
import sys
import numpy as np

def generate_visualizations(matrix_file, num_clusters, output_prefix):
    if not os.path.exists(matrix_file):
        print(f"CRITICAL ERROR: File not found: {matrix_file}")
        sys.exit(1)

    print(f"Loading Sigmoid matrix: {matrix_file}")
    
    try:
        df = pd.read_csv(matrix_file)
        
        # Dynamic Gene column detection
        cols = [c for c in df.columns if c.lower() == 'gene']
        if not cols:
            print(f"CRITICAL ERROR: No 'Gene' column found.")
            sys.exit(1)
            
        gene_col_name = cols[0]
        df = df.set_index(gene_col_name)
        df = df.apply(pd.to_numeric, errors='coerce')
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    # 1. PREPROCESS & NaN HANDLING
    # Drop rows that are entirely empty, then fill ALL remaining NaNs with 0
    df_filtered = df.dropna(how='all').fillna(0)
    
    # 2. K-MEANS CLUSTERING
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df_filtered['Cluster'] = kmeans.fit_predict(df_filtered)
    df_filtered = df_filtered.sort_values('Cluster')

    # --- PLOT 1: HEATMAP (BLUE-TO-RED) ---
    print("Generating Heatmap...")
    plot_data = df_filtered.drop(columns=['Cluster'])
    
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # cmap="RdYlBu_r" provides Blue (0) -> Yellow -> Red (1)
    sns.heatmap(
        plot_data, 
        cmap="RdYlBu_r", 
        vmin=0, vmax=1,
        ax=ax,
        cbar_kws={'label': 'Sigmoid Score (0-1)'},
        xticklabels=True
    )
    
    plt.title(f'K-Means Gene Heatmap (k={num_clusters})\nBlue=0, Red=1', fontsize=18, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    
    # Save Heatmap as both PNG and PDF
    plt.savefig(f"{output_prefix}_Heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}_Heatmap.pdf", bbox_inches='tight')
    plt.close()

    # --- PLOT 2: TRAJECTORY (BOLD STYLE) ---
    print("Generating Trajectory Plot...")
    df_long = df_filtered.reset_index().melt(
        id_vars=[gene_col_name, 'Cluster'], 
        var_name='Editor', 
        value_name='Score'
    )
    
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    ax_traj = sns.lineplot(
        data=df_long, 
        x='Editor', 
        y='Score', 
        hue='Cluster', 
        style='Cluster',
        palette='tab10', 
        linewidth=5,
        marker='o',
        markersize=12,
        dashes=False,
        errorbar=('ci', 95)
    )
    
    ax_traj.set_ylim(-0.02, 1.02)
    
    counts = df_filtered['Cluster'].value_counts().sort_index()
    handles, labels = ax_traj.get_legend_handles_labels()
    new_labels = [f"Cluster {l} (n={counts[int(l)]})" for l in labels]
    plt.legend(handles, new_labels, title="Clusters", loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.title('K-Means Editor Trajectory Analysis', fontsize=20, fontweight='bold')
    plt.ylabel('Mean Sigmoid Score', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    
    # Save Trajectory as both PNG and PDF
    plt.savefig(f"{output_prefix}_Trajectory.png", dpi=300)
    plt.savefig(f"{output_prefix}_Trajectory.pdf")
    plt.close()

    # Save assignments
    df_filtered.to_csv(f"{output_prefix}_Assignments.csv")
    print(f"Success! PNG, PDF, and CSV generated with prefix: {output_prefix}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-k", "--clusters", type=int, default=5)
    parser.add_argument("-p", "--prefix", default="Sigmoid_Analysis_Final")
    args = parser.parse_args()
    generate_visualizations(args.input, args.clusters, args.prefix)
