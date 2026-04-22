import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import sys
import os

def generate_heatmap(matrix_file, output_prefix):
    if not os.path.exists(matrix_file):
        print(f"CRITICAL ERROR: File not found: {matrix_file}")
        sys.exit(1)

    print(f"Loading matrix: {matrix_file}")
    
    try:
        df = pd.read_csv(matrix_file, index_col='gene')
    except ValueError:
        print("ERROR: 'gene' column not found. Ensure your matrix has a 'gene' header.")
        sys.exit(1)

    # Ensure all data is numeric 
    df = df.apply(pd.to_numeric, errors='coerce')

    num_genes, num_editors = df.shape
    print(f"Matrix loaded: {num_genes} Genes x {num_editors} Editors.")

    # Dynamic figure sizing
    fig_height = min(max(8, num_genes * 0.1), 30) 
    plt.figure(figsize=(10, fig_height))

    # 1. Define the Colormap (Blue to Red)
    cmap = sns.color_palette("coolwarm", as_cmap=True)

    # 2. Hardcode the NaN color to Dark Grey
    cmap.set_bad(color='darkgrey')

    print("Rendering heatmap (this may take a moment for large datasets)...")

    # 3. Plot the Heatmap
    ax = sns.heatmap(df, 
                     cmap=cmap, 
                     vmin=0, 
                     vmax=1, 
                     cbar_kws={'label': 'Sigmoid Score (0.0 = Blue, 1.0 = Red)'},
                     yticklabels=(num_genes <= 100))

    # 4. Formatting
    plt.title('CRISPR Multi-Editor Efficiency Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Editor Type', fontsize=14)
    plt.ylabel(f'Genes (n={num_genes})', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # 5. Save the image in BOTH formats
    plt.tight_layout()
    
    png_path = f"{output_prefix}.png"
    pdf_path = f"{output_prefix}.pdf"
    
    # Save PNG for quick viewing/presentations
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    # Save PDF for publication (Vector format)
    # The 'format="pdf"' ensures it's saved as a true vector graphic
    plt.savefig(pdf_path, format="pdf", bbox_inches='tight')
    
    print(f"\nSuccess! Heatmaps saved to:")
    print(f" -> [Raster/Presentation]: {png_path}")
    print(f" -> [Vector/Publication]:  {pdf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a Blue-Red heatmap and save as both PDF and PNG.")
    parser.add_argument("-i", "--input", required=True, help="Path to Traceable_Gene_Matrix.csv")
    
    # Changed from --output to --prefix to handle multiple file extensions natively
    parser.add_argument("-p", "--prefix", default="Final_Editor_Heatmap", help="Base name for outputs (do not include .png or .pdf)")
    
    args = parser.parse_args()
    generate_heatmap(args.input, args.prefix)
