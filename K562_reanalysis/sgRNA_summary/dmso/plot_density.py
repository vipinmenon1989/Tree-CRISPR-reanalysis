import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_sigmoid_density(matrix_file, control_label, output_image):
    print(f"Loading scored matrix: {matrix_file}")
    df = pd.read_csv(matrix_file)
    
    # Separate the data
    ntc_mask = df['Gene'].str.contains(control_label, case=False, na=False)
    df_ntc = df[ntc_mask]
    df_targets = df[~ntc_mask]
    
    # Extract the threshold used
    # Assuming all rows have the same threshold from the previous script
    threshold = df['Dynamic_Threshold'].iloc[0] if 'Dynamic_Threshold' in df.columns else 0.8377
    
    print(f"Plotting {len(df_ntc)} NTCs vs {len(df_targets)} Targeting Guides...")
    
    # Setup the plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Plot KDE (Kernel Density Estimate) for both groups
    sns.kdeplot(data=df_ntc['Sigmoid_Score'], fill=True, color="red", label=f"{control_label} (Noise)", alpha=0.5)
    sns.kdeplot(data=df_targets['Sigmoid_Score'], fill=True, color="blue", label="Targeting Guides", alpha=0.5)
    
    # Add the threshold line
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'95th Percentile Threshold ({threshold:.4f})')
    
    # Formatting
    plt.title('Density Distribution of CRISPRi Sigmoid Scores', fontsize=16, fontweight='bold')
    plt.xlabel('Sigmoid Score (0.0 = Enriched/Failure, 1.0 = Pure Dropout)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(-0.05, 1.05)
    plt.legend(loc='upper right', frameon=True)
    
    # Save the image
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Success! High-resolution density plot saved to {output_image}")

if __name__ == "__main__":
    MATRIX_CSV = "Scored_sgRNA_Matrix.csv"
    CONTROL_LABEL = "NO-TARGET"
    OUTPUT_FILE = "Sigmoid_Density_Distribution.png"
    
    plot_sigmoid_density(MATRIX_CSV, CONTROL_LABEL, OUTPUT_FILE)
