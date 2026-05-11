import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os

def plot_ntc_vs_hits_strict(matrix_file, control_label, output_image):
    if not os.path.exists(matrix_file):
        print(f"CRITICAL ERROR: File not found: {matrix_file}")
        sys.exit(1)

    print(f"Loading matrix: {matrix_file}")
    df = pd.read_csv(matrix_file)
    
    # Identify NTCs vs Hits
    # We use 'Gene' (Capital G) as confirmed
    ntc_mask = df['Gene'].str.contains(control_label, case=False, na=False)
    
    df_ntc = df[ntc_mask]
    df_hits = df[~ntc_mask]
    
    if df_ntc.empty:
        print(f"ERROR: No NTCs found matching '{control_label}' in 'Gene' column.")
        sys.exit(1)
    if df_hits.empty:
        print("ERROR: No hits found in file.")
        sys.exit(1)

    print(f"Plotting {len(df_ntc)} NTCs vs {len(df_hits)} Hits...")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Density plot for NTC (Background)
    sns.kdeplot(data=df_ntc['Sigmoid_Score'], fill=True, color="red", 
                label=f"NTC ({control_label})", alpha=0.4)
    
    # Density plot for Hits (Signal)
    sns.kdeplot(data=df_hits['Sigmoid_Score'], fill=True, color="blue", 
                label="High-Confidence Hits", alpha=0.5)

    # Formatting
    plt.title('Sigmoid Score Distribution: NTC vs. Hits', fontsize=16, fontweight='bold')
    plt.xlabel('Sigmoid Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(-0.05, 1.05)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    print(f"Success! Plot saved to: {output_image}")

def main():
    parser = argparse.ArgumentParser(description="Generate density plot for NTC vs. Hits distribution.")
    parser.add_argument("-i", "--input", required=True, help="Path to the K_NTC.csv file")
    parser.add_argument("-c", "--control", default="NO-TARGET", help="Label for NTC in the 'Gene' column")
    parser.add_argument("-o", "--output", default="K_NTC_Density_Plot.png", help="Path for output PNG")

    args = parser.parse_args()
    plot_ntc_vs_hits_strict(args.input, args.control, args.output)

if __name__ == "__main__":
    main()



