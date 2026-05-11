import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
import os

def plot_resistance_distribution(input_file, control_label, output_image):
    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: File not found: {input_file}")
        sys.exit(1)

    print(f"Loading distribution data: {input_file}")
    df = pd.read_csv(input_file)
    
    # 1. Identify NTCs vs Significant Hits
    # Using the standard regex/string match for control labels
    ntc_mask = df['Gene'].str.contains(control_label, case=False, na=False)
    
    df_ntc = df[ntc_mask]
    df_hits = df[~ntc_mask]
    
    if df_ntc.empty:
        print(f"ERROR: No NTCs found matching '{control_label}' in 'Gene' column.")
        sys.exit(1)
    if df_hits.empty:
        print("ERROR: No significant hits found in file.")
        sys.exit(1)

    print(f"Plotting {len(df_ntc)} NTC sgRNAs vs {len(df_hits)} Hit sgRNAs...")
    
    # 2. Plotting Setup
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Density plot for NTC (Background - usually centered near 0)
    sns.kdeplot(data=df_ntc['Delta_Score'], fill=True, color="#808080", 
                label=f"NTC Background (n={len(df_ntc)})", alpha=0.4)
    
    # Density plot for Resistance Hits (Signal - shifted right)
    sns.kdeplot(data=df_hits['Delta_Score'], fill=True, color="red", 
                label=f"Significant Resistance Hits (n={len(df_hits)})", alpha=0.6)

    # 3. Formatting for Publication
    plt.title('Resistance Gain Distribution: NTC vs. Hits', fontsize=16, fontweight='bold')
    plt.xlabel('Delta Score (Sigmoid_Drug - Sigmoid_DMSO)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    # We use a slightly wider x-limit to see the full tail of resistance
    plt.xlim(-0.1, 1.1) 
    plt.axvline(x=0.1, color='black', linestyle='--', alpha=0.5, label='Hit Threshold (0.1)')
    
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # 4. Save Output
    plt.savefig(output_image, dpi=300)
    print(f"Success! Density plot saved to: {output_image}")

def main():
    parser = argparse.ArgumentParser(description="Generate density plot for Resistance Delta Score distribution.")
    parser.add_argument("-i", "--input", required=True, help="Path to Hits_plus_NTC_Distribution.csv")
    parser.add_argument("-c", "--control", default="NO-TARGET", help="Label for NTC in the 'Gene' column")
    parser.add_argument("-o", "--output", default="Resistance_Density_Plot.png", help="Output filename")

    args = parser.parse_args()
    plot_resistance_distribution(args.input, args.control, args.output)

if __name__ == "__main__":
    main()
