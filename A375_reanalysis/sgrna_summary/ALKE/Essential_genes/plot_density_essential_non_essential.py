import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    input_file = "Output_File4_Density_Three_Groups.csv"
    output_image = "Essential_vs_NonEssential_Density_Plot.png"
    
    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: Input matrix '{input_file}' not found in the current directory.")
        sys.exit(1)
        
    df = pd.read_csv(input_file)
    
    score_col = 'Sigmoid_Score' if 'Sigmoid_Score' in df.columns else 'Sigmoid_score'
    group_col = 'Visualization_Group'
    
    if score_col not in df.columns or group_col not in df.columns:
        print(f"CRITICAL ERROR: Missing required columns '{score_col}' or '{group_col}' in source CSV.")
        sys.exit(1)
        
    # Isolate strictly rows containing 'Essential' keywords, purging NTC completely
    df_filtered = df[df[group_col].str.contains('Essential', case=False, na=False)].copy()
    
    if df_filtered.empty:
        # Fallback block to capture alternative raw string tags
        df_filtered = df[df[group_col].isin(['Essential_Hit', 'Non_Essential_Hit'])].copy()
        
    if df_filtered.empty:
        print("CRITICAL ERROR: No essential or non-essential hit groups found in data.")
        sys.exit(1)
        
    print("[*] Extracted target cohorts for isolated plotting:")
    print(df_filtered[group_col].value_counts().to_string())
    
    # Configure plotting aesthetics
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 5.5), dpi=300)
    
    # Assign distinct colors dynamically based on cohort identity strings
    color_palette = {}
    for group_name in df_filtered[group_col].unique():
        if 'non' in group_name.lower():
            color_palette[group_name] = "#3498DB"  # Precision Blue for Non-Essentials
        else:
            color_palette[group_name] = "#E74C3C"  # Vibrant Red for Essentials

    # Render Two-Cohort Kernel Density Estimation
    sns.kdeplot(
        data=df_filtered,
        x=score_col,
        hue=group_col,
        fill=True,
        common_norm=False,  # Evaluates shapes independently based on their own population sizes
        palette=color_palette,
        alpha=0.40,
        linewidth=2.5,
        warn_singular=False
    )
    
    # Graph layout definitions
    plt.xlim(-0.05, 1.05)
    plt.title("High-Confidence Screen Hits: Essential vs. Non-Essential Genes", fontsize=13, pad=15, weight='bold')
    plt.xlabel("Calibrated Sigmoid Score", fontsize=11, labelpad=10)
    plt.ylabel("Density Probability", fontsize=11, labelpad=10)
    
    if plt.gca().get_legend() is not None:
        plt.gca().get_legend().set_title("Gene Category")
        
    plt.tight_layout()
    plt.savefig(output_image, bbox_inches='tight')
    plt.close()
    
    print(f"\n[-->] SUCCESS: Isolated density graph saved to: {output_image}")

if __name__ == "__main__":
    main()

