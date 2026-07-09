import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import os

# 1. Load data
df = pd.read_csv("merged_alke_dataset.csv")

# 2. Setup output directories
output_dir = "publication_plots"
os.makedirs(f"{output_dir}/threshold_plots", exist_ok=True)
os.makedirs(f"{output_dir}/class_dist_plots", exist_ok=True)
os.makedirs(f"{output_dir}/both_significant", exist_ok=True) # New folder

# 3. Identify features
drop_cols = ['ID', 'gene', 'mean_sigmoid_score', 'class']
features = [c for c in df.columns if c not in drop_cols]

# 4. Global list for results
results = []

# 5. Process Features
for feat in features:
    subset = df[[feat, 'class', 'mean_sigmoid_score']].dropna().copy()
    subset['class'] = subset['class'].astype(int)
    
    # --- TEST 1: Class Distribution ---
    class_0 = subset[subset['class'] == 0][feat]
    class_1 = subset[subset['class'] == 1][feat]
    stat_c, p_val_c = ks_2samp(class_0, class_1)
    
    # --- TEST 2: Threshold Impact ---
    subset['Intensity_Category'] = subset[feat].apply(
        lambda x: 'Low (<= 0.5)' if x <= 0.5 else 'High (> 0.5)'
    )
    low_sig = subset[subset['Intensity_Category'] == 'Low (<= 0.5)']['mean_sigmoid_score']
    high_sig = subset[subset['Intensity_Category'] == 'High (> 0.5)']['mean_sigmoid_score']
    stat_t, p_val_t = ks_2samp(low_sig, high_sig)
    
    results.append({'Feature': feat, 'KS_Class_P': p_val_c, 'KS_Threshold_P': p_val_t})
    
    # Plotting Logic
    # Always plot if significant in EITHER
    if p_val_c <= 0.05 or p_val_t <= 0.05:
        # Plot 1: Threshold Impact
        plt.figure(figsize=(7, 6))
        sns.boxplot(x='Intensity_Category', y='mean_sigmoid_score', hue='Intensity_Category', 
                    data=subset, palette="viridis", legend=False)
        plt.title(f"Threshold Impact: {feat}\n(KS p={p_val_t:.2e})", weight='bold')
        for fmt in ['png', 'pdf']:
            plt.savefig(f"{output_dir}/threshold_plots/{feat}_threshold.{fmt}", dpi=300)
        plt.close()

        # Plot 2: Class Distribution
        plt.figure(figsize=(7, 6))
        sns.boxplot(x='class', y=feat, hue='class', data=subset, 
                    palette={0: "#e74c3c", 1: "#3498db"}, legend=False)
        plt.title(f"Class Distribution: {feat}\n(KS p={p_val_c:.2e})", weight='bold')
        for fmt in ['png', 'pdf']:
            plt.savefig(f"{output_dir}/class_dist_plots/{feat}_class_dist.{fmt}", dpi=300)
        plt.close()

    # --- NEW: High-Confidence "Both Significant" Folder ---
    if p_val_c <= 0.05 and p_val_t <= 0.05:
        # Create a combined figure for these "Gold Standard" Gatekeepers
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot Threshold
        sns.boxplot(ax=axes[0], x='Intensity_Category', y='mean_sigmoid_score', hue='Intensity_Category', 
                    data=subset, palette="viridis", legend=False)
        axes[0].set_title(f"Threshold Impact (p={p_val_t:.2e})")
        
        # Plot Class
        sns.boxplot(ax=axes[1], x='class', y=feat, hue='class', data=subset, 
                    palette={0: "#e74c3c", 1: "#3498db"}, legend=False)
        axes[1].set_title(f"Class Distribution (p={p_val_c:.2e})")
        
        fig.suptitle(f"Confirmed Gatekeeper: {feat}", fontsize=16, weight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        for fmt in ['png', 'pdf']:
            plt.savefig(f"{output_dir}/both_significant/{feat}_confirmed_gatekeeper.{fmt}", dpi=300)
        plt.close()

pd.DataFrame(results).to_csv(f"{output_dir}/combined_ks_test_results.csv", index=False)
print(f"[+] Audit complete. Confirmed gatekeepers saved in ./{output_dir}/both_significant/")
