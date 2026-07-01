import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def identify_discriminative_features(file_path):
    df = pd.read_csv(file_path)
    
    # Separate data based on class
    class_0 = df[df['class'] == 0]
    class_1 = df[df['class'] == 1]
    
    results = []
    # Assume features are all columns except id, gene, and class
    drop_cols = ['id', 'gene', 'mean_sigmoid_score', 'class']
    features = [c for c in df.columns if c not in drop_cols]
    
    for feat in features:
        stat, p_val = ks_2samp(class_0[feat], class_1[feat])
        results.append({'Feature': feat, 'D_Statistic': stat, 'P_Value': p_val})
    
    # Create results dataframe
    res_df = pd.DataFrame(results).sort_values(by='D_Statistic', ascending=False)
    
    print("[*] TOP 10 MOST DISCRIMINATIVE FEATURES (KS TEST):")
    print(res_df.head(10).to_string(index=False))
    
    # Save to file for manuscript documentation
    res_df.to_csv("ks_test_results.csv", index=False)
    print("\n[+] Full KS-test rankings saved to ks_test_results.csv")

if __name__ == "__main__":
    identify_discriminative_features("merged_alke_dataset.csv")
