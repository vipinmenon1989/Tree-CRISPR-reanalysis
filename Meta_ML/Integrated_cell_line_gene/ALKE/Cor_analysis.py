import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_feature_correlations(file_path):
    df = pd.read_csv(file_path)
    # Ensure numeric only for correlation
    df_numeric = df.select_dtypes(include=['number'])
    
    # Calculate correlations specifically against the target score
    correlations = df_numeric.corr()['mean_sigmoid_score'].abs().sort_values(ascending=False)
    
    # Drop the target itself (corr = 1.0)
    top_features = correlations.drop('mean_sigmoid_score').head(10)
    
    print("[*] TOP 10 BIOLOGICAL DRIVERS (Correlation with mean_sigmoid_score):")
    print(top_features)
    
    # Plotting for the presentation
    plt.figure(figsize=(10, 6))
    top_features.plot(kind='barh', color='#3498db')
    plt.title('Top 10 Epigenetic Features Correlated with Editing Score', weight='bold')
    plt.xlabel('Absolute Pearson Correlation')
    plt.tight_layout()
    plt.savefig("feature_correlation_drivers.png", dpi=300)
    print("\n[+] Correlation plot saved as feature_correlation_drivers.png")

if __name__ == "__main__":
    analyze_feature_correlations("merged_alke_dataset.csv")
