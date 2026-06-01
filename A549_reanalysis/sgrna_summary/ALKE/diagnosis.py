# --- 1. Calculate Sigmoid Score within the script ---
# Use your n=1.5 and b=2 parameters
def calculate_sigmoid(lfc, n=1.5, b=2.0):
    # Standardize LFC to Z-score first
    z = (lfc - lfc.mean()) / lfc.std()
    return 1 / (1 + np.exp(-(n * z + b)))

df['Sigmoid_Score'] = calculate_sigmoid(df['LFC'])

# --- 2. Now plot safely ---
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='control_mean', y='Sigmoid_Score', alpha=0.3)
plt.xscale('log')
plt.title("Sigmoid Score vs. Control Depth (Audit)")
plt.xlabel("Control Mean Read Counts (log scale)")
plt.show()
