import pandas as pd
import numpy as np
import statsmodels.api as sm

# 1. Load the data
df = pd.read_csv("merged_alke_dataset.csv")

# 2. Select only the feature columns (Indices 4 to 73)
# ID (0), Gene (1), Score (2), Class (3) are excluded
feature_names = df.columns[4:].tolist()
X_raw = df.iloc[:, 4:]  # Feature matrix
y = df['class']         # Binary outcome

# 3. Sanitize: Ensure X_raw is float and drop any row with a NaN
X_raw = X_raw.apply(pd.to_numeric, errors='coerce')
# Combine X and y to drop rows consistently
temp_df = pd.concat([X_raw, y], axis=1).dropna()
X_clean = temp_df.iloc[:, :-1]
y_clean = temp_df.iloc[:, -1]

# 4. Run Regression per feature
results = []
print(f"[*] Analyzing {len(feature_names)} features...")

for feat in feature_names:
    try:
        # Add constant for intercept
        X = sm.add_constant(X_clean[feat])
        model = sm.Logit(y_clean, X).fit(disp=0, method='bfgs')
        
        # Odds Ratio = exp(coefficient)
        odds_ratio = np.exp(model.params[1])
        p_val = model.pvalues[1]
        
        results.append({'Feature': feat, 'Odds_Ratio': odds_ratio, 'P_Value': p_val})
    except Exception as e:
        print(f"[-] Skipped {feat}: {e}")

# 5. Save results
res_df = pd.DataFrame(results).sort_values(by='Odds_Ratio', ascending=False)
res_df.to_csv("logistic_regression_results.csv", index=False)

print("[+] Analysis complete. Results saved to logistic_regression_results.csv")
print(res_df.head(10).to_string(index=False))
