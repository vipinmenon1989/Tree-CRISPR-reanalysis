import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def calculate_optimal_heuristic_weights(matrix_path, output_report_path):
    print("======================================================================")
    print("[*] STEP-BY-STEP HEURISTIC ENGINE: APPROACH 2 (OLS OPTIMIZATION)")
    print("======================================================================\n")

    if not os.path.exists(matrix_path):
        print(f"CRITICAL ERROR: Matrix file missing at target path: {matrix_path}")
        sys.exit(1)

    # 1. Ingest Master Dataset
    print("[*] STEP 1: Ingesting dataset and auditing columns...")
    df = pd.read_csv(matrix_path, sep='\t')
    df.columns = [col.lower() for col in df.columns]
    print(f"--> Success. Footprint: {df.shape[0]} rows x {df.shape[1]} columns.")

    # Categorize the epigenetic feature groups
    act_features = [c for c in df.columns if any(m in c for m in ['h3k4me3', 'h3k27ac', 'h3k4me2', 'h3k4me1'])]
    acc_features = [c for c in df.columns if 'atac_seq' in c]
    rep_features = [c for c in df.columns if 'methylation' in c]
    editor_cols = [c for c in df.columns if 'editor_' in c]

    all_signal_features = act_features + acc_features + rep_features
    print(f"--> Isolated Modalities: Activation ({len(act_features)}), Accessibility ({len(acc_features)}), Repression ({len(rep_features)})")

    # 2. Z-Score Standardization
    print("\n[*] STEP 2: Standardizing signal scales via Z-score transformation...")
    df_norm = df.copy()
    for col in all_signal_features:
        std_dev = df_norm[col].std()
        if std_dev > 0:
            df_norm[col] = (df_norm[col] - df_norm[col].mean()) / std_dev
        else:
            df_norm[col] = 0.0

    # 3. Directional Modality Aggregation
    print("[*] STEP 3: Aggregating spatial bins into core biological forces...")
    df['x_act'] = df_norm[act_features].mean(axis=1)
    df['x_acc'] = df_norm[acc_features].mean(axis=1)
    df['x_rep'] = df_norm[rep_features].mean(axis=1)

    # 4. Weight Calculation Loop via OLS Regression
    print("[*] STEP 4: Executing OLS matrix solutions per editor population...")
    
    summary_data = []

    with open(output_report_path, 'w') as f:
        f.write("=== MATHEMATICALLY CALCULATED OPTIMAL HEURISTIC WEIGHTS ===\n")
        f.write(f"Total Master Rows: {df.shape[0]} | Total Available Features: {df.shape[1]}\n")
        f.write("-" * 85 + "\n\n")

        for editor in editor_cols:
            # Isolate rows specific to this editor
            editor_mask = df[editor] == 1
            df_sub = df[editor_mask]
            
            # Require a minimal row presence to maintain statistical viability
            if df_sub.shape[0] < 15:
                continue

            X = df_sub[['x_act', 'x_acc', 'x_rep']]
            y = df_sub['mean_sigmoid_score']

            # Compute mathematically optimal coefficients
            reg_engine = LinearRegression(fit_intercept=True).fit(X, y)
            
            alpha = reg_engine.coef_[0]   # Optimal weight for Activation
            beta = reg_engine.coef_[1]    # Optimal weight for Accessibility
            gamma = reg_engine.coef_[2]   # Optimal weight for Repression

            # Check model fit performance (Pearson r)
            predicted_sigmoid = reg_engine.predict(X)
            r_corr = np.corrcoef(y, predicted_sigmoid)[0, 1]

            summary_data.append({
                'Editor': editor.upper(),
                'Alpha': alpha,
                'Beta': beta,
                'Gamma': gamma,
                'R_Corr': r_corr,
                'N_Rows': df_sub.shape[0]
            })

            f.write(f"EDITOR PROFILE: {editor.upper()}\n")
            f.write(f"  Target Sample Count (n) : {df_sub.shape[0]} rows\n")
            f.write(f"  Calculated Formula      : Score = ({alpha:+.4f} * Act) + ({beta:+.4f} * Acc) + ({gamma:+.4f} * Rep)\n")
            f.write(f"  Resulting Pearson r      : {r_corr:.4f}\n")
            f.write("-" * 85 + "\n")

    # Display clear summary terminal log
    print("\n" + "="*85)
    print(f"{'EDITOR':<15} | {'ALPHA (HISTONE)':<16} | {'BETA (ATAC)':<14} | {'GAMMA (METH)':<14} | {'PEARSON R':<10}")
    print("="*85)
    for s in summary_data:
        print(f"{s['Editor']:<15} | {s['Alpha']:<16.4f} | {s['Beta']:<14.4f} | {s['Gamma']:<14.4f} | {s['R_Corr']:.4f}")
    print("="*85 + "\n")
    print(f"[-->] Step-by-step heuristic optimization complete. Full configuration log exported to: {output_report_path}")

if __name__ == "__main__":
    # Target configurations matching your workspace paths
    working_directory = "./"
    input_file = os.path.join(working_directory, "CRISPRi_ML_Train_80_random.txt")
    output_log = os.path.join(working_directory, "calculated_heuristic_weights_report.txt")
    
    calculate_optimal_heuristic_weights(matrix_path=input_file, output_report_path=output_log)
