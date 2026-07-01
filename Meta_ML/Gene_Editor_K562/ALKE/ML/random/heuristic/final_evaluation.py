import os
import sys
import numpy as np
import pandas as pd

def execute_final_evaluation(weights_path, holdout_path, n_shuffles=10000):
    print("======================================================================")
    print("[*] SCORING & VALIDATION: PROJECTING CLEAN WEIGHTS ONTO HOLDOUT")
    print("======================================================================\n")

    if not os.path.exists(weights_path) or not os.path.exists(holdout_path):
        print("CRITICAL ERROR: Weights file or Holdout file missing.")
        sys.exit(1)

    # 1. Ingest Data
    df_weights = pd.read_csv(weights_path)
    df_holdout = pd.read_csv(holdout_path, sep='\t')
    
    df_holdout.columns = [c.lower() for c in df_holdout.columns]
    df_holdout['gene'] = df_holdout['gene'].str.upper()

    bin_cols = df_weights['spatial_bin'].tolist()
    unique_genes = df_holdout['gene'].unique()
    
    weight_cols = [c for c in df_weights.columns if c.startswith('weight_editor_')]
    editors = [c.replace('weight_editor_', '').lower() for c in weight_cols]

    print(f"[*] Projecting physics-informed weights onto {len(unique_genes)} out-of-sample loci...")

    # 2. Score the Holdout Pool (In-Memory)
    pred_records = []
    for gene in unique_genes:
        df_gene_prof = df_holdout[df_holdout['gene'] == gene]
        if df_gene_prof.empty:
            continue
            
        record = {'gene_name': gene}
        spatial_features = df_gene_prof[bin_cols].iloc[0].values
        
        for ed_name, w_col in zip(editors, weight_cols):
            spatial_weights = df_weights[w_col].values
            record[f'pred_{ed_name}'] = np.dot(spatial_features, spatial_weights)
            
        pred_records.append(record)

    df_preds = pd.DataFrame(pred_records)

    # Calculate Predicted Selectivity Lift
    pred_score_cols = [f'pred_{ed}' for ed in editors]
    df_preds['pred_baseline'] = df_preds[pred_score_cols].mean(axis=1)

    for ed in editors:
        df_preds[f'lift_{ed}'] = df_preds[f'pred_{ed}'] - df_preds['pred_baseline']

    # 3. Stratified Permutation Validation
    print("\n" + "="*85)
    print(f"{'EDITOR':<12} | {'TOP-10 MEAN':<12} | {'POOL MEAN':<12} | {'FOLD CHANGE':<12} | {'EMPIRICAL P-VALUE':<15}")
    print("="*85)

    global_true_top_means = []
    global_simulated_means = []

    for ed in sorted(editors):
        true_ed_col = f'editor_{ed}'
        lift_col = f'lift_{ed}'
        
        # Grab top 10 genes based on our new clean predicted lift
        top_10_genes = df_preds.sort_values(by=lift_col, ascending=False).head(10)['gene_name'].tolist()

        # Extract true experimental scores from the unfiltered holdout pool
        df_true_pool = df_holdout[df_holdout[true_ed_col] == 1]
        if df_true_pool.empty:
            continue
            
        true_top_scores = df_true_pool[df_true_pool['gene'].isin(top_10_genes)]['mean_sigmoid_score'].tolist()
        all_pool_scores = df_true_pool['mean_sigmoid_score'].tolist()

        if not true_top_scores:
            continue

        mean_top = np.mean(true_top_scores)
        mean_pool = np.mean(all_pool_scores)
        fold_change = mean_top / mean_pool if mean_pool > 0 else 0.0
        
        global_true_top_means.append(mean_top)

        # Bootstrap Permutation Test
        np.random.seed(42)
        better_shuffles = 0
        for _ in range(n_shuffles):
            random_draw = np.random.choice(all_pool_scores, size=len(true_top_scores), replace=False)
            rand_mean = np.mean(random_draw)
            global_simulated_means.append(rand_mean)
            if rand_mean >= mean_top:
                better_shuffles += 1

        empirical_p = better_shuffles / n_shuffles
        print(f"--> {ed.upper():<10} | {mean_top:<12.4f} | {mean_pool:<12.4f} | {fold_change:<11.2f}x | {empirical_p:.4f}")

    meta_p = np.sum(np.array(global_simulated_means) >= np.mean(global_true_top_means)) / len(global_simulated_means)
    print("-" * 85)
    print(f"COMBINED METRIC VERDICT --> AGGREGATE STRATIFIED P-VALUE: {meta_p:.4e}")
    print("=" * 85)

if __name__ == "__main__":
    working_dir = "./"
    
    calculated_weights = os.path.join(working_dir, "highres_calculated_bin_weights.csv")
    raw_holdout_file = os.path.join(working_dir, "CRISPRi_ML_Holdout_Test_20_random.txt")
    
    execute_final_evaluation(weights_path=calculated_weights, holdout_path=raw_holdout_file)
