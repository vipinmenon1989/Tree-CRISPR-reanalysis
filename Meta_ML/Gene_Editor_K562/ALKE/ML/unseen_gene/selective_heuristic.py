import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def run_comprehensive_heuristic_pipeline(train_path, holdout_path, working_dir):
    print("======================================================================")
    print("[*] UNIFIED CRITICAL ENGINES: GENERATING ALL STRUCTURAL DATA VIEWS")
    print("======================================================================\n")

    if not os.path.exists(train_path) or not os.path.exists(holdout_path):
        print("CRITICAL ERROR: Core Training or Holdout file paths missing.")
        sys.exit(1)

    os.makedirs(working_dir, exist_ok=True)
    
    # Target file output path strings
    weights_csv = os.path.join(working_dir, "highres_calculated_bin_weights.csv")
    editor_report_txt = os.path.join(working_dir, "sterile_holdout_selectivity_report.txt")
    gene_clash_csv = os.path.join(working_dir, "gene_multi_editor_clash_matrix.csv")
    exclusivity_csv = os.path.join(working_dir, "gene_editor_exclusivity_matrix.csv")

    # 1. Ingest Datasets
    print("[*] STEP 1: Ingesting dataset files...")
    df_train = pd.read_csv(train_path, sep='\t')
    df_holdout = pd.read_csv(holdout_path, sep='\t')
    
    df_train.columns = [col.lower() for col in df_train.columns]
    df_holdout.columns = [col.lower() for col in df_holdout.columns]

    bin_features = [c for c in df_train.columns if any(m in c for m in ['_bin_', 'bin_'])]
    editor_cols = [c for c in df_train.columns if 'editor_' in c]

    # 2. Lock Scale Parameters & Standardize (Preventing Leakage)
    print("[*] STEP 2: Executing Z-score standardization across 70 spatial bins...")
    X_train_scaled = df_train[bin_features].copy()
    X_holdout_scaled = df_holdout[bin_features].copy()
    
    for col in bin_features:
        mean_val = df_train[col].mean()
        std_val = df_train[col].std()
        X_train_scaled[col] = (df_train[col] - mean_val) / std_val if std_val > 0 else 0.0
        X_holdout_scaled[col] = (df_holdout[col] - mean_val) / std_val if std_val > 0 else 0.0

    # 3. Solve Ridge Coefficients & Export File 1 (The Weights)
    print("[*] STEP 3: Training models and saving high-resolution bin weights...")
    trained_models = {}
    weight_export_dict = {'spatial_bin': bin_features}

    for editor in editor_cols:
        train_mask = df_train[editor] == 1
        X_sub = X_train_scaled[train_mask]
        y_sub = df_train[train_mask]['mean_sigmoid_score']
        
        if X_sub.shape[0] >= 15:
            model = Ridge(alpha=1.0).fit(X_sub, y_sub)
            trained_models[editor] = model
            weight_export_dict[f'weight_{editor}'] = model.coef_

    # Write File 1 out to disk workspace
    pd.DataFrame(weight_export_dict).to_csv(weights_csv, index=False)
    print(f"      --> [EXPORTED OK]: {weights_csv}")

    # 4. Process Unique Holdout Gene Scopes
    print("[*] STEP 4: Evaluating out-of-sample predictions across unique holdout loci...")
    unique_holdout_genes = df_holdout.drop_duplicates(subset=['gene']).copy()
    X_unique_holdout = X_holdout_scaled.loc[unique_holdout_genes.index]

    # Compute absolute estimated sigmoid scores
    for editor, model in trained_models.items():
        unique_holdout_genes[f'score_{editor}'] = model.predict(X_unique_holdout)

    score_cols = [f'score_{editor}' for editor in trained_models.keys()]
    unique_holdout_genes['gene_baseline_mean'] = unique_holdout_genes[score_cols].mean(axis=1)

    # Compute relative Selectivity Index values
    for editor in trained_models.keys():
        unique_holdout_genes[f'selectivity_{editor}'] = unique_holdout_genes[f'score_{editor}'] - unique_holdout_genes['gene_baseline_mean']
        # Rank descending per editor column context
        unique_holdout_genes[f'rank_in_{editor}'] = unique_holdout_genes[f'selectivity_{editor}'].rank(ascending=False, method='min').astype(int)

    # 5. Compile and Export File 2 (Editor-Centric View)
    print("[*] STEP 5: Generating Editor-Centric Selectivity Rankings Report...")
    with open(editor_report_txt, 'w') as f:
        f.write("=== TRUE UNSEEN HOLDOUT HIGH-RESOLUTION SPATIAL PAIRING REPORT ===\n")
        f.write("=" * 95 + "\n\n")
        for editor in sorted(trained_models.keys()):
            f.write(f"PURE HOLDOUT OPTIMAL TARGET GENES FOR: {editor.upper()}\n")
            f.write("-" * 95 + "\n")
            
            # Sort full target gene space descending by specific editor selectivity lift
            sub_ranked = unique_holdout_genes.sort_values(by=f'selectivity_{editor}', ascending=False).head(10)
            for idx, (_, row) in enumerate(sub_ranked.iterrows(), 1):
                f.write(f"  Rank #{idx:<2} | Gene: {row['gene'].upper():<10} | Selectivity Lift: {row[f'selectivity_{editor}']:+.4f} | Holdout Est. Sigmoid: {row[f'score_{editor}']:.4f}\n")
            f.write("\n" + "=" * 95 + "\n\n")
    print(f"      --> [EXPORTED OK]: {editor_report_txt}")

    # 6. Compile and Export File 3 (Gene-Centric View)
    print("[*] STEP 6: Generating Gene-Centric Multi-Editor Clash Matrix...")
    clash_rows = []
    exclusivity_rows = []
    
    for idx, row in unique_holdout_genes.iterrows():
        scores_series = pd.Series({ed: row[f'score_{ed}'] for ed in trained_models.keys()})
        selectivity_series = pd.Series({ed: row[f'selectivity_{ed}'] for ed in trained_models.keys()})
        
        abs_winner = scores_series.idxmax().replace('score_editor_', '').upper()
        sel_winner = selectivity_series.idxmax().replace('selectivity_editor_', '').upper()

        # Build Standard Clash Row Information
        record = {
            'Gene_Name': row['gene'].upper(),
            'Absolute_Best_Editor': abs_winner,
            'Selectivity_Best_Editor': sel_winner,
            'Locus_Baseline_Mean': row['gene_baseline_mean']
        }
        
        # Build Exclusivity Calculations
        sorted_scores = scores_series.sort_values(ascending=False)
        best_ed = sorted_scores.index[0].replace('score_editor_', '').upper()
        max_score = sorted_scores.iloc[0]
        second_ed = sorted_scores.index[1].replace('score_editor_', '').upper()
        second_score = sorted_scores.iloc[1]
        exclusivity_delta = max_score - second_score
        
        excl_record = {
            'Gene_Name': row['gene'].upper(),
            'Dominant_Editor': best_ed,
            'Max_Predicted_Score': max_score,
            'Runner_Up_Editor': second_ed,
            'Runner_Up_Score': second_score,
            'Exclusivity_Delta': exclusivity_delta
        }
        
        for editor in sorted(trained_models.keys()):
            clean_name = editor.replace('editor_', '').upper()
            
            # Map parameters across both files explicitly
            record[f'Rank_under_{clean_name}'] = row[f'rank_in_{editor}']
            record[f'EstSigmoid_under_{clean_name}'] = row[f'score_{editor}']
            excl_record[f'Score_under_{clean_name}'] = row[f'score_{editor}']

        clash_rows.append(record)
        exclusivity_rows.append(excl_record)

    # Write Out Standard Clash Matrix View
    pd.DataFrame(clash_rows).to_csv(gene_clash_csv, index=False)
    print(f"      --> [EXPORTED OK]: {gene_clash_csv}")
    
    # Write Out High-Resolution Exclusivity Sheet View (Sorted by Delta Domination)
    df_excl_out = pd.DataFrame(exclusivity_rows).sort_values(by='Exclusivity_Delta', ascending=False)
    df_excl_out.to_csv(exclusivity_csv, index=False)
    print(f"      --> [EXPORTED OK]: {exclusivity_csv}")
    
    print("\n[-->] Pipeline run complete. All 4 structural data views saved cleanly to workspace.")

if __name__ == "__main__":
    working_dir_path = "./"
    train_file_path = os.path.join(working_dir_path, "CRISPRi_ML_Train_80_unseen_genes.txt")
    holdout_file_path = os.path.join(working_dir_path, "CRISPRi_ML_Holdout_Test_20_unseen_genes.txt")
    
    run_comprehensive_heuristic_pipeline(
        train_path=train_file_path, 
        holdout_path=holdout_file_path, 
        working_dir=working_dir_path
    )
