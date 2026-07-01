import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

def retrain_physics_informed_weights(train_raw_path, output_weights_path, alpha_penalty=1.0, training_floor=0.25):
    print("======================================================================")
    print(f"[*] RE-ENGINEERED FIX: TRUE BASELINE -> ABSOLUTE FILTER -> RIDGE")
    print("======================================================================\n")

    if not os.path.exists(train_raw_path):
        print(f"CRITICAL ERROR: Training file missing at {train_raw_path}")
        sys.exit(1)

    # 1. Ingest Main Training Data Pool (All 81 columns)
    df_train = pd.read_csv(train_raw_path, sep='\t')
    df_train.columns = [c.lower() for c in df_train.columns]
    
    # 2. STEP A: CALCULATE TRUE UNBIASED GENE BASELINE (Using all rows before filtering)
    print(f"[*] Total Raw Rows: {len(df_train)}")
    df_train['true_locus_baseline'] = df_train.groupby('gene')['mean_sigmoid_score'].transform('mean')
    
    # Calculate target Selectivity Lift based on true global gene means
    df_train['selectivity_lift'] = df_train['mean_sigmoid_score'] - df_train['true_locus_baseline']

    # 3. STEP B: APPLY ABSOLUTE GATEKEEPER FLOOR
    # Filter rows using your exact class threshold alignment
    df_functional = df_train[df_train['mean_sigmoid_score'] >= training_floor].copy()
    print(f"[*] Functional Training Rows Surviving Floor (>= {training_floor}): {len(df_functional)}")
    
    if df_functional.empty:
        print("CRITICAL ERROR: No rows survived the functional floor.")
        return

    # Identify the 70 explicit epigenetic spatial bin columns
    bin_cols = [c for c in df_functional.columns if 'bin_' in c or 'atac_' in c or 'h3k' in c]
    if len(bin_cols) != 70:
        bin_cols = [c for c in df_functional.columns if c not in ['gene', 'mean_sigmoid_score', 'unnamed: 0', 'class', 'true_locus_baseline', 'selectivity_lift'] and not c.startswith('editor_')]
        bin_cols = bin_cols[:70]
    
    print(f"[*] Features Locked: {len(bin_cols)} epigenetic spatial channels.")

    # Isolate independent editor lanes
    editor_cols = [c for c in df_functional.columns if 'editor_' in c]
    editors = [c.replace('editor_', '').upper() for c in editor_cols]

    weight_dictionary = {}

    # 4. STEP C: SOLVE COEFFICIENTS PER VARIANT LANE
    for ed_col, ed_name in zip(editor_cols, editors):
        # Isolate functional records where this specific architecture was applied
        df_ed = df_functional[df_functional[ed_col] == 1]
        
        if len(df_ed) < 5:
            print(f"--> Skipping {ed_name}: Insufficient functional training tokens.")
            continue

        X_train = df_ed[bin_cols]
        y_train = df_ed['selectivity_lift'] # True Lift targeted to functional space

        # Fit Ridge model with forced tracking
        model = Ridge(alpha=alpha_penalty, fit_intercept=True, random_state=42)
        model.fit(X_train, y_train)

        weight_dictionary[f'weight_editor_{ed_name.lower()}'] = model.coef_

    # 5. Export clean constant matrix
    df_weights_out = pd.DataFrame(weight_dictionary)
    df_weights_out.index = bin_cols
    df_weights_out.index.name = 'spatial_bin'
    df_weights_out = df_weights_out.reset_index()

    df_weights_out.to_csv(output_weights_path, index=False)
    print(f"\n[-->] SUCCESS: Corrected, physics-informed weights saved to: {output_weights_path}")

if __name__ == "__main__":
    working_dir = "./"
    main_training_file = os.path.join(working_dir, "CRISPRi_ML_Train_80_random.txt")
    final_weights_output = os.path.join(working_dir, "highres_calculated_bin_weights.csv")
    
    retrain_physics_informed_weights(
        train_raw_path=main_training_file, 
        output_weights_path=final_weights_output, 
        alpha_penalty=1.0, 
        training_floor=0.25
    )
