import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import xgboost as xgb

# ==========================================
# GLOBAL SEED FOR REPRODUCIBILITY
# ==========================================
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

def main():
    # 1. Cluster Workspace Configuration
    working_dir = "./"
    
    # Update this to wherever your .json models are stored
    models_dir = os.path.join(working_dir, "models") 
    input_data_path = os.path.join(working_dir, "CRISPR_merged_final_ML_final_cleaned.csv")
    output_data_path = os.path.join(working_dir, "multimodel_predictions_effectiveness.csv")

    # Core system validation checks
    if not os.path.exists(input_data_path):
        print(f"CRITICAL ERROR: Input file missing at {input_data_path}")
        sys.exit(1)
        
    if not os.path.exists(models_dir):
        print(f"CRITICAL ERROR: Models directory missing at {models_dir}")
        sys.exit(1)

    model_files = glob.glob(os.path.join(models_dir, "*.json"))
    if not model_files:
        print(f"CRITICAL ERROR: No .json models found in {models_dir}")
        sys.exit(1)

    # 2. Ingest Data Matrix
    print("[*] Ingesting data matrix...")
    df_raw = pd.read_csv(input_data_path, sep=',')
    
    # Create a lowercase version of columns solely for model feature matching
    df_features = df_raw.copy()
    df_features.columns = [col.lower() for col in df_features.columns]

    # Dynamically locate the ID column
    id_col = 'ID' if 'ID' in df_raw.columns else 'unique_sgrna_id' if 'unique_sgrna_id' in df_raw.columns else 'id'
    
    if id_col.lower() not in df_features.columns:
        print("CRITICAL ERROR: No ID tracking column found in the dataset.")
        sys.exit(1)

    # Initialize the final output structure with just the ID column
    results_df = pd.DataFrame({'ID': df_raw[id_col]})
    
    prob_dict = {}
    eff_dict = {}
    prob_col_names = []
    eff_col_names = []

    # 3. Dynamic Multi-Model Inference Loop
    print(f"[*] Discovered {len(model_files)} models. Initiating forward pass inference...")
    
    for model_path in model_files:
        # Extract base name (e.g., 'K_model.json' -> 'K')
        base_filename = os.path.basename(model_path)
        model_prefix = base_filename.replace('_model.json', '').replace('.json', '')
        
        print(f" -> Executing {model_prefix}...")
        
        # Load Booster (Explicitly passing the seed to the XGBoost instance)
        model = xgb.XGBClassifier(random_state=SEED)
        model.load_model(model_path)
        expected_features = model.get_booster().feature_names
        
        # Validate feature existence for this specific model
        missing = [f for f in expected_features if f not in df_features.columns]
        if missing:
            print(f"    [!] Warning: {len(missing)} expected features missing for {model_prefix}. Skipping model.")
            print(f"    [!] Missing columns: {missing}")
            continue

        # Sub-select exact features required by this specific model
        X = df_features[expected_features].select_dtypes(include=[np.number])
        
        # Predict probability of class 1
        y_proba = model.predict_proba(X)[:, 1]
        
        # Generate column headers
        prob_col = f"{model_prefix}_probability"
        eff_col = f"{model_prefix}_effectiveness"
        
        # Store prediction arrays
        prob_dict[prob_col] = np.round(y_proba, 4)
        eff_dict[eff_col] = np.where(y_proba > 0.5, 'High', 'Low')
        
        prob_col_names.append(prob_col)
        eff_col_names.append(eff_col)

    # 4. Construct Final Matrix Block
    print("[*] Assembling final predictions matrix...")
    
    # Append all probabilities first
    for col in prob_col_names:
        results_df[col] = prob_dict[col]
        
    # Append all effectiveness classifications next
    for col in eff_col_names:
        results_df[col] = eff_dict[col]

    # 5. Export Output
    results_df.to_csv(output_data_path, index=False)
    
    print("\n" + "="*65)
    print("    MULTI-MODEL INFERENCE COMPLETE")
    print("="*65)
    print(f" -> Models successfully evaluated: {len(prob_col_names)}")
    print(f" -> Output structure: {results_df.shape[0]} rows x {results_df.shape[1]} columns")
    print(f" -> Output saved to: {output_data_path}")
    print("="*65)

if __name__ == "__main__":
    main()
