import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def split_raw_data_by_guide_stratified(matrix_path, train_out_path, test_out_path):
    print("======================================================================")
    print("[*] STEP 4: STRUCTURAL DATA SPLIT ENGINE (STRATIFIED GUIDE GROUPS)")
    print("======================================================================\n")

    if not os.path.exists(matrix_path):
        print(f"CRITICAL ERROR: Unified matrix file not found at: {matrix_path}")
        sys.exit(1)

    # 1. Load the raw unified master dataset
    print("[*] Ingesting unified master flat matrix...")
    df = pd.read_csv(matrix_path, sep=None, engine='python') 
    print(f"--> Success. Raw Footprint: {df.shape[0]} rows x {df.shape[1]} columns")

    # Lowercase headers for clean internal lookups
    df.columns = [col.lower() for col in df.columns]

    # Required target column validation
    required_cols = ['sgrna sequence', 'class']
    for col in required_cols:
        if col not in df.columns:
            print(f"CRITICAL ERROR: Required column '{col}' not detected in matrix header!")
            sys.exit(1)

    # 2. Execute Stratified Group-Aware Split (Ensures zero sequence leakage + balanced classes)
    print("\n[*] Grouping rows by unique guides with target class stratification...")
    
    # We leverage StratifiedGroupKFold to separate groups while balancing target classes.
    # Setting n_splits=5 naturally yields an 80/20 train/test split on fold 1.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    train_idx, test_idx = next(sgkf.split(X=df, y=df['class'], groups=df['sgrna sequence']))

    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # 3. Set-Theoretic Guide Sequence Leakage Audit
    train_guides = set(df_train['sgrna sequence'].unique())
    test_guides = set(df_test['sgrna sequence'].unique())
    overlapping_guides = train_guides.intersection(test_guides)

    print("\n" + "-"*70)
    print("STRATIFIED CROSS-GUIDE EXTRACTION VALIDATION AUDIT")
    print("-"*70)
    print(f" -> Total Unique Guides across entire dataset: {df['sgrna sequence'].nunique()}")
    print(f" -> Unique Guides locked in Training file (80%): {len(train_guides)}")
    print(f" -> Unique Guides locked in Holdout Test file (20%): {len(test_guides)}")
    print(f" --> Overlapping / Leaked Guide Strings Detected: {len(overlapping_guides)}")
    print("-"*70)
    print(f" -> Train Class Distribution:\n{df_train['class'].value_counts(normalize=True).to_string()}")
    print(f" -> Test Class Distribution:\n{df_test['class'].value_counts(normalize=True).to_string()}")
    print("-"*70)

    if len(overlapping_guides) > 0:
        print("CRITICAL ERROR: Guide sequence leakage detected! Splitting halted.")
        sys.exit(1)
    else:
        print("--> SUCCESS: Zero guide sequence overlap. Holdout test file is sterile and balanced.")
    print("-"*70 + "\n")

    # 4. Export the complete, unmutated raw files back to the workspace
    output_sep = ',' if train_out_path.endswith('.csv') else '\t'
    
    print("[*] Writing raw partitions to disk...")
    df_train.to_csv(train_out_path, sep=output_sep, index=False)
    df_test.to_csv(test_out_path, sep=output_sep, index=False)
    
    print(f"--> SUCCESS: Raw 80% Train File written to: {train_out_path} ({df_train.shape[0]} rows)")
    print(f"--> SUCCESS: Raw 20% Test File written to:  {test_out_path} ({df_test.shape[0]} rows)")
    print("======================================================================\n")

if __name__ == "__main__":
    split_raw_data_by_guide_stratified(
        matrix_path='Unified_GenomeWide_3CellLine_Matrix.txt',
        train_out_path='CRISPRi_ML_Train_80.txt',
        test_out_path='CRISPRi_ML_Holdout_Test_20.txt'
    )
