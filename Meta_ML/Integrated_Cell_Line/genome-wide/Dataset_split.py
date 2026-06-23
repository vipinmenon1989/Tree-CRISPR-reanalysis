import os
import sys
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def split_raw_data_by_gene(matrix_path, train_out_path, test_out_path):
    print("======================================================================")
    print("[*] STEP 4: STRUCTURAL DATA SPLIT ENGINE (STRICT GENE GROUPING)")
    print("======================================================================\n")

    if not os.path.exists(matrix_path):
        print(f"CRITICAL ERROR: Unified matrix file not found at: {matrix_path}")
        sys.exit(1)

    # 1. Load the raw unified master dataset
    print("[*] Ingesting unified master flat matrix...")
    df = pd.read_csv(matrix_path, sep=None, engine='python') 
    print(f"--> Success. Raw Footprint: {df.shape[0]} rows x {df.shape[1]} columns")

    # Lowercase headers just for safe internal lookup strings
    df.columns = [col.lower() for col in df.columns]

    if 'gene' not in df.columns:
        print("CRITICAL ERROR: Column 'gene' not detected in the matrix header!")
        sys.exit(1)

    # 2. Execute the Grouped 80/20 Shuffle Split
    print("\n[*] Sorting and partitioning all rows based on gene groups...")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    
    train_idx, test_idx = next(gss.split(df, groups=df['gene']))

    # 3. Generate the pure structural subsets
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # 4. Strict Set-Theoretic Leakage Audit
    train_genes = set(df_train['gene'].unique())
    test_genes = set(df_test['gene'].unique())
    overlapping_genes = train_genes.intersection(test_genes)

    print("\n" + "-"*70)
    print("DATA LEAKAGE VALIDATION AUDIT")
    print("-"*70)
    print(f" -> Total Unique Genes across entire dataset: {df['gene'].nunique()}")
    print(f" -> Unique Genes locked in Training file (80%): {len(train_genes)}")
    print(f" -> Unique Genes locked in Holdout Test file (20%): {len(test_genes)}")
    print(f" --> Overlapping / Leaked Genes Detected: {len(overlapping_genes)}")
    print("-"*70)

    if len(overlapping_genes) > 0:
        print("CRITICAL ERROR: Gene structural leakage detected! Splitting halted.")
        sys.exit(1)
    else:
        print("--> SUCCESS: Zero gene overlap. Holdout partition is completely sterile.")
    print("-"*70 + "\n")

    # 5. Export the complete, unmutated raw files back to the workspace
    output_sep = ',' if train_out_path.endswith('.csv') else '\t'
    
    print("[*] Writing raw partitions to disk...")
    df_train.to_csv(train_out_path, sep=output_sep, index=False)
    df_test.to_csv(test_out_path, sep=output_sep, index=False)
    
    print(f"--> SUCCESS: Raw 80% Train File written to: {train_out_path} ({df_train.shape[0]} rows)")
    print(f"--> SUCCESS: Raw 20% Test File written to:  {test_out_path} ({df_test.shape[0]} rows)")
    print("======================================================================\n")

if __name__ == "__main__":
    split_raw_data_by_gene(
        matrix_path='Unified_GenomeWide_3CellLine_Matrix.txt',
        train_out_path='CRISPRi_ML_Train_80.txt',
        test_out_path='CRISPRi_ML_Holdout_Test_20.txt'
    )
