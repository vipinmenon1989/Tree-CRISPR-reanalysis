import os
import sys
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def execute_unseen_guides_split(matrix_path, output_dir):
    """
    Ingests the unified master matrix, strictly recalculates the class labels 
    (sigmoid_score > 0.5), and generates Train/Test partitions for the 
    Unseen Guides cross-validation configuration.
    """
    print("======================================================================")
    print("[*] STEP 4: BATCH STRUCTURAL DATA SPLIT ENGINE (UNSEEN GUIDES ONLY)")
    print("======================================================================\n")

    if not os.path.exists(matrix_path):
        print(f"CRITICAL ERROR: Unified master matrix file missing at: {matrix_path}")
        sys.exit(1)

    # 1. Ingest Master Dataset
    print("[*] Ingesting unified master flat matrix...")
    df = pd.read_csv(matrix_path, sep=',', engine='python') 
    print(f"--> Success. Raw Data Footprint: {df.shape[0]} rows x {df.shape[1]} columns")

    # Lowercase headers for absolute foolproof string lookups
    df.columns = [col.lower() for col in df.columns]

    # ==================================================================
    # STRICT CLASS RECALCULATION (THRESHOLD > 0.25)
    # ==================================================================
    if 'sigmoid_score' not in df.columns:
        print("CRITICAL ERROR: Matrix header is missing required column: 'sigmoid_score'")
        sys.exit(1)

    # Destroy the old class column to prevent data leakage
    if 'class' in df.columns:
        df = df.drop(columns=['class'])

    # Generate new strict binary labels
    print("[*] Applying strict > 0.25 threshold to sigmoid_score...")
    df['class'] = (df['sigmoid_score'] > 0.25).astype(int)

    # Verify that administrative and validation tracking columns are present
    for required_column in ['gene', 'sgrna sequence', 'class']:
        if required_column not in df.columns:
            print(f"CRITICAL ERROR: Matrix header is missing tracking column: '{required_column}'")
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    ext = 'csv' if matrix_path.endswith('.csv') else 'txt'
    output_sep = ',' if matrix_path.endswith('.csv') else '\t'

    print("\n" + "="*75)
    print(f"PROCESSING PARTITION LAYER: UNSEEN_GUIDES")
    print("="*75)
    
    train_out_path = os.path.join(output_dir, f"CRISPRi_ML_Train_80_unseen_guides.{ext}")
    test_out_path = os.path.join(output_dir, f"CRISPRi_ML_Holdout_Test_20_unseen_guides.{ext}")

    # 2. Split Execution: Unseen Guides
    print(f"[*] Sorting and isolating arrays based on unique sgRNA Sequence strings...")
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['sgrna sequence']))
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()

    # 3. Cross-Data Structural Leakage Audit
    train_genes = set(df_train['gene'].unique())
    test_genes = set(df_test['gene'].unique())
    overlapping_genes = train_genes.intersection(test_genes)

    train_guides = set(df_train['sgrna sequence'].unique())
    test_guides = set(df_test['sgrna sequence'].unique())
    overlapping_guides = train_guides.intersection(test_guides)

    print("\n" + "-"*70)
    print(f"CROSS-DATASET SET-THEORETIC INTEGRITY AUDIT")
    print("-"*70)
    print(f" Total Rows  | Train: {df_train.shape[0]:<6} | Holdout Test: {df_test.shape[0]:<6}")
    print(f" Genes Split | Train: {len(train_genes):<6} | Holdout Test: {len(test_genes):<6} | Overlap: {len(overlapping_genes)}")
    print(f" Guides Split| Train: {len(train_guides):<6} | Holdout Test: {len(test_guides):<6} | Overlap: {len(overlapping_guides)}")
    print("-"*70)

    # Hard safety rails to enforce validation sterility
    if len(overlapping_guides) > 0:
        print("CRITICAL HARDWARE FAULT: Guide overlap detected in pure unseen guide mode! Halted.")
        sys.exit(1)
    else:
        print(f"--> Validation Check Passed. UNSEEN_GUIDES holdout partitions are clean.")
    print("-"*70 + "\n")

    # 4. Write Files out back to disk workspace
    print(f"[*] Exporting files to disk...")
    df_train.to_csv(train_out_path, sep=output_sep, index=False)
    df_test.to_csv(test_out_path, sep=output_sep, index=False)
    
    print(f"--> [SUCCESS] Train Matrix Exported: {os.path.basename(train_out_path)}")
    print(f"--> [SUCCESS] Test Matrix Exported:  {os.path.basename(test_out_path)}")

    print("\n======================================================================")
    print("[-->] UNSEEN GUIDES DATASET GENERATED AND WRITTEN CLEANLY TO DISK")
    print("======================================================================\n")

if __name__ == "__main__":
    # Cluster pathing targeted directly to your matrix workspace layout
    target_working_dir = "./"
    input_matrix = os.path.join(target_working_dir, "essential_genes_subset.csv")
    
    execute_unseen_guides_split(
        matrix_path=input_matrix,
        output_dir=target_working_dir
    )
