import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

def execute_all_matrix_splits(matrix_path, output_dir):
    """
    Ingests the unified master matrix and automatically generates Train/Test partitions
    for two cross-validation configurations: Randomized and Unseen Genes.
    """
    print("======================================================================")
    print("[*] STEP 4: BATCH STRUCTURAL DATA SPLIT ENGINE (RANDOM & UNSEEN GENES)")
    print("======================================================================\n")

    if not os.path.exists(matrix_path):
        print(f"CRITICAL ERROR: Unified master matrix file missing at: {matrix_path}")
        sys.exit(1)

    # 1. Ingest Master Dataset
    print("[*] Ingesting unified master flat matrix...")
    df = pd.read_csv(matrix_path, sep='\t', engine='python') 
    print(f"--> Success. Raw Data Footprint: {df.shape[0]} rows x {df.shape[1]} columns")

    # Lowercase headers for absolute foolproof string lookups ('Gene' -> 'gene', 'Class' -> 'class')
    df.columns = [col.lower() for col in df.columns]

    # Verify that administrative and validation tracking columns are present
    for required_column in ['gene', 'class']:
        if required_column not in df.columns:
            print(f"CRITICAL ERROR: Matrix header is missing tracking column: '{required_column}'")
            sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    ext = 'csv' if matrix_path.endswith('.csv') else 'txt'
    output_sep = ',' if matrix_path.endswith('.csv') else '\t'

    # Define the 2 targeted split frameworks to execute sequentially
    split_modalities = ['random', 'unseen_genes']

    for mode in split_modalities:
        print("\n" + "="*75)
        print(f"PROCESSING PARTITION LAYER: {mode.upper()}")
        print("="*75)
        
        train_out_path = os.path.join(output_dir, f"CRISPRi_ML_Train_80_{mode}.{ext}")
        test_out_path = os.path.join(output_dir, f"CRISPRi_ML_Holdout_Test_20_{mode}.{ext}")

        # 2. Split Execution Switch
        if mode == 'random':
            print(f"[*] Shuffling rows into Stratified Randomized 80/20 train/test pools...")
            df_train, df_test = train_test_split(
                df, 
                train_size=0.8, 
                stratify=df['class'], 
                random_state=42
            )

        elif mode == 'unseen_genes':
            print(f"[*] Sorting and isolating arrays based on unique Entrez/Gene targets...")
            gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
            train_idx, test_idx = next(gss.split(df, groups=df['gene']))
            df_train = df.iloc[train_idx].copy()
            df_test = df.iloc[test_idx].copy()

        # 3. Cross-Data Structural Leakage Audit
        train_genes = set(df_train['gene'].unique())
        test_genes = set(df_test['gene'].unique())
        overlapping_genes = train_genes.intersection(test_genes)

        print("\n" + "-"*70)
        print(f"CROSS-DATASET SET-THEORETIC INTEGRITY AUDIT")
        print("-"*70)
        print(f" Total Rows  | Train: {df_train.shape[0]:<6} | Holdout Test: {df_test.shape[0]:<6}")
        print(f" Genes Split | Train: {len(train_genes):<6} | Holdout Test: {len(test_genes):<6} | Overlap: {len(overlapping_genes)}")
        print("-"*70)

        # Hard safety rails to enforce validation sterility
        if mode == 'unseen_genes' and len(overlapping_genes) > 0:
            print("CRITICAL HARDWARE FAULT: Gene overlap detected in pure out-of-sample mode! Halted.")
            sys.exit(1)
        else:
            print(f"--> Validation Check Passed. {mode.upper()} holdout partitions are clean.")
        print("-"*70 + "\n")

        # 4. Write Files out back to disk workspace
        print(f"[*] Exporting files to disk...")
        df_train.to_csv(train_out_path, sep=output_sep, index=False)
        df_test.to_csv(test_out_path, sep=output_sep, index=False)
        
        print(f"--> [SUCCESS] Train Matrix Exported: {os.path.basename(train_out_path)}")
        print(f"--> [SUCCESS] Test Matrix Exported:  {os.path.basename(test_out_path)}")

    print("\n======================================================================")
    print("[-->] BOTH SYSTEM FORMATS GENERATED AND WRITTEN CLEANLY TO DISK")
    print("======================================================================\n")

if __name__ == "__main__":
    # Cluster pathing targeted directly to your matrix workspace layout
    target_working_dir = "./"
    input_matrix = os.path.join(target_working_dir,"K562_final_subset_training_matrix.txt")
    
    execute_all_matrix_splits(
        matrix_path=input_matrix,
        output_dir=target_working_dir
    )
