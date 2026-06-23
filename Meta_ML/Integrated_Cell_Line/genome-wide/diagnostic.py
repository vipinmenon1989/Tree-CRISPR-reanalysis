import os
import sys
import pandas as pd

def audit_partition_class_balance(train_path, test_path):
    print("======================================================================")
    print("[*] DIAGNOSTIC AUDIT: PARTITION LABEL BALANCE CHECK")
    print("======================================================================\n")

    for name, path in [('Train (80%)', train_path), ('Test (20%)', test_path)]:
        if not os.path.exists(path):
            print(f"CRITICAL ERROR: {name} file missing at: {path}")
            sys.exit(1)

    # Ingest data slices
    df_train = pd.read_csv(train_path, sep='\t')
    df_test = pd.read_csv(test_path, sep='\t')

    # Calculate distributions
    train_counts = df_train['class'].value_counts()
    test_counts = df_test['class'].value_counts()

    train_pct = df_train['class'].value_counts(normalize=True) * 100
    test_pct = df_test['class'].value_counts(normalize=True) * 100

    print("-"*70)
    print(f"TRAINING FILE BALANCE (Total Rows: {len(df_train)})")
    print("-"*70)
    print(f"  -> Class 0 (Failures): {train_counts.get(0, 0):>5} rows ({train_pct.get(0, 0.0):.2f}%)")
    print(f"  -> Class 1 (Hits):     {train_counts.get(1, 0):>5} rows ({train_pct.get(1, 0.0):.2f}%)")
    
    print("\n" + "-"*70)
    print(f"HOLDOUT TEST FILE BALANCE (Total Rows: {len(df_test)})")
    print("-"*70)
    print(f"  -> Class 0 (Failures): {test_counts.get(0, 0):>5} rows ({test_pct.get(0, 0.0):.2f}%)")
    print(f"  -> Class 1 (Hits):     {test_counts.get(1, 0):>5} rows ({test_pct.get(1, 0.0):.2f}%)")
    print("-"*70 + "\n")

    # Analytical Safety Threshold Check
    diff = abs(train_pct.get(1, 0.0) - test_pct.get(1, 0.0))
    if diff > 10.0:
        print(f"[!] WARNING: Serious class representation drift detected! (Delta: {diff:.2f}%)")
        print("    Consider altering the random_state seed to find a more balanced split layout.\n")
    else:
        print(f"--> SUCCESS: Label distributions are well-aligned. (Delta: {diff:.2f}%)")
        print("    This test set will provide an unconfounded, publishable evaluation metric.\n")
    print("======================================================================\n")

if __name__ == "__main__":
    audit_partition_class_balance(
        train_path='CRISPRi_ML_Train_80.txt',
        test_path='CRISPRi_ML_Holdout_Test_20.txt'
    )
