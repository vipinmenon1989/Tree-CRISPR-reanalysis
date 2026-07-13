import pandas as pd
import numpy as np

def check_distribution():
    file_path = "CRISPRi_ML_Train_80_unseen_guides.csv"
    print(f"[*] Loading dataset: {file_path}")
    
    df = pd.read_csv(file_path, sep=',')
    df.columns = [col.lower() for col in df.columns]

    if 'sigmoid_score' in df.columns:
        # ---------------------------------------------------------
        # APPLY STRICT 0.50 THRESHOLD
        # ---------------------------------------------------------
        df['class'] = (df['sigmoid_score'] > 0.5).astype(int)
        
        print("\n========================================")
        print("  NEW CLASS DISTRIBUTION (Threshold > 0.5)")
        print("========================================")
        class_counts = df['class'].value_counts()
        total_rows = len(df)
        
        class_1_count = class_counts.get(1, 0)
        class_0_count = class_counts.get(0, 0)
        
        print(f"Total Rows: {total_rows}")
        print(f"Class 1 (Strong Hit > 0.5) : {class_1_count} ({class_1_count/total_rows*100:.2f}%)")
        print(f"Class 0 (Noise/Weak <= 0.5): {class_0_count} ({class_0_count/total_rows*100:.2f}%)")
        print("========================================\n")
        
        if class_1_count > 0:
            ratio = class_0_count / class_1_count
            print(f"[*] XGBoost scale_pos_weight parameter should be: {ratio:.4f}")
    else:
        print("CRITICAL ERROR: 'sigmoid_score' column not found in dataset.")
        return

    # Audit Feature Retention for your ML script
    explicit_metadata_drops = [
        'unique_sgrna_id', 'id', 'gene', 'sgrna sequence', 'sgrna_sequence',
        'cell_line_origin', 'sigmoid_score', 'class',
        'start', 'start_30', 'end', 'end_30', 'closest_tss_coord', 
        'gene_strand', 'strand', 'guide_strand', 'pam', 'extended_sequence'
    ]

    gene_epi_keywords = ['atac', 'methylation', 'cpg', 'h3k']
    gene_epi_drops = [
        col for col in df.columns 
        if any(kw in col for kw in gene_epi_keywords) and not col.startswith('guide_')
    ]

    all_drops = list(set(explicit_metadata_drops + gene_epi_drops))
    X = df.drop(columns=all_drops, errors='ignore').select_dtypes(include=[np.number])

    print("\n=== FEATURE SPACE PURGE METRICS ===")
    print(f"Total Broad Epigenetic Columns Flagged to Drop: {len(gene_epi_drops)}")
    if len(gene_epi_drops) == 0:
        print("WARNING: No broad features dropped. (Check if your column names match the keywords).")
    print(f"Sample Retained Local Features: {[c for c in X.columns if 'guide' in c][:5]}")
    print(f"Remaining Active Columns for Tree Ingestion: {X.shape[1]}")

if __name__ == "__main__":
    check_distribution()
