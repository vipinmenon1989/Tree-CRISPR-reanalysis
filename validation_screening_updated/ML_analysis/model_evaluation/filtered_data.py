import pandas as pd
import os
import sys

def main():
    # ==========================================
    # 1. FILE CONFIGURATION
    # ==========================================
    input_file = "Xiaofeng_validation_list.csv"
    output_file = "Xiaofeng_validation_list_filtered.csv"
    
    # Set the minimum number of guides required per gene
    MIN_GUIDES = 3

    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: File '{input_file}' not found.")
        sys.exit(1)

    print(f"[*] Loading dataset: {input_file}")
    df = pd.read_csv(input_file)

    if 'Gene' not in df.columns:
        print("CRITICAL ERROR: 'Gene' column missing from the dataset.")
        sys.exit(1)

    # ==========================================
    # 2. CALCULATE METRICS & FILTER
    # ==========================================
    initial_guide_count = len(df)
    initial_gene_count = df['Gene'].nunique()

    print(f"[*] Filtering genes with < {MIN_GUIDES} guides...")
    
    # The transform('size') calculates the count per gene and aligns it to the original rows.
    # We keep only rows where that count is >= MIN_GUIDES.
    filtered_df = df[df.groupby('Gene')['Gene'].transform('size') >= MIN_GUIDES].copy()

    final_guide_count = len(filtered_df)
    final_gene_count = filtered_df['Gene'].nunique()
    
    dropped_guides = initial_guide_count - final_guide_count
    dropped_genes = initial_gene_count - final_gene_count

    # ==========================================
    # 3. EXPORT & REPORT
    # ==========================================
    filtered_df.to_csv(output_file, index=False)

    print("\n" + "="*65)
    print("    GENE THRESHOLD FILTERING METRICS")
    print("="*65)
    print(f" -> Initial Guides (Total Rows) : {initial_guide_count}")
    print(f" -> Initial Unique Genes        : {initial_gene_count}")
    print("-" * 65)
    print(f" -> Dropped Guides              : {dropped_guides}")
    print(f" -> Dropped Genes               : {dropped_genes}")
    print("-" * 65)
    print(f" -> Final Guides (Total Rows)   : {final_guide_count}")
    print(f" -> Final Unique Genes          : {final_gene_count}")
    print("="*65)
    print(f"[-->] Filtered dataset saved to: {output_file}")

if __name__ == "__main__":
    main()
