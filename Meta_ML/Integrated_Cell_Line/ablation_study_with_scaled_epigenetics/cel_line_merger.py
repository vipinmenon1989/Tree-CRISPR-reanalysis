import os
import argparse
import pandas as pd

def standardize_columns(df, cell_prefix):
    """
    Strips cell-line specific prefixes from column names to allow for intersection.
    """
    new_cols = {}
    for col in df.columns:
        if col.startswith(cell_prefix):
            new_cols[col] = col.replace(cell_prefix, "", 1)
        elif col.startswith(f"guide_{cell_prefix}"):
            new_cols[col] = col.replace(f"guide_{cell_prefix}", "guide_", 1)
    
    return df.rename(columns=new_cols)

def resolve_biological_synonyms(df):
    """
    Maps known biological synonyms to a single standard name so they survive the intersection.
    """
    new_cols = {}
    for col in df.columns:
        # Standardize A549's 'coverage' nomenclature to match K562
        if 'DNA_methylation_coverage_bin' in col:
            new_cols[col] = col.replace('DNA_methylation_coverage_bin', 'DNA_methylation_bin')
        else:
            new_cols[col] = col
            
    return df.rename(columns=new_cols)

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Merge K562 and A549 CRISPR datasets for ML.")
    parser.add_argument("--k562", required=True, help="Path to the K562 features CSV")
    parser.add_argument("--a549", required=True, help="Path to the A549 features CSV")
    parser.add_argument("--output", required=True, help="Path to save the merged CSV")
    args = parser.parse_args()

    print(f"Loading K562 dataset from: {args.k562}")
    df_k562 = pd.read_csv(args.k562)
    
    print(f"Loading A549 dataset from: {args.a549}")
    df_a549 = pd.read_csv(args.a549)

    # 1. Generate the Target 'class' Variable
    df_k562['class'] = (df_k562['Sigmoid_Score'] > 0.25).astype(int)
    df_a549['class'] = (df_a549['Sigmoid_Score'] > 0.25).astype(int)

    # 2. Add the Context Feature (Cell_Line: 1 for K562, 0 for A549)
    df_k562['Cell_Line'] = 1
    df_a549['Cell_Line'] = 0

    # 3. Standardize Column Names (Strip prefixes)
    df_k562 = standardize_columns(df_k562, "K562_")
    df_a549 = standardize_columns(df_a549, "A549_")

    # 4. Resolve Synonyms (Fix the methylation naming mismatch)
    df_k562 = resolve_biological_synonyms(df_k562)
    df_a549 = resolve_biological_synonyms(df_a549)

    # 5. Extract Common Columns & Print Diagnostics
    k562_cols = set(df_k562.columns)
    a549_cols = set(df_a549.columns)
    
    common_cols = list(k562_cols & a549_cols)
    
    k562_only = k562_cols - a549_cols
    a549_only = a549_cols - k562_cols

    print("\n================ COLUMN DIAGNOSTICS ================")
    print(f"Total K562 columns (post-standardization): {len(k562_cols)}")
    print(f"Total A549 columns (post-standardization): {len(a549_cols)}")
    print(f"Number of shared features identified:      {len(common_cols)}")
    print("----------------------------------------------------")
    
    if k562_only:
        print(f"[DROPPED] Columns found ONLY in K562 ({len(k562_only)}):")
        for col in sorted(k562_only):
            print(f"   -> {col}")
    else:
        print("[OK] No unique columns found in K562.")

    if a549_only:
        print(f"\n[DROPPED] Columns found ONLY in A549 ({len(a549_only)}):")
        for col in sorted(a549_only):
            print(f"   -> {col}")
    else:
        print("[OK] No unique columns found in A549.")
    print("====================================================\n")

    # 6. Filter to common columns
    df_k562_clean = df_k562[common_cols]
    df_a549_clean = df_a549[common_cols]

    # 7. Merge the Datasets
    print("Merging datasets...")
    df_merged = pd.concat([df_k562_clean, df_a549_clean], ignore_index=True)
    
    # Optional: Reorder columns for readability
    front_cols = ['ID', 'Sigmoid_Score', 'class']
    remaining_cols = [c for c in df_merged.columns if c not in front_cols]
    df_merged = df_merged[front_cols + remaining_cols]

    # 8. Export
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df_merged.to_csv(args.output, index=False)
    
    print(f"Integration complete. Merged dataset contains {len(df_merged)} rows and {len(df_merged.columns)} columns.")
    print(f"Saved to: {args.output}")

if __name__ == "__main__":
    main()
