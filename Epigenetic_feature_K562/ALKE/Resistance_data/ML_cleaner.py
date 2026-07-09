import os
import sys
import argparse
import pandas as pd

def extract_minimal_ml_columns(input_file, output_file):
    print(f"[*] Ingesting imbalanced master matrix: {input_file}")
    
    # 1. Read the complete ML master pool from disk
    try:
        df = pd.read_csv(input_file, sep=None, engine='python')
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to read {input_file}: {e}")
        sys.exit(1)

    # Standardize column names instantly to eliminate case-mismatch KeyErrors
    df.columns = [str(col).strip().lower() for col in df.columns]

    # ==================================================================
    # 2. DEFENSIVE HEADERS MATCHING SYSTEM
    # ==================================================================
    # Locate the best available alternative names for the sgRNA sequence column
    sgrna_candidates = ['sgrna', 'sgrna sequence', 'sequence', 'spacer', 'guide']
    sgrna_col = next((c for c in sgrna_candidates if c in df.columns), None)
    
    # Locate the best available alternative names for the gene identifier column
    gene_candidates = ['gene', 'gene_name', 'target_gene', 'locus']
    gene_col = next((c for c in gene_candidates if c in df.columns), None)
    
    # Locate the binary classification class column
    class_candidates = ['class', 'ml_target_class', 'target']
    class_col = next((c for c in class_candidates if c in df.columns), None)

    # Hard audit step: verify all three coordinates are present
    missing_elements = []
    if not sgrna_col: missing_elements.append("sgRNA (Sequence)")
    if not gene_col: missing_elements.append("Gene (Identifier)")
    if not class_col: missing_elements.append("Class (Binary Target)")
    
    if missing_elements:
        print(f"CRITICAL ERROR: Could not map required machine learning features: {missing_elements}")
        print(f"Available file headers: {list(df.columns)}")
        sys.exit(1)

    # ==================================================================
    # 3. EXTRACTION AND CLEAN EXPORT
    # ==================================================================
    print(f"--> Successfully mapped coordinate tracks:")
    print(f"    * sgRNA Column -> '{sgrna_col}'")
    print(f"    * Gene Column  -> '{gene_col}'")
    print(f"    * Class Column -> '{class_col}'")

    # Isolate the exact 3-column matrix array
    df_minimal = df[[sgrna_col, gene_col, class_col]].copy()
    
    # Rename headers to the exact standardized canonical formats requested
    df_minimal.columns = ['sgrna', 'gene', 'class']

    # Clean out any accidental null lines or broken formatting rows
    df_minimal = df_minimal.dropna(subset=['sgrna', 'gene', 'class'])
    
    # Sort by class and gene for structured downstream cross-validation slicing
    df_minimal = df_minimal.sort_values(by=['class', 'gene']).reset_index(drop=True)

    # Save to disk
    df_minimal.to_csv(output_file, index=False)

    # Calculate metrics for console validation report
    pos_count = sum(df_minimal['class'] == 1)
    neg_count = sum(df_minimal['class'] == 0)

    print("-" * 65)
    print("MINIMAL THREE-COLUMN ML EXPORT COMPLETE")
    print(f"--> Saved pristine ML reference matrix to: {output_file}")
    print(f"    * Total Extracted Matrix Dimensions: {df_minimal.shape[0]} rows x {df_minimal.shape[1]} columns")
    print(f"    * Retained Active Positives (Class 1): {pos_count} rows")
    print(f"    * Retained Inactive Baselines (Class 0): {neg_count} rows")
    print("-" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CRISPRi Minimal ML Matrix Coordinator')
    parser.add_argument('-i', '--input', default='CRISPR_ml_complete_imbalanced_pool.csv', help='Path to complete ML master input matrix')
    parser.add_argument('-o', '--output', default='CRISPR_ml_three_columns_final.csv', help='Destination path for clean 3-column reference file')
    
    args = parser.parse_args()
    extract_minimal_ml_columns(args.input, args.output)
