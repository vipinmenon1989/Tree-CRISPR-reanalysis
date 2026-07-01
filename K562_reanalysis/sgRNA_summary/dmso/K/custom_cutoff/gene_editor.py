import os
import sys
import numpy as np
import pandas as pd

def main():
    # ======================================================================
    # 1. CLUSTER WORKSPACE PATH CONFIGURATIONS
    # ======================================================================
    working_dir = "./"
    
    # Input CSV file name as specified
    input_file = os.path.join(working_dir, "K_hits_complete_isolated.csv")
    output_file = os.path.join(working_dir, "gene_editor_summary_matrix.txt")

    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: Target file missing at {input_file}")
        sys.exit(1)

    print(f"[*] Ingesting guide-level targets from: {input_file}")
    # Read the CSV file
    df_raw = pd.read_csv(input_file, sep=',')
    
    # Normalize headers to handle whitespace or casing variations cleanly
    df_raw.columns = [col.strip() for col in df_raw.columns]
    
    # Map column names to internal standard keys based on lowercase lookup
    header_map = {col.lower(): col for col in df_raw.columns}
    
    required_cols = ['gene', 'sigmoid_score']
    for req in required_cols:
        if req not in header_map:
            print(f"CRITICAL ERROR: Missing required column match for '{req}' in file headers!")
            print(f"Found headers: {list(df_raw.columns)}")
            sys.exit(1)
            
    gene_col = header_map['gene']
    score_col = header_map['sigmoid_score']

    # ======================================================================
    # 2. RUN GENE-LEVEL AGGREGATION & SIGMOID MEAN CALCULATIONS
    # ======================================================================
    print("[*] Computing localized mean sigmoid trajectories per gene locus...")
    df_grouped = df_raw.groupby(gene_col, as_index=False).agg(
        mean_sigmoid_score=(score_col, 'mean')
    )
    # Rename grouped column to standard target layout
    df_grouped = df_grouped.rename(columns={gene_col: 'Gene'})

    # ======================================================================
    # 3. ENFORCE PI'S CONDITIONAL BINARY CLASS RULES
    # ======================================================================
    print("[*] Assigning categorical target flags using strict boundary thresholds...")
    def assign_class(score):
        if score > 0.25:
            return 1
        elif score <= 0.05:
            return 0
        else:
            return -1 # Safeguard flag for intermediate values (0.05 < score <= 0.25)

    df_grouped['class'] = df_grouped['mean_sigmoid_score'].apply(assign_class)

    # ======================================================================
    # 4. SPIN UP STRUCTURAL 2D EDITOR FLAG VECTORS
    # ======================================================================
    print("[*] Expanding feature matrix dimensions to accommodate editor profiles...")
    

    # Ensure columns follow your exact requested header positioning sequence
    target_hierarchy = [
        'Gene', 'mean_sigmoid_score', 'class'
    ]
    df_final = df_grouped[target_hierarchy]

    # ======================================================================
    # 5. SERIALIZE SUMMARY ROSTER TO DISK
    # ======================================================================
    print(f"[*] Writing finalized multi-modality matrix block to: {output_file}")
    df_final.to_csv(output_file, sep='\t', index=False)
    
    print("\n" + "="*80)
    print("      SUMMARY MATRIX PROCESSING COMPLETE")
    print("="*80)
    print(df_final.head(10).to_string(index=False))
    print(f"\n[-->] Total genes processed: {len(df_final)}")
    print(f"[-->] Results output location: {output_file}")

if __name__ == "__main__":
    main()

