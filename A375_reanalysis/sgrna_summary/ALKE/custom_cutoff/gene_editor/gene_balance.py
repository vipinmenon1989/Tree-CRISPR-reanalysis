import os
import sys
import pandas as pd

def main():
    # ======================================================================
    # 1. CLUSTER WORKSPACE PATH CONFIGURATIONS
    # ======================================================================
    working_dir = "./"
    
    input_file = os.path.join(working_dir, "gene_editor_summary_matrix.txt")
    output_file = os.path.join(working_dir, "gene_editor_balanced_matrix.txt")

    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: Input summary matrix missing at {input_file}")
        sys.exit(1)

    print(f"[*] Ingesting summary matrix from: {input_file}")
    df = pd.read_csv(input_file, sep='\t')
    df.columns = [col.strip() for col in df.columns]

    # Validate tracking identifiers
    if 'class' not in df.columns:
        print("CRITICAL ERROR: 'class' column missing from data matrix headers!")
        sys.exit(1)

    # ======================================================================
    # 2. DROP INTERMEDIATE DEAD-ZONE ROWS (class == -1)
    # ======================================================================
    initial_row_count = len(df)
    df_filtered = df[df['class'] != -1].copy()
    dropped_rows = initial_row_count - len(df_filtered)
    print(f"--> Dropped {dropped_rows} rows containing intermediate class (-1) labels.")

    # ======================================================================
    # 3. ENFORCE STRICT 50/50 CLASS BALANCING via DOWNSAMPLING
    # ======================================================================
    count_zeros = int((df_filtered['class'] == 0).sum())
    count_ones = int((df_filtered['class'] == 1).sum())
    
    print(f"[*] Post-filtering class layout: Class 0 = {count_zeros} | Class 1 = {count_ones}")

    if count_zeros > count_ones:
        print(f"[*] Class imbalance detected (Zeros > Ones). Commencing downsampling pipeline...")
        
        # Isolate individual class pools
        df_zeros = df_filtered[df_filtered['class'] == 0]
        df_ones = df_filtered[df_filtered['class'] == 1]
        
        # Randomly sample matching number of rows from the Zeros pool
        df_zeros_downsampled = df_zeros.sample(n=count_ones, random_state=42)
        
        # Re-merge the balanced data splits
        df_final = pd.concat([df_ones, df_zeros_downsampled], axis=0)
        # Sort by Gene name to maintain standard matrix layout
        df_final = df_final.sort_values(by='Gene').reset_index(drop=True)
        
    elif count_ones > count_zeros:
        print(f"[*] Class imbalance detected (Ones > Zeros). Downsampling Class 1 pool...")
        df_zeros = df_filtered[df_filtered['class'] == 0]
        df_ones = df_filtered[df_filtered['class'] == 1]
        
        df_ones_downsampled = df_ones.sample(n=count_zeros, random_state=42)
        df_final = pd.concat([df_zeros, df_ones_downsampled], axis=0)
        df_final = df_final.sort_values(by='Gene').reset_index(drop=True)
        
    else:
        print("[*] Classes are already perfectly balanced (1:1 ratio). Skipping downsampling step.")
        df_final = df_filtered.sort_values(by='Gene').reset_index(drop=True)

    # Print final validation metrics to cluster log
    final_zeros = int((df_final['class'] == 0).sum())
    final_ones = int((df_final['class'] == 1).sum())
    print(f"--> Final balanced matrix layout: Class 0 = {final_zeros} | Class 1 = {final_ones} (Total: {len(df_final)} rows)")

    # ======================================================================
    # 4. SERIALIZE BALANCED ROSTER TO DISK
    # ======================================================================
    print(f"[*] Saving optimized balanced matrix to target path: {output_file}")
    df_final.to_csv(output_file, sep='\t', index=False)
    print("[-->] Execution completed successfully.")

if __name__ == "__main__":
    main()

