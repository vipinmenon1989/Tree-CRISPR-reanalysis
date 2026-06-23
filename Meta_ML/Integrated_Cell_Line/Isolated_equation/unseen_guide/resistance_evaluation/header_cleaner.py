import os
import sys
import pandas as pd

def main():
    # 1. Cluster Workspace Configuration
    working_dir = "./"
    # INPUTS: The true reference template and the file needing structural improvement
    neg_selection_template_file = os.path.join(working_dir, "CRISPRi_ML_Train_80_unseen_guides.txt") 
    resistance_file_to_fix = os.path.join(working_dir, "CRISPR_ml_features_pure_resistance_filtered.csv")
    
    # OUTPUT: The corrected version of your resistance file
    output_corrected_file = os.path.join(working_dir, "CRISPR_ml_features_pure_resistance_perfect_align.csv")

    # Safe validation paths audit
    if not os.path.exists(neg_selection_template_file):
        print(f"CRITICAL ERROR: Correct Reference Template Missing: {neg_selection_template_file}")
        sys.exit(1)
    if not os.path.exists(resistance_file_to_fix):
        print(f"CRITICAL ERROR: Resistance File To Fix Missing: {resistance_file_to_fix}")
        sys.exit(1)

    print("[*] Ingesting correct Negative Selection template headers...")
    df_neg = pd.read_csv(neg_selection_template_file, sep=None, engine='python', nrows=2)
    neg_headers = [str(col).strip().lower() for col in df_neg.columns]

    print("[*] Ingesting scrambled Resistance dataset...")
    df_res = pd.read_csv(resistance_file_to_fix)
    df_res.columns = [str(col).strip().lower() for col in df_res.columns]

    # ==================================================================
    # 2. DYNAMIC HEURISTIC MAPPING ENGINE (CROSS-REFERENCING COLUMNS)
    # ==================================================================
    print("[*] Cross-referencing resistance headers against true template...")
    
    rename_dict = {}
    unmapped_res_cols = []

    for res_col in df_res.columns:
        # If it's a direct exact match, lock it in instantly
        if res_col in neg_headers:
            continue
            
        # Strip prefixes and look for matches
        stripped_res = res_col.replace('k562_', '').replace('guide_k562_', 'guide_')
        
        # Look for structural matching variants in the true template list
        match_found = False
        
        # Direct check after clean prefix strip
        if stripped_res in neg_headers:
            rename_dict[res_col] = stripped_res
            continue

        # Heuristic Rule: Handle variations in methylation track nomenclature
        if 'methylation' in res_col and 'bin' in res_col:
            # Check if template uses 'coverage_bin' variants vs 'cpg_percent_bin'
            bin_num = res_col.split('bin_')[-1] if 'bin_' in res_col else ''
            is_guide = 'guide' in res_col
            
            # Reconstruct and search for matching string structure inside neg_headers
            for neg_col in neg_headers:
                if f"bin_{bin_num}" in neg_col and ('guide' in neg_col) == is_guide:
                    if 'methylation' in neg_col:
                        # Ensure we match CpG percent to CpG percent, coverage to coverage
                        if ('cpg' in res_col) == ('cpg' in neg_col):
                            rename_dict[res_col] = neg_col
                            match_found = True
                            break
            if match_found:
                continue

        # If it hasn't matched any rules, store it for logging
        unmapped_res_cols.append(res_col)

    # ==================================================================
    # 3. APPLY RENAMING SCHEMA & DISCARD METADATA LEAKS
    # ==================================================================
    print(f"--> Found {len(rename_dict)} clear structural header mismatches. Renaming variables...")
    df_res = df_res.rename(columns=rename_dict)

    # Force drop explicit administrative string tracks that aren't in your template
    drop_metadata = ['cell_line_origin', 'cell_line', 'origin']
    df_res = df_res.drop(columns=[c for c in drop_metadata if c in df_res.columns], errors='ignore')

    # ==================================================================
    # 4. REORDER RESISTANCE COLUMNS TO MATCH TEMPLATE SCHEMA EXACTLY
    # ==================================================================
    print("[*] Re-indexing and ordering columns to match reference layout...")
    
    # Only keep features that exist inside your true negative selection template layout
    final_features_to_keep = [col for col in neg_headers if col in df_res.columns]
    
    # Missing tracks check flag
    dropped_mismatches = set(neg_headers) - set(final_features_to_keep)
    if dropped_mismatches:
         print(f"[!] Note: {len(dropped_mismatches)} template features were not found in resistance data and skipped.")

    # Slice the data to get identical feature positioning order
    df_final_resistance = df_res[final_features_to_keep].copy()

    # Save the output
    df_final_resistance.to_csv(output_corrected_file, index=False)
    
    print("-" * 75)
    print("RESISTANCE MATRIX IMPROVEMENT LOG COMPLETE:")
    print(f"Output Corrected Path:       {output_corrected_file}")
    print(f"Total Rows Kept:             {df_final_resistance.shape[0]}")
    print(f"Original Resistance Columns: {df_res.shape[1] + len(drop_metadata)}")
    print(f"Perfectly Aligned Columns:   {df_final_resistance.shape[1]} / {len(neg_headers)} matched to Template")
    print("-" * 75)

if __name__ == "__main__":
    main()
