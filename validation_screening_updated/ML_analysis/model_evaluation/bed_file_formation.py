import pandas as pd
import numpy as np
import os

def get_model_color(prob):
    """Assigns RGB color based on exact probability thresholds."""
    if pd.isna(prob):
        return "128,128,128"   # Gray fallback for missing values
    if prob >= 0.75:
        return "255,0,0"       # Red
    elif prob > 0.50:
        return "255,165,0"     # Orange
    elif prob > 0.25:
        return "144,238,144"   # Light Green
    else:
        return "128,128,0"     # Olive Green

def main():
    input_file = "Xiaofeng_validation_list_filtered.csv"
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"CRITICAL ERROR: {input_file} not found.")

    print(f"[*] Loading unified matrix: {input_file}")
    df = pd.read_csv(input_file)

    # ==================================================================
    # 1. TSS BED FILE GENERATION
    # ==================================================================
    print("[*] Generating TSS.bed...")
    tss_config = {
        "Primary TSS, 5'": ("0,0,139", "Primary_5"),
        "Primary TSS, 3'": ("173,216,230", "Primary_3"),
        "Secondary TSS, 5'": ("238,130,238", "Secondary_5"),
        "Secondary TSS, 3'": ("128,0,128", "Secondary_3")
    }

    tss_records = []
    
    for idx, row in df.iterrows():
        chrom = str(row['Chr'])
        if not chrom.startswith('chr'):
            chrom = f"chr{chrom}"
            
        # Safely extract Gene name
        gene = str(row['Gene']) if pd.notna(row.get('Gene')) else "UnknownGene"
        strand = str(row['Strand']) if pd.notna(row['Strand']) else "+"
        
        for col_name, (rgb, tag) in tss_config.items():
            if col_name in df.columns and pd.notna(row[col_name]):
                try:
                    tss_pos = int(float(row[col_name]))
                    start = tss_pos
                    end = tss_pos + 1
                    
                    # Prefix with Gene name for clear visibility
                    name = f"{gene}_{tag}"
                    
                    # BED9 Format
                    tss_records.append([
                        chrom, start, end, name, "0", strand, start, end, rgb
                    ])
                except ValueError:
                    continue 

    if tss_records:
        tss_df = pd.DataFrame(tss_records)
        tss_df.sort_values(by=[0, 1], inplace=True)
        tss_df.to_csv("TSS.bed", sep='\t', index=False, header=False)
        print(f" -> Exported TSS.bed ({len(tss_records)} features)")
    else:
        print(" -> Warning: No valid TSS coordinates found.")

    # ==================================================================
    # 2. DYNAMIC MODEL BED FILE GENERATION
    # ==================================================================
    model_columns = [col for col in df.columns if col.endswith('_probability')]
    
    if not model_columns:
        print("[!] No probability columns found. Skipping model BED generation.")
        return

    print(f"[*] Detected {len(model_columns)} models. Generating specific BED files...")

    for prob_col in model_columns:
        model_name = prob_col.replace('_probability', '')
        output_name = f"{model_name}_model.bed"
        model_records = []
        
        for idx, row in df.iterrows():
            if pd.isna(row['Start']) or pd.isna(row['End']):
                continue
                
            chrom = str(row['Chr'])
            if not chrom.startswith('chr'):
                chrom = f"chr{chrom}"
                
            try:
                start = int(float(row['Start']))
                end = int(float(row['End']))
            except ValueError:
                continue
                
            strand = str(row['Strand']) if pd.notna(row['Strand']) else "+"
            sgrna_id = str(row['ID'])
            prob = row[prob_col]
            
            # Safely extract Gene and Sequence
            gene = str(row['Gene']) if pd.notna(row.get('Gene')) else "UnknownGene"
            seq = str(row['protospacer sequence']) if pd.notna(row.get('protospacer sequence')) else "NoSeq"
            
            rgb = get_model_color(prob)
            score = int(prob * 1000) if pd.notna(prob) else 0
            
            # Embed Gene at the front of the popup string
            rich_name = f"{gene}|{sgrna_id}|{seq}|Prob:{prob:.4f}"
            
            # BED9 Format
            model_records.append([
                chrom, start, end, rich_name, score, strand, start, end, rgb
            ])
            
        if model_records:
            model_df = pd.DataFrame(model_records)
            model_df.sort_values(by=[0, 1], inplace=True)
            model_df.to_csv(output_name, sep='\t', index=False, header=False)
            print(f" -> Exported {output_name} ({len(model_records)} guides)")
            
    print("\n======================================================================")
    print("    BED FILE GENERATION COMPLETE")
    print("======================================================================")

if __name__ == "__main__":
    main()
