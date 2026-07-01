import os
import sys
import pandas as pd

def extract_sgrna_pool(input_file, output_csv):
    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: Source file missing at path: {input_file}")
        sys.exit(1)
        
    print(f"[*] Ingesting prediction matrix from: {input_file}")
    
    # Ingest using delim_whitespace=True to seamlessly catch tab or space variations
    df = pd.read_csv(input_file, sep="\t")
    
    # Normalize column names to guarantee match matching
    df.columns = [col.lower().strip() for col in df.columns]
    
    target_col = 'sgrna sequence'
    if target_col not in df.columns:
        print(f"CRITICAL ERROR: Column '{target_col}' not found in file header!")
        sys.exit(1)
        
    print(f"[*] Extracting guide sequence array vectors...")
    # Isolate the column as a distinct DataFrame to preserve the header in the output CSV
    df_sgrna = df[[target_col]]
    
    # Export to clean comma-separated format without row indices
    df_sgrna.to_csv(output_csv, index=False)
    print(f"[✔] Successfully exported {len(df_sgrna)} records to → {output_csv}")

if __name__ == "__main__":
    # Update paths to match your current evaluation workspace layout if needed
    working_dir = "/local/projects-t3/lilab/vmenon/CRISPRi/Re-analysis/Meta_ML/Integrated_Cell_Line/Isolated_equation/Essential_unfied_gene/unseen_guide_no_gene_epi_geatures/evaluation/"
    
    input_txt = os.path.join(working_dir, "prediction.txt")
    output_csv = os.path.join(working_dir, "Test_SSC.csv")
    
    extract_sgrna_pool(input_txt, output_csv)
