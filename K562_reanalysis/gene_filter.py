import os
import shutil
import pandas as pd
import sys
import re

def process_gene_summary():
    # 1. Validate Command Line Argument
    if len(sys.argv) < 2:
        print("Error: Missing filename. Usage: python script.py <filename>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)

    # 2. Define logic for Group and Treatment detection
    # Note: Longest strings are placed first in regex to prevent partial matching (e.g., matching 'K' inside 'ALKM')
    groups = ['ALKM_E', 'ALKM_L', 'ALKM_H', 'ALK_E', 'ALK_L', 'ALK_H', 'WT', 'K']
    treatments = ['dmso', 'rig']

    group_match = re.search('|'.join(groups), input_file)
    treatment_match = re.search('|'.join(treatments), input_file)

    if not group_match or not treatment_match:
        print("Error: Filename does not match recognized Group or Treatment patterns.")
        sys.exit(1)

    group_name = group_match.group()
    treatment_name = treatment_match.group()

    # 3. Directory Management
    target_dir = os.path.join(group_name, treatment_name)
    os.makedirs(target_dir, exist_ok=True)

    # 4. Copy input file to target directory
    shutil.copy(input_file, target_dir)

    # 5. Data Processing
    try:
        df = pd.read_csv(input_file, sep='\t')

        # Logic check: Verify column existence
        if 'neg|goodsgrna' not in df.columns:
            print("Error: Column 'neg|goodsgrna' not found.")
            sys.exit(1)

        # 6. Range Filter: Keep values where 3 <= x <= 6
        # This addresses your concern about losing data between 3 and 6
        df_filtered = df[(df['neg|goodsgrna'] >= 3) & (df['neg|goodsgrna'] <= 6)].copy()

        # 7. Column Selection and Significance logic
        required_cols = ['id', 'neg|p-value', 'neg|goodsgrna', 'neg|lfc']
        df_subset = df_filtered[required_cols].copy()
        
        # Add boolean significance column
        df_subset['p-value significant'] = df_subset['neg|p-value'] < 0.05

        # 8. Save Filtered File
        output_filename = f"{group_name}_{treatment_name}_filtered.txt"
        output_path = os.path.join(target_dir, output_filename)
        
        df_subset.to_csv(output_path, sep='\t', index=False)
        
        print(f"Process complete.")
        print(f"Output saved to: {output_path}")
        print(f"Rows retained: {len(df_subset)}")

    except Exception as e:
        print(f"Data processing error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    process_gene_summary()
