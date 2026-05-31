import pandas as pd
import numpy as np

def clean_columns(col, prefix):
    guide_prefix = f'guide_{prefix}'
    if col.startswith(guide_prefix):
        return 'guide_' + col[len(guide_prefix):]
    elif col.startswith(prefix):
        return col[len(prefix):]
    return col

def main():
    # Load the datasets
    print("Loading datasets...")
    k562_df = pd.read_csv('./dataset/ALKE_epigenetic_merged_common_only_K562.txt', sep='\t')
    a549_df = pd.read_csv('./dataset/ALKE_epigenetic_merged_common_only_A549.txt', sep='\t')
    
    # Inject Classifier Tag
    print("Injecting classifier tag (is_K562)...")
    k562_df['is_K562'] = 1
    a549_df['is_K562'] = 0
    
    # Standardize Column Schemas
    print("Standardizing column schemas (stripping prefixes)...")
    k562_df.columns = [clean_columns(col, 'K562_') for col in k562_df.columns]
    a549_df.columns = [clean_columns(col, 'A549_') for col in a549_df.columns]
    
    # Resolve Feature Mismatches
    print("Resolving feature mismatches...")
    # Drop columns in K562 containing 'DNA_methylation_bin' but NOT containing 'CpG'
    k562_drop = [col for col in k562_df.columns if 'DNA_methylation_bin' in col and 'CpG' not in col]
    print(f"Dropping {len(k562_drop)} DNA methylation columns from K562...")
    k562_df.drop(columns=k562_drop, inplace=True)
    
    # Drop columns in A549 containing 'methylation_coverage'
    a549_drop = [col for col in a549_df.columns if 'methylation_coverage' in col]
    print(f"Dropping {len(a549_drop)} methylation coverage columns from A549...")
    a549_df.drop(columns=a549_drop, inplace=True)
    
    # Merge and Verify
    print("Merging dataframes vertically...")
    merged_df = pd.concat([k562_df, a549_df], axis=0, ignore_index=True)
    
    # Check for and strictly drop mismatched columns containing any NaN
    nan_cols = merged_df.columns[merged_df.isnull().any()].tolist()
    if len(nan_cols) > 0:
        print(f"Mismatched columns found introducing NaNs: {nan_cols}")
        print("Dropping mismatched columns...")
        merged_df.dropna(axis=1, inplace=True)
    else:
        print("No mismatched columns introducing NaNs found.")
        
    # ID Reassignment
    print("Reassigning unique sequential IDs...")
    merged_df['ID'] = [f'sgrna_{i}' for i in range(1, len(merged_df) + 1)]
    
    # Save the final merged dataframe
    output_file = 'ALKE_ml_features.csv'
    print(f"Saving merged dataset to: {output_file}...")
    merged_df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    main()
