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
    Maps A375's specific 'Coverage' nomenclature variants to match the standard training format.
    """
    new_cols = {}
    for col in df.columns:
        if 'DNA_methylation_Coverage_bin' in col:
            new_cols[col] = col.replace('DNA_methylation_Coverage_bin', 'DNA_methylation_bin')
        elif 'DNA_methylation_coverage_bin' in col:
            new_cols[col] = col.replace('DNA_methylation_coverage_bin', 'DNA_methylation_bin')
        else:
            new_cols[col] = col
            
    return df.rename(columns=new_cols)

def main():
    parser = argparse.ArgumentParser(description="Format A375 CRISPR dataset layout to match the master training schema.")
    parser.add_argument("--input", required=True, help="Path to the input A375 tab-separated file")
    parser.add_argument("--output", required=True, help="Path to save the standardized A375 tab-separated file")
    args = parser.parse_args()

    print(f"Loading A375 dataset from: {args.input}")
    df = pd.read_csv(args.input, sep=',')

    # 1. Generate the Target 'class' Variable using your exact threshold logic
    if 'Sigmoid_Score' in df.columns:
        df['class'] = (df['Sigmoid_Score'] > 0.25).astype(int)
    elif 'sigmoid_score' in df.columns:
        df['class'] = (df['sigmoid_score'] > 0.25).astype(int)
    else:
        raise KeyError("Could not locate 'Sigmoid_Score' column for target categorization.")

    # 2. Standardize Column Names (Strip 'A375_' prefixes)
    df = standardize_columns(df, "A375_")

    # 3. Resolve Synonyms (Map Coverage terms directly to training definitions)
    df = resolve_biological_synonyms(df)

    # 4. Drop any potential duplicate column artifacts from renaming operations
    df = df.loc[:, ~df.columns.duplicated()]

    # 5. Reorder primary keys to the front exactly matching your master script layout
    front_cols = ['ID', 'Sigmoid_Score', 'class']
    remaining_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + remaining_cols]

    # 6. Export back out as a tab-separated file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, sep='\t', index=False)
    
    print("\n================ PROCESSING COMPLETE ================")
    print(f"Total processed rows:    {len(df)}")
    print(f"Total processed columns: {len(df.columns)}")
    print(f"Standardized file saved to: {args.output}")
    print("=====================================================\n")

if __name__ == "__main__":
    main()
