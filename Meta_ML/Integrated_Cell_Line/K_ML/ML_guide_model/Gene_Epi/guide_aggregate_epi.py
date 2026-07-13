import os
import sys
import pandas as pd

def aggregate_guide_data(input_file, output_file):
    print(f"[*] Ingesting dataset: {input_file}")
    if not os.path.exists(input_file):
        print(f"CRITICAL ERROR: Input file '{input_file}' not found.")
        sys.exit(1)

    df = pd.read_csv(input_file, sep=',')
    df.columns = [col.lower().strip() for col in df.columns]

    # ==================================================================
    # 1. IDENTIFY GUIDE-LEVEL EPIGENETIC COLUMNS
    # ==================================================================
    # Filter: Must strictly start with the 'guide_' prefix
    guide_epi_cols = [
        col for col in df.columns 
        if col.startswith('guide_')
    ]
    
    print(f"[*] Identified {len(guide_epi_cols)} guide-level epigenetic columns.")

    # ==================================================================
    # 2. MATRIX ISOLATION
    # ==================================================================
    # Verify mandatory grouping and scoring columns exist
    for required_col in ['gene', 'sigmoid_score']:
        if required_col not in df.columns:
            print(f"CRITICAL ERROR: Missing required column '{required_col}'.")
            sys.exit(1)

    # Subset the dataframe to only the target columns
    target_columns = ['gene', 'sigmoid_score'] + guide_epi_cols
    df_filtered = df[target_columns].copy()

    # Drop any rows where 'gene' is NaN to prevent grouping errors
    df_filtered = df_filtered.dropna(subset=['gene'])

    # ==================================================================
    # 3. GROUPING AND AGGREGATION
    # ==================================================================
    print(f"[*] Aggregating data by gene...")
    
    # Calculate the mean for sigmoid_score and all guide-specific epigenetic bins
    df_aggregated = df_filtered.groupby('gene', as_index=False).mean()

    # Rename the aggregated sigmoid score column for explicit clarity
    df_aggregated = df_aggregated.rename(columns={'sigmoid_score': 'mean_sigmoid_score'})

    # ==================================================================
    # 4. EXPORT
    # ==================================================================
    df_aggregated.to_csv(output_file, index=False)
    print(f"[-->] Success. Aggregated matrix for {len(df_aggregated)} unique genes exported to: {output_file}")

if __name__ == "__main__":
    # Point this to your target data file
    INPUT_MATRIX = "essential_genes_subset.csv"
    OUTPUT_MATRIX = "Guide_Level_Aggregated_Epigenetics.csv"
    
    aggregate_guide_data(
        input_file=INPUT_MATRIX,
        output_file=OUTPUT_MATRIX
    )
