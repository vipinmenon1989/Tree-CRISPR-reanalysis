import pandas as pd

def process_and_save_gene_data(input_file_path, output_file_path):
    # 1. Ingest the data (assumes standard comma-separated input; change sep='\t' if input is also a text file)
    df = pd.read_csv(input_file_path,sep="\t")
    
    # 2. Filter rows where Editor_ALKE is exactly 1
    df_filtered = df[df['Editor_ALKE'] == 1].copy()
    
    # 3. Define the static core columns to retain
    base_cols = ['ID', 'Gene', 'mean_sigmoid_score', 'class']
    
    # 4. Dynamically identify epigenetic tracking columns via substring matching
    epi_cols = [col for col in df.columns if 'bin' in col]
    
    # 5. Form the definitive list of target columns
    cols_to_keep = base_cols + epi_cols
    
    # 6. Restrict the DataFrame to the target columns, intrinsically dropping all others
    final_df = df_filtered[cols_to_keep]
    
    # 7. Write the processed DataFrame to a new .txt file
    # Using tab separation (sep='\t') is standard practice for plain text tabular data
    final_df.to_csv(output_file_path, sep='\t', index=False)
    
    return final_df

# Execution sequence
if __name__ == "__main__":
    # Define your input and output file paths here
    input_file = "K562_final_subset_training_matrix.txt"
    output_file = "filtered_gene_data.txt"
    
    # Run the processor
    process_and_save_gene_data(input_file, output_file)
