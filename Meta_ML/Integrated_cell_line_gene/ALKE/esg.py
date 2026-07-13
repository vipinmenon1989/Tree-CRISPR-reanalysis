import os
import pandas as pd

# Define cluster file paths
file_1_path = "merged_alke_dataset.csv"  # Your main feature table
file_2_path = "TreeCRISPRi_ESG_list.txt"         # File 2 containing your gene list

# 1. Load File 2 (Target Essential Genes)
print("Reading target essential gene list from File 2...")
df_essential_list = pd.read_csv(file_2_path, sep="\t")

# Standardize case to prevent mismatched lookup bugs
df_essential_list["gene_upper"] = df_essential_list["gene"].astype(str).str.upper()

# 2. Load File 1 (The Complete Aggregated Feature Space)
print("Reading full aggregated feature matrix from File 1...")
df_features = pd.read_csv(file_1_path, sep=",")

# Standardize case for the matching column
df_features["gene_upper"] = df_features["gene"].astype(str).str.upper()

# 3. Extract the subset via an Inner Join
print("Subsetting datasets...")
df_merged = pd.merge(
    df_essential_list[["gene_upper"]], 
    df_features, 
    on="gene_upper", 
    how="inner"
).drop(columns=["gene_upper"])

print("==================================================")
print("             EXTRACTION LOG PROFILE               ")
print("==================================================")
print(f"Total Unique Targets in File 2: {df_essential_list['gene_upper'].nunique()}")
print(f"Successfully Matched and Retained: {len(df_merged)} rows")
print("==================================================")

# 4. Save the final curated data matrix
output_file = "essential_gene.txt"
df_merged.to_csv(output_file, sep="\t", index=False)
print(f"Standalone matrix generated: '{output_file}'")
