import pandas as pd

df1 = pd.read_csv("Unified_target_genes_metadata.txt", sep="\t")
df2 = pd.read_csv("Unified_target_genes_data.txt", sep="\t")

# Dynamically find all guide columns
guide_columns = [col for col in df2.columns if col.startswith("guide_")]

# Create the temporary DataFrame with ALL guide columns
temp_df = pd.merge(
    df1[["unique_sgrna_id", "gene"]],
    df2[["unique_sgrna_id"] + ["sigmoid_score"] + guide_columns],
    on="unique_sgrna_id",
)

# Save intermediate file
temp_df.to_csv("temporary_guide_features.txt", sep="\t", index=False)


# To calculate the mean for EVERY guide column dynamically by Gene:
agg_dict = {col: "mean" for col in guide_columns}
agg_dict["sigmoid_score"] = "mean"  # Include sigmoid score if needed

final_df = temp_df.groupby("gene").agg(agg_dict).reset_index()

# Rename columns to reflect they are means
final_df.columns = [
    f"mean_{col}" if col != "gene" else "Gene" for col in final_df.columns
]
final_df.insert(0, "Gene_Id", [f"Gene_{i+1}" for i in range(len(final_df))])

# Save final aggregated file
final_df.to_csv("gene_aggregated_all_guides.txt", sep="\t", index=False)
