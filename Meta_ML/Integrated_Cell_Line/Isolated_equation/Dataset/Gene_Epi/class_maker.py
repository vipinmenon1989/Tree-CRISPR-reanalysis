import numpy as np
import pandas as pd

# 1. Load your aggregated dataset
file_path = "gene_aggregated_all_guides.txt"
df = pd.read_csv(file_path, sep="\t")

# 2. Define the vectorized classification conditions
conditions = [
    df["mean_sigmoid_score"] > 0.25,
    df["mean_sigmoid_score"] <= 0.05,
    (df["mean_sigmoid_score"] > 0.05) & (df["mean_sigmoid_score"] <= 0.25),
]

# 3. Define matching class values
classes = [1, 0, -1]

# 4. Assign class column dynamically using numpy
df["class"] = np.select(conditions, classes, default=-1)

# 5. Calculate and print the exact distribution of classes
class_counts = df["class"].value_counts()

print("========================================")
print("          CLASS DISTRIBUTION            ")
print("========================================")
print(f"Class  1 (> 0.25)        : {class_counts.get(1, 0)}")
print(f"Class  0 (<= 0.05)       : {class_counts.get(0, 0)}")
print(f"Class -1 (In-between)    : {class_counts.get(-1, 0)}")
print(f"Total Rows Processed     : {len(df)}")
print("========================================")

# 6. Reorder columns to place 'class' directly after 'Gene' for readability
column_order = ["Gene_Id", "Gene", "class"] + [
    col for col in df.columns if col not in ["Gene_Id", "Gene", "class"]
]
df = df[column_order]

# 7. Save to the final output file
output_file = "gene_aggregated_classified.txt"
df.to_csv(output_file, sep="\t", index=False)
print(f"\nProcessing complete. Final file generated: '{output_file}'")
