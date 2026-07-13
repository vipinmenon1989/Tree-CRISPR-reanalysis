import pandas as pd

# 1. Load the classified dataset
file_path = "gene_aggregated_classified.txt"
df = pd.read_csv(file_path, sep="\t")

# 2. Extract counts prior to filtering for baseline reference
total_original = len(df)
count_minus_1 = (df["class"] == -1).sum()

# 3. Filter out class -1
df_model = df[df["class"] != -1].reset_index(drop=True)

# 4. Compute metrics on the clean training set
total_retained = len(df_model)
class_1_count = (df_model["class"] == 1).sum()
class_0_count = (df_model["class"] == 0).sum()

# 5. Output metrics directly to the console
print("==================================================")
print("          PRE-TRAINING DATASET INSPECTION         ")
print("==================================================")
print(f"Original Row Count (with -1)   : {total_original}")
print(f"Dropped Rows (Class -1)        : {count_minus_1}")
print(f"Total Retained Rows for Model  : {total_retained}")
print("--------------------------------------------------")
print(f"Class 1 (Score > 0.25) Count   : {class_1_count}")
print(f"Class 0 (Score <= 0.05) Count  : {class_0_count}")
print("==================================================")

# Check for severe class imbalance
imbalance_ratio = max(class_1_count, class_0_count) / max(
    1, min(class_1_count, class_0_count)
)
print(f"Majority-to-Minority Ratio    : {imbalance_ratio:.2f}:1")
