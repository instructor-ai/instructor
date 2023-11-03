from datasets import load_dataset, concatenate_datasets

# Load the 'annotated' configuration
dataset_annotated = load_dataset("griffin/chain_of_density", "annotated")

# Load the 'unannotated' configuration
dataset_unannotated = load_dataset("griffin/chain_of_density", "unannotated")

# Combine the two datasets
combined_dataset = concatenate_datasets(
    [dataset_annotated["test"], dataset_unannotated["train"]]
)

# Select specific columns
cols_to_remove = combined_dataset.column_names
cols_to_keep = ["article", "highlights"]
for col in cols_to_keep:
    cols_to_remove.remove(col)
combined_dataset = combined_dataset.remove_columns(cols_to_remove)

# Convert to pandas DataFrame
df = combined_dataset.to_pandas()

# Save to .csv file
df.to_csv("output.csv", columns=cols_to_keep, index=False)
