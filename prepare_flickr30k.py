# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import os

from datasets import load_dataset

# Load the dataset
ds = load_dataset("nlphuji/flickr30k")

# Convert the single "test" split to a Pandas DataFrame
df = ds["test"].to_pandas()

# Filter out the splits into separate datasets
train_dataset = df[df["split"] == "train"]
val_dataset = df[df["split"] == "val"]
test_dataset = df[df["split"] == "test"]

# Define the directory to save CSV files
csv_dir = "flickr30k_csvs"
os.makedirs(csv_dir, exist_ok=True)

# Save the filtered splits as CSV files in the specified directory
train_dataset.to_csv(os.path.join(csv_dir, "flickr30k_train.csv"), index=False)
print(f"Saved train split to {os.path.join(csv_dir, 'flickr30k_train.csv')}")
val_dataset.to_csv(os.path.join(csv_dir, "flickr30k_validation.csv"), index=False)
print(f"Saved validation split to {os.path.join(csv_dir, 'flickr30k_validation.csv')}")
test_dataset.to_csv(os.path.join(csv_dir, "flickr30k_test.csv"), index=False)
print(f"Saved test split to {os.path.join(csv_dir, 'flickr30k_test.csv')}")

print(f"Datasets saved successfully in the folder: {csv_dir}")
