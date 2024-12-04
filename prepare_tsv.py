# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import os

import pandas as pd


def prepare_tsv(csv_path, tar_name, output_tsv_path):
    """
    Convert a Flickr30k CSV file into a TSV file for fine-tuning.
    Args:
        csv_path (str): Path to the input CSV file.
        tar_name (str): Name of the .tar file containing images.
        output_tsv_path (str): Path to save the output .tsv file.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Validate required columns
    required_columns = {"caption", "filename"}
    if not required_columns.issubset(set(df.columns)):
        raise ValueError(f"CSV file must contain columns: {required_columns}")

    # Add 'tar' and 'file' columns
    df["tar"] = tar_name
    df["file"] = df["filename"]

    # Select only the required columns
    tsv_df = df[["tar", "file", "caption"]]

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_tsv_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the DataFrame to a .tsv file
    tsv_df.to_csv(output_tsv_path, sep="\t", index=False)
    print(f"Saved .tsv file to {output_tsv_path}")


if __name__ == "__main__":
    # Input CSV paths
    csv_files = {
        "train": "flickr30k_csvs/flickr30k_train.csv",
        "validation": "flickr30k_csvs/flickr30k_validation.csv",
        "test": "flickr30k_csvs/flickr30k_test.csv",
    }

    # Corresponding .tar file names
    tar_files = {
        "train": "flickr30k_tars/flickr30k_train.tar",
        "validation": "flickr30k_tars/flickr30k_validation.tar",
        "test": "flickr30k_tars/flickr30k_test.tar",
    }

    # Output TSV paths
    output_tsv_files = {
        "train": "flickr30k_tsvs/flickr30k_train.tsv",
        "validation": "flickr30k_tsvs/flickr30k_validation.tsv",
        "test": "flickr30k_tsvs/flickr30k_test.tsv",
    }

    # Process each split
    for split, csv_path in csv_files.items():
        if not os.path.exists(csv_path):
            print(f"CSV file {csv_path} not found. Skipping {split} split.")
            continue

        prepare_tsv(
            csv_path=csv_path,
            tar_name=tar_files[split],
            output_tsv_path=output_tsv_files[split],
        )
