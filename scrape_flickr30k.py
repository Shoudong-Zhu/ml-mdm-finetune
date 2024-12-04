# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import logging
import os
import tarfile
from dataclasses import dataclass, field

import pandas as pd
import simple_parsing


@dataclass
class Flickr30kConfig:
    input_dir: str = field(
        default="flickr30k", metadata={"help": "Directory containing images"}
    )
    output_dir: str = field(
        default="flickr30k_tars", metadata={"help": "Output directory for TAR files"}
    )
    csv_dir: str = field(
        default="flickr30k_csvs", metadata={"help": "Directory containing CSV files"}
    )


def get_parser():
    parser = simple_parsing.ArgumentParser(
        description="Prepare Flickr30k TAR Files", add_config_path_arg=True
    )
    parser.add_arguments(Flickr30kConfig, dest="options")
    return parser


def create_tar_files(config: Flickr30kConfig):
    os.makedirs(config.output_dir, exist_ok=True)

    # CSV file paths
    csv_files = {
        "train": os.path.join(config.csv_dir, "flickr30k_train.csv"),
        "validation": os.path.join(config.csv_dir, "flickr30k_validation.csv"),
        "test": os.path.join(config.csv_dir, "flickr30k_test.csv"),
    }

    # Process each split
    for split_name, csv_path in csv_files.items():
        if not os.path.exists(csv_path):
            logging.warning(f"{csv_path} not found. Skipping {split_name} split.")
            continue

        logging.info(f"Processing {split_name} split...")

        # Read CSV file
        df = pd.read_csv(csv_path)
        if "filename" not in df.columns:
            logging.error(f"{csv_path} does not contain a 'filename' column. Skipping.")
            continue

        tar_path = os.path.join(config.output_dir, f"flickr30k_{split_name}.tar")

        # Create TAR file
        with tarfile.open(tar_path, "w") as tar:
            for _, row in df.iterrows():
                img_path = os.path.join(config.input_dir, row["filename"])
                if os.path.exists(img_path):
                    tar.add(img_path, arcname=row["filename"])
                else:
                    logging.warning(f"Image {img_path} not found. Skipping.")

        logging.info(f"{split_name.capitalize()} TAR saved to {tar_path}")

    logging.info("All TAR files created successfully.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    create_tar_files(args.options)
