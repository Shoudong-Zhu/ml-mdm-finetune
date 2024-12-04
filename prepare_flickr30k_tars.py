# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
import os
import tarfile
from io import BytesIO

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def filter_by_split(dataset, split_name):
    """
    Filter the dataset for a specific split.

    Args:
        dataset: Hugging Face dataset.
        split_name (str): Desired split (e.g., "train", "val", "test").

    Returns:
        Filtered dataset for the split.
    """
    return dataset.filter(lambda example: example["split"] == split_name)


def create_tar(dataset, tar_path):
    """
    Create a tar file from the given dataset split.

    Args:
        dataset: Filtered dataset split containing image data.
        tar_path (str): Path to save the tar file.
    """
    with tarfile.open(tar_path, "w") as tar:
        for item in tqdm(dataset, desc=f"Creating tar {os.path.basename(tar_path)}"):
            image_data = item["image"]
            filename = item["filename"]

            try:
                # Save image to an in-memory buffer
                buffer = BytesIO()
                image_data.save(buffer, format="JPEG")
                buffer.seek(0)

                # Create a tarinfo object
                tarinfo = tarfile.TarInfo(name=filename)
                tarinfo.size = len(buffer.getvalue())

                # Add the image buffer to the tar file
                tar.addfile(tarinfo, fileobj=buffer)
            except Exception as e:
                print(f"Failed to add {filename} to tar: {e}")


if __name__ == "__main__":
    # Load the Flickr30k dataset
    dataset = load_dataset("nlphuji/flickr30k")

    # Create the output directory for tar files
    tar_dir = "flickr30k_tars"
    os.makedirs(tar_dir, exist_ok=True)

    # Define split names and corresponding tar file paths
    splits = ["train", "val", "test"]
    tar_files = {
        "train": os.path.join(tar_dir, "flickr30k_train.tar"),
        "val": os.path.join(tar_dir, "flickr30k_val.tar"),
        "test": os.path.join(tar_dir, "flickr30k_test.tar"),
    }

    # Process each split
    for split_name in splits:
        print(f"Processing {split_name} split...")

        # Filter the dataset for the specific split
        split_dataset = filter_by_split(dataset["test"], split_name)

        if len(split_dataset) == 0:
            print(f"No samples found for split '{split_name}'. Skipping.")
            continue

        # Create the tar file for the split
        create_tar(split_dataset, tar_files[split_name])

    print("Tar file preparation completed!")
