# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All rights reserved.
from datasets import load_dataset
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import transforms


def preprocess_data(batch, transform):
    """
    Preprocess each batch by applying the given transform to the images.

    Args:
        batch (dict): A batch from the dataset with 'image' and 'caption'.
        transform: Transformation to apply to the images.

    Returns:
        dict: Preprocessed batch with 'image' and 'caption'.
    """
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch


def create_data_loader(split, batch_size=32, num_workers=4):
    """
    Create a PyTorch DataLoader for the Flickr30k dataset.

    Args:
        split (str): Dataset split to load ('train', 'val', 'test').
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of worker processes for DataLoader.

    Returns:
        DataLoader: PyTorch DataLoader for the dataset.
    """
    # Load the dataset from Hugging Face
    dataset = load_dataset("nlphuji/flickr30k", split=split)

    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Apply transformations using the `map` function
    dataset = dataset.map(
        lambda batch: preprocess_data(batch, transform),
        batched=True,
        num_proc=num_workers,
    )

    # Create a DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=lambda x: {
            "images": torch.stack([item["image"] for item in x]),
            "captions": [item["caption"] for item in x],
        },
    )


if __name__ == "__main__":
    # Example usage
    batch_size = 32
    train_loader = create_data_loader("train", batch_size=batch_size)
    val_loader = create_data_loader("val", batch_size=batch_size)

    for batch in train_loader:
        images, captions = batch["images"], batch["captions"]
        print(f"Batch images: {images.shape}")
        print(f"Batch captions: {captions}")
        break
