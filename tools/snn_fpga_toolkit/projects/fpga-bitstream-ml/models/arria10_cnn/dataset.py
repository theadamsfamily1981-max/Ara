#!/usr/bin/env python3
"""
PyTorch Dataset for Arria 10 Entropy Images

Loads preprocessed entropy maps (PNG images) for CNN training/inference.

Directory structure expected:
    data/images/arria10/
        clean/
            seed_001_entropy.png
            seed_002_entropy.png
            ...
        infected/
            timebomb_001_entropy.png
            comparator_001_entropy.png
            ...

Labels:
    - clean = 0
    - infected = 1
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional
import json


class Arria10BitstreamDataset(Dataset):
    """PyTorch Dataset for Arria 10 entropy map images."""

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        transform: Optional[transforms.Compose] = None,
        cache_metadata: bool = True
    ):
        """
        Args:
            root_dir: Root directory containing clean/ and infected/ subdirs
            split: 'train', 'val', or 'test' (assumes pre-split directories)
            transform: Optional torchvision transforms
            cache_metadata: Load metadata.json files if available
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.cache_metadata = cache_metadata

        # Collect image paths and labels
        self.samples = []
        self._load_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {root_dir}")

    def _load_samples(self):
        """Scan directories and collect (image_path, label, metadata) tuples."""

        # Load clean samples (label = 0)
        clean_dir = self.root_dir / 'clean'
        if clean_dir.exists():
            for img_path in clean_dir.glob('**/*_entropy.png'):
                metadata = None
                if self.cache_metadata:
                    metadata_path = img_path.with_suffix('.json')
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            metadata = json.load(f)

                self.samples.append((str(img_path), 0, metadata))

        # Load infected samples (label = 1)
        infected_dir = self.root_dir / 'infected'
        if infected_dir.exists():
            for img_path in infected_dir.glob('**/*_entropy.png'):
                metadata = None
                if self.cache_metadata:
                    metadata_path = img_path.with_suffix('.json')
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            metadata = json.load(f)

                self.samples.append((str(img_path), 1, metadata))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            Tuple of (image_tensor, label)
            - image_tensor: shape (1, H, W) for grayscale
            - label: 0 (clean) or 1 (infected)
        """
        img_path, label, metadata = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert('L')  # Grayscale

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = transforms.ToTensor()(image)

        return image, label

    def get_metadata(self, idx: int) -> dict:
        """Get metadata for a specific sample."""
        _, _, metadata = self.samples[idx]
        return metadata if metadata is not None else {}

    def get_class_distribution(self) -> dict:
        """Get count of samples per class."""
        clean_count = sum(1 for _, label, _ in self.samples if label == 0)
        infected_count = sum(1 for _, label, _ in self.samples if label == 1)

        return {
            'clean': clean_count,
            'infected': infected_count,
            'total': len(self.samples),
            'balance_ratio': clean_count / max(infected_count, 1)
        }


def get_default_transforms(augment: bool = True) -> transforms.Compose:
    """
    Get default image transforms for training/validation.

    Args:
        augment: Apply data augmentation (for training)

    Returns:
        Composed transforms
    """
    transform_list = []

    if augment:
        # Data augmentation for training
        transform_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),  # 90° rotations only (maintain structure)
            transforms.RandomCrop(size=(224, 224), pad_if_needed=True),
        ])
    else:
        # Validation/test: center crop only
        transform_list.append(
            transforms.CenterCrop(size=(224, 224))
        )

    # Common transforms
    transform_list.extend([
        transforms.ToTensor(),  # Convert to [0, 1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])

    return transforms.Compose(transform_list)


def create_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.2
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders with automatic splitting.

    Args:
        root_dir: Root directory with clean/infected subdirs
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        val_split: Fraction of data to use for validation

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load full dataset
    full_dataset = Arria10BitstreamDataset(
        root_dir=root_dir,
        transform=get_default_transforms(augment=False)  # Will apply per-split
    )

    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    # Apply different transforms to each split
    train_dataset.dataset.transform = get_default_transforms(augment=True)
    val_dataset.dataset.transform = get_default_transforms(augment=False)

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset loading
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_root>")
        print("\nExample: python dataset.py data/images/arria10")
        sys.exit(1)

    root_dir = sys.argv[1]

    print(f"Loading dataset from: {root_dir}")
    print("-" * 60)

    dataset = Arria10BitstreamDataset(
        root_dir=root_dir,
        transform=get_default_transforms(augment=False)
    )

    print(f"Total samples: {len(dataset)}")
    print("\nClass distribution:")
    dist = dataset.get_class_distribution()
    for k, v in dist.items():
        print(f"  {k}: {v}")

    print("\nSample batch:")
    img, label = dataset[0]
    print(f"  Image shape: {img.shape}")
    print(f"  Label: {label} ({'clean' if label == 0 else 'infected'})")
    print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")

    if len(dataset) > 0:
        metadata = dataset.get_metadata(0)
        if metadata:
            print(f"\nMetadata keys: {list(metadata.keys())}")
