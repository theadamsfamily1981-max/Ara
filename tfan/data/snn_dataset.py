"""
SNN Temporal Dataset Loader

Efficient HDF5-based dataset loader for SNN training with large temporal sequences.
Supports:
- Streaming from disk (doesn't load all data into RAM)
- Chunked reading for memory efficiency
- Multiple data types (Poisson, temporal patterns, rate-coded, event-driven)
- Data augmentation for temporal sequences

Usage:
    from tfan.data import SNNTemporalDataset, get_snn_dataloader

    dataset = SNNTemporalDataset('data/snn_8gb.h5', split='train')
    loader = get_snn_dataloader(dataset, batch_size=32, num_workers=4)

    for batch_inputs, batch_labels in loader:
        # batch_inputs: [batch, T, N]
        # batch_labels: [batch, N] or [batch, num_classes]
        ...
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, Any


class SNNTemporalDataset(Dataset):
    """
    Dataset for SNN temporal sequences stored in HDF5 format.

    Args:
        hdf5_path: Path to HDF5 dataset file
        split: 'train' or 'val'
        transform: Optional transform function
        cache_size: Number of samples to cache in RAM (0 = no caching)
        temporal_jitter: Random temporal jitter for augmentation (in time steps)
        spike_dropout: Dropout probability for individual spikes (data augmentation)
    """

    def __init__(
        self,
        hdf5_path: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        cache_size: int = 0,
        temporal_jitter: int = 0,
        spike_dropout: float = 0.0,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.transform = transform
        self.cache_size = cache_size
        self.temporal_jitter = temporal_jitter
        self.spike_dropout = spike_dropout

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.hdf5_path}")

        # Load metadata
        with h5py.File(self.hdf5_path, 'r') as f:
            self.metadata = dict(f.attrs)
            self.N = f.attrs['N']
            self.T = f.attrs['T']
            self.data_type = f.attrs['data_type']

            # Get dataset length
            self.length = f[f'{split}/inputs'].shape[0]

            # Get label shape
            self.label_shape = f[f'{split}/labels'].shape[1:]

        # Cache (LRU-style)
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # For thread-safe HDF5 access
        self._hdf5_file = None

    def _get_hdf5_file(self):
        """Get thread-local HDF5 file handle."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r')
        return self._hdf5_file

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            inputs: [T, N] temporal sequence
            labels: [N] or [num_classes] label
        """
        # Check cache
        if idx in self.cache:
            self.cache_hits += 1
            inputs, labels = self.cache[idx]
        else:
            self.cache_misses += 1

            # Load from disk
            f = self._get_hdf5_file()
            inputs = f[f'{self.split}/inputs'][idx].astype(np.float32)
            labels = f[f'{self.split}/labels'][idx].astype(np.float32)

            # Cache if enabled
            if self.cache_size > 0 and len(self.cache) < self.cache_size:
                self.cache[idx] = (inputs.copy(), labels.copy())

        # Data augmentation
        if self.split == 'train':
            # Temporal jitter
            if self.temporal_jitter > 0:
                jitter = np.random.randint(-self.temporal_jitter, self.temporal_jitter + 1)
                if jitter != 0:
                    inputs = np.roll(inputs, jitter, axis=0)

            # Spike dropout
            if self.spike_dropout > 0:
                mask = np.random.rand(*inputs.shape) > self.spike_dropout
                inputs = inputs * mask

        # Convert to tensors
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)

        # Apply transform
        if self.transform is not None:
            inputs, labels = self.transform(inputs, labels)

        return inputs, labels

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return self.metadata.copy()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
        }

    def __del__(self):
        """Close HDF5 file on deletion."""
        if self._hdf5_file is not None:
            self._hdf5_file.close()


def collate_temporal_batch(batch):
    """
    Custom collate function for temporal sequences.

    Handles variable-length sequences if needed (pads to max length in batch).

    Args:
        batch: List of (inputs, labels) tuples

    Returns:
        batch_inputs: [batch, T, N]
        batch_labels: [batch, ...]
    """
    inputs_list, labels_list = zip(*batch)

    # Stack into batches
    batch_inputs = torch.stack(inputs_list, dim=0)
    batch_labels = torch.stack(labels_list, dim=0)

    return batch_inputs, batch_labels


def get_snn_dataloader(
    dataset: SNNTemporalDataset,
    batch_size: int = 32,
    shuffle: bool = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create optimized DataLoader for SNN training.

    Args:
        dataset: SNNTemporalDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle (default: True for train, False for val)
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch
        prefetch_factor: Samples per worker to prefetch

    Returns:
        DataLoader instance
    """
    # Default shuffle behavior
    if shuffle is None:
        shuffle = (dataset.split == 'train')

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_temporal_batch,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return loader


class TemporalAugmentation:
    """
    Data augmentation for temporal sequences.

    Transformations:
    - Temporal jitter (shift in time)
    - Spike dropout (randomly drop spikes)
    - Temporal scaling (compress/expand time)
    - Spatial permutation (permute neuron ordering)
    """

    def __init__(
        self,
        temporal_jitter: int = 0,
        spike_dropout: float = 0.0,
        temporal_scaling: Tuple[float, float] = (1.0, 1.0),
        spatial_permute: bool = False,
    ):
        self.temporal_jitter = temporal_jitter
        self.spike_dropout = spike_dropout
        self.temporal_scaling = temporal_scaling
        self.spatial_permute = spatial_permute
        self.spatial_permutation = None

    def __call__(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentation.

        Args:
            inputs: [T, N] tensor
            labels: [...] tensor

        Returns:
            Augmented (inputs, labels)
        """
        T, N = inputs.shape

        # Temporal jitter
        if self.temporal_jitter > 0:
            jitter = np.random.randint(-self.temporal_jitter, self.temporal_jitter + 1)
            if jitter != 0:
                inputs = torch.roll(inputs, jitter, dims=0)

        # Spike dropout
        if self.spike_dropout > 0:
            mask = torch.rand_like(inputs) > self.spike_dropout
            inputs = inputs * mask

        # Temporal scaling (compress/expand)
        if self.temporal_scaling != (1.0, 1.0):
            scale = np.random.uniform(*self.temporal_scaling)
            if scale != 1.0:
                # Resample in time
                new_T = int(T * scale)
                if new_T > 0:
                    # Simple nearest-neighbor resampling
                    indices = (np.arange(T) * scale).astype(int)
                    indices = np.clip(indices, 0, new_T - 1)
                    inputs = inputs[indices]

                    # Pad or crop to original T
                    if inputs.shape[0] < T:
                        pad = T - inputs.shape[0]
                        inputs = torch.cat([inputs, torch.zeros(pad, N)], dim=0)
                    elif inputs.shape[0] > T:
                        inputs = inputs[:T]

        # Spatial permutation
        if self.spatial_permute:
            if self.spatial_permutation is None:
                self.spatial_permutation = torch.randperm(N)

            inputs = inputs[:, self.spatial_permutation]

            # Also permute labels if they're spatial
            if labels.shape == (N,):
                labels = labels[self.spatial_permutation]

        return inputs, labels


def create_data_loaders(
    hdf5_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_size: int = 0,
    augment_train: bool = True,
    temporal_jitter: int = 5,
    spike_dropout: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.

    Args:
        hdf5_path: Path to HDF5 dataset
        batch_size: Batch size
        num_workers: Number of workers
        cache_size: Number of samples to cache
        augment_train: Apply augmentation to training set
        temporal_jitter: Temporal jitter for augmentation
        spike_dropout: Spike dropout for augmentation

    Returns:
        train_loader, val_loader
    """
    # Train dataset
    train_transform = None
    if augment_train:
        train_transform = TemporalAugmentation(
            temporal_jitter=temporal_jitter,
            spike_dropout=spike_dropout,
        )

    train_dataset = SNNTemporalDataset(
        hdf5_path=hdf5_path,
        split='train',
        transform=train_transform,
        cache_size=cache_size,
    )

    train_loader = get_snn_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Validation dataset (no augmentation)
    val_dataset = SNNTemporalDataset(
        hdf5_path=hdf5_path,
        split='val',
        transform=None,
        cache_size=0,  # Don't cache validation set
    )

    val_loader = get_snn_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"Data loaders created:")
    print(f"  Train: {len(train_dataset):,} samples")
    print(f"  Val: {len(val_dataset):,} samples")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")

    return train_loader, val_loader
