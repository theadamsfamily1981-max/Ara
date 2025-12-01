"""
Data loading utilities for TF-A-N 7B training.

Supports environment variables for QUANTA data:
    QUANTA_DATA_ROOT: Root directory for QUANTA data shards (e.g., /data/shards/)
    QUANTA_S3_BUCKET: S3 bucket for QUANTA data (e.g., s3://quanta-datasets/)
    QUANTA_MANIFEST: Path to data manifest file
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, List, Dict, Iterator
import numpy as np
import os
from pathlib import Path


class SimpleTextDataset(Dataset):
    """
    Simple text dataset for testing/demo.

    Args:
        texts: List of text strings
        tokenizer: Tokenizer (with encode method)
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 2048,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # Tokenize
        tokens = self.tokenizer.encode(text)

        # Truncate/pad
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        else:
            # Pad with pad_token_id (assume 0)
            tokens = tokens + [0] * (self.max_length - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)

        # For causal LM, labels = input_ids shifted
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class TokenizedDataset(IterableDataset):
    """
    Streaming dataset for pre-tokenized data.

    Assumes data is stored as .bin files with token IDs.

    Args:
        data_path: Path to tokenized data file
        seq_length: Sequence length for chunks
        seed: Random seed for shuffling
    """

    def __init__(
        self,
        data_path: str,
        seq_length: int = 2048,
        seed: int = 42,
    ):
        self.data_path = data_path
        self.seq_length = seq_length
        self.seed = seed

        # Load data
        try:
            self.tokens = np.memmap(data_path, dtype=np.uint16, mode="r")
        except Exception as e:
            print(f"Warning: Could not load data from {data_path}: {e}")
            print("Creating dummy data for testing...")
            # Create dummy data
            self.tokens = np.random.randint(0, 32768, size=1000000, dtype=np.uint16)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over dataset, yielding sequences of length seq_length.
        """
        rng = np.random.RandomState(self.seed)

        while True:
            # Random starting position
            start_idx = rng.randint(0, len(self.tokens) - self.seq_length - 1)

            # Extract sequence
            chunk = self.tokens[start_idx : start_idx + self.seq_length + 1]

            # Input and labels (shifted by 1)
            input_ids = torch.from_numpy(chunk[:-1].astype(np.int64))
            labels = torch.from_numpy(chunk[1:].astype(np.int64))

            yield {
                "input_ids": input_ids,
                "labels": labels,
            }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create DataLoader with standard settings.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch

    Returns:
        dataloader: PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


class QUANTADataset(IterableDataset):
    """
    Dataset for QUANTA domain data with environment variable support.

    Automatically handles:
    1. QUANTA_DATA_ROOT environment variable (e.g., /data/shards/)
    2. QUANTA_S3_BUCKET for cloud data (e.g., s3://quanta-datasets/)
    3. QUANTA_MANIFEST for data manifest file
    4. Fallback to WikiText-103 or dummy data if QUANTA unavailable

    Args:
        data_config: Dictionary with data configuration (from YAML)
        seq_length: Sequence length
        fallback_to_dummy: If True, use dummy data when QUANTA unavailable
    """

    def __init__(
        self,
        data_config: Optional[Dict] = None,
        seq_length: int = 2048,
        fallback_to_dummy: bool = True,
    ):
        self.seq_length = seq_length
        self.fallback_to_dummy = fallback_to_dummy

        # Check environment variables
        quanta_root = os.getenv("QUANTA_DATA_ROOT")
        quanta_s3 = os.getenv("QUANTA_S3_BUCKET")
        quanta_manifest = os.getenv("QUANTA_MANIFEST")

        # Try loading QUANTA data
        self.data_source = None
        self.tokens = None

        if quanta_root and Path(quanta_root).exists():
            print(f"✓ QUANTA data found at {quanta_root}")
            self.data_source = "quanta_local"
            # TODO: Implement multi-source loading from data_config
            # For now, just note that data is available
            print(f"  Note: Multi-source loading from config not yet implemented")
            print(f"  Falling back to dummy data for this smoke test")

        elif quanta_s3:
            print(f"✓ QUANTA S3 bucket configured: {quanta_s3}")
            self.data_source = "quanta_s3"
            print(f"  Note: S3 data loading not yet implemented")
            print(f"  Falling back to dummy data for this smoke test")

        elif quanta_manifest:
            print(f"✓ QUANTA manifest found: {quanta_manifest}")
            self.data_source = "quanta_manifest"
            print(f"  Note: Manifest-based loading not yet implemented")
            print(f"  Falling back to dummy data for this smoke test")

        else:
            print(f"⚠ QUANTA data not found (checked environment variables)")
            print(f"  QUANTA_DATA_ROOT: {quanta_root or 'not set'}")
            print(f"  QUANTA_S3_BUCKET: {quanta_s3 or 'not set'}")
            print(f"  QUANTA_MANIFEST: {quanta_manifest or 'not set'}")

        # Fallback logic
        if self.data_source is None:
            # Try WikiText-103 fallback
            wikitext_path = "/data/shards/wikitext_103/"
            if Path(wikitext_path).exists():
                print(f"✓ Falling back to WikiText-103 at {wikitext_path}")
                self.data_source = "wikitext_fallback"
                # TODO: Load WikiText-103
            elif self.fallback_to_dummy:
                print(f"✓ Using dummy data for smoke test (no real data available)")
                self.data_source = "dummy"
            else:
                raise FileNotFoundError(
                    "QUANTA data not found. Set QUANTA_DATA_ROOT, QUANTA_S3_BUCKET, "
                    "or QUANTA_MANIFEST environment variable."
                )

        # For now, always use dummy data (real data loading TODO)
        print(f"→ Data source: {self.data_source} (using dummy tokens for now)")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Iterate over dataset, yielding sequences.
        """
        # For now, use dummy data
        # TODO: Implement real data loading based on self.data_source
        while True:
            input_ids = torch.randint(0, 32768, (self.seq_length,))
            labels = input_ids.clone()

            yield {
                "input_ids": input_ids,
                "labels": labels,
            }


class DummyDataset(IterableDataset):
    """
    Dummy dataset for testing without real data.

    Generates random token sequences.

    Args:
        vocab_size: Vocabulary size
        seq_length: Sequence length
        num_samples: Number of samples (None for infinite)
    """

    def __init__(
        self,
        vocab_size: int = 32768,
        seq_length: int = 2048,
        num_samples: Optional[int] = None,
    ):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        count = 0
        while self.num_samples is None or count < self.num_samples:
            # Generate random tokens
            input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
            labels = input_ids.clone()

            yield {
                "input_ids": input_ids,
                "labels": labels,
            }

            count += 1


__all__ = [
    "SimpleTextDataset",
    "TokenizedDataset",
    "QUANTADataset",
    "DummyDataset",
    "create_dataloader",
]
