#!/usr/bin/env python
"""
8GB Dataset Generator for SNN Training

Generates a large-scale temporal dataset optimized for SNN emulation training.
The dataset consists of spike-encoded temporal patterns with ground truth labels.

Dataset characteristics:
- Total size: ~8 GB
- N=4096 neurons (matching SNN architecture)
- T=256 time steps per sequence
- ~2,048 sequences (training + validation)
- Data format: HDF5 with chunked storage for efficient streaming

Data types:
1. Synthetic spike trains (Poisson processes)
2. Encoded temporal patterns (rate coding)
3. Event-driven sequences (sparse)

Usage:
    python scripts/generate_8gb_dataset.py --output data/snn_8gb.h5 --type poisson
    python scripts/generate_8gb_dataset.py --output data/snn_8gb.h5 --type temporal --num-classes 10
    python scripts/generate_8gb_dataset.py --quick  # Generate 1GB for testing
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time


def calculate_dataset_params(target_size_gb=8.0, N=4096, T=256, dtype=np.float32):
    """
    Calculate dataset parameters to achieve target size.

    Args:
        target_size_gb: Target dataset size in GB
        N: Number of neurons
        T: Time steps per sequence
        dtype: Data type

    Returns:
        params: Dict with calculated parameters
    """
    bytes_per_element = np.dtype(dtype).itemsize

    # Size per sequence (input + label)
    # Input: [T, N], Label: [N] (multi-label) or scalar (classification)
    input_size = T * N * bytes_per_element
    label_size = N * bytes_per_element  # Multi-label
    sequence_size = input_size + label_size

    # Calculate number of sequences
    target_bytes = target_size_gb * (1024 ** 3)
    num_sequences = int(target_bytes / sequence_size)

    # Split into train/val
    num_train = int(num_sequences * 0.9)
    num_val = num_sequences - num_train

    actual_size_gb = (num_sequences * sequence_size) / (1024 ** 3)

    params = {
        'N': N,
        'T': T,
        'num_sequences': num_sequences,
        'num_train': num_train,
        'num_val': num_val,
        'sequence_size_mb': sequence_size / (1024 ** 2),
        'actual_size_gb': actual_size_gb,
        'dtype': dtype,
    }

    return params


def generate_poisson_spike_trains(N, T, rate_min=0.05, rate_max=0.3, seed=None):
    """
    Generate Poisson spike trains.

    Args:
        N: Number of neurons
        T: Time steps
        rate_min: Minimum firing rate
        rate_max: Maximum firing rate
        seed: Random seed

    Returns:
        spikes: [T, N] binary spike array
        rates: [N] firing rates per neuron
    """
    if seed is not None:
        np.random.seed(seed)

    # Random firing rates per neuron
    rates = np.random.uniform(rate_min, rate_max, size=N)

    # Generate Poisson spikes
    spikes = np.random.rand(T, N) < rates[None, :]

    return spikes.astype(np.float32), rates.astype(np.float32)


def generate_temporal_pattern(N, T, num_patterns=5, pattern_length=20, noise_rate=0.05, seed=None):
    """
    Generate temporal patterns embedded in noise.

    Creates sequences with embedded temporal patterns that SNNs need to detect.

    Args:
        N: Number of neurons
        T: Time steps
        num_patterns: Number of embedded patterns
        pattern_length: Length of each pattern
        noise_rate: Background noise rate
        seed: Random seed

    Returns:
        sequence: [T, N] spike sequence
        pattern_labels: [num_patterns] pattern IDs
    """
    if seed is not None:
        np.random.seed(seed)

    # Background noise
    sequence = (np.random.rand(T, N) < noise_rate).astype(np.float32)

    # Embed patterns
    pattern_labels = []
    neurons_per_pattern = N // num_patterns

    for i in range(num_patterns):
        # Random pattern start time
        start_t = np.random.randint(0, T - pattern_length)

        # Pattern neurons
        start_n = i * neurons_per_pattern
        end_n = start_n + neurons_per_pattern

        # Create pattern (random binary pattern)
        pattern = np.random.rand(pattern_length, neurons_per_pattern) > 0.5

        # Embed pattern
        sequence[start_t:start_t+pattern_length, start_n:end_n] = pattern.astype(np.float32)

        pattern_labels.append(i)

    return sequence, np.array(pattern_labels, dtype=np.int32)


def generate_rate_encoded_sequence(N, T, num_classes=10, class_id=None, seed=None):
    """
    Generate rate-encoded temporal sequence for classification.

    Args:
        N: Number of neurons
        T: Time steps
        num_classes: Number of classes
        class_id: Class ID (if None, random)
        seed: Random seed

    Returns:
        sequence: [T, N] rate-encoded sequence
        label: [num_classes] one-hot label
    """
    if seed is not None:
        np.random.seed(seed)

    # Random class if not specified
    if class_id is None:
        class_id = np.random.randint(0, num_classes)

    # Create class-specific rate profile
    # Different classes have different spatial rate patterns
    neurons_per_class = N // num_classes
    rates = np.ones(N) * 0.05  # Background rate

    # Increase rate for neurons corresponding to this class
    start_n = class_id * neurons_per_class
    end_n = start_n + neurons_per_class
    rates[start_n:end_n] = np.random.uniform(0.3, 0.6, size=neurons_per_class)

    # Temporal modulation (creates temporal structure)
    temporal_modulation = np.sin(np.linspace(0, 4*np.pi, T)) * 0.2 + 1.0

    # Generate spikes
    sequence = np.random.rand(T, N) < (rates[None, :] * temporal_modulation[:, None])

    # One-hot label
    label = np.zeros(num_classes, dtype=np.float32)
    label[class_id] = 1.0

    return sequence.astype(np.float32), label


def generate_event_driven_sequence(N, T, num_events=50, event_spread=5, seed=None):
    """
    Generate event-driven sparse sequences.

    Simulates event-based sensors (DVS cameras, cochlea).

    Args:
        N: Number of neurons
        T: Time steps
        num_events: Number of events
        event_spread: Temporal spread of each event
        seed: Random seed

    Returns:
        sequence: [T, N] sparse event sequence
        event_times: [num_events] event timestamps
    """
    if seed is not None:
        np.random.seed(seed)

    sequence = np.zeros((T, N), dtype=np.float32)
    event_times = []

    for _ in range(num_events):
        # Random event time
        event_t = np.random.randint(0, T)
        event_times.append(event_t)

        # Random event neurons (localized)
        num_active = np.random.randint(5, 20)
        active_neurons = np.random.choice(N, size=num_active, replace=False)

        # Spread event over time
        for dt in range(event_spread):
            if event_t + dt < T:
                sequence[event_t + dt, active_neurons] = 1.0

    return sequence, np.array(event_times, dtype=np.int32)


def create_hdf5_dataset(
    output_path,
    params,
    data_type='poisson',
    num_classes=10,
    chunk_size=16,
    compression='gzip'
):
    """
    Create HDF5 dataset file.

    Args:
        output_path: Path to output HDF5 file
        params: Dataset parameters from calculate_dataset_params()
        data_type: 'poisson', 'temporal', 'rate', or 'event'
        num_classes: Number of classes (for 'rate' type)
        chunk_size: HDF5 chunk size
        compression: Compression type
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    N = params['N']
    T = params['T']
    num_train = params['num_train']
    num_val = params['num_val']

    print(f"Creating dataset: {output_path}")
    print(f"  Type: {data_type}")
    print(f"  Size: {params['actual_size_gb']:.2f} GB")
    print(f"  Sequences: {params['num_sequences']:,} ({num_train:,} train, {num_val:,} val)")
    print(f"  Shape per sequence: [{T}, {N}]")
    print(f"  Chunk size: {chunk_size}")
    print("")

    with h5py.File(output_path, 'w') as f:
        # Create datasets with appropriate chunk sizes
        train_chunk = min(chunk_size, num_train)
        val_chunk = min(chunk_size, num_val)

        train_inputs = f.create_dataset(
            'train/inputs',
            shape=(num_train, T, N),
            dtype=np.float32,
            chunks=(train_chunk, T, N),
            compression=compression,
            compression_opts=4 if compression == 'gzip' else None
        )

        if data_type == 'rate':
            # Classification: one-hot labels
            train_labels = f.create_dataset(
                'train/labels',
                shape=(num_train, num_classes),
                dtype=np.float32,
                chunks=(train_chunk, num_classes),
                compression=compression,
                compression_opts=4 if compression == 'gzip' else None
            )
        else:
            # Multi-label or regression
            train_labels = f.create_dataset(
                'train/labels',
                shape=(num_train, N),
                dtype=np.float32,
                chunks=(train_chunk, N),
                compression=compression,
                compression_opts=4 if compression == 'gzip' else None
            )

        val_inputs = f.create_dataset(
            'val/inputs',
            shape=(num_val, T, N),
            dtype=np.float32,
            chunks=(val_chunk, T, N),
            compression=compression,
            compression_opts=4 if compression == 'gzip' else None
        )

        if data_type == 'rate':
            val_labels = f.create_dataset(
                'val/labels',
                shape=(num_val, num_classes),
                dtype=np.float32,
                chunks=(val_chunk, num_classes),
                compression=compression,
                compression_opts=4 if compression == 'gzip' else None
            )
        else:
            val_labels = f.create_dataset(
                'val/labels',
                shape=(num_val, N),
                dtype=np.float32,
                chunks=(val_chunk, N),
                compression=compression,
                compression_opts=4 if compression == 'gzip' else None
            )

        # Generate training data
        print("Generating training data...")
        for i in tqdm(range(num_train), desc="Train"):
            seed = i

            if data_type == 'poisson':
                sequence, rates = generate_poisson_spike_trains(N, T, seed=seed)
                label = rates
            elif data_type == 'temporal':
                sequence, pattern_ids = generate_temporal_pattern(N, T, seed=seed)
                # Label: binary vector indicating pattern presence
                label = np.zeros(N, dtype=np.float32)
                for pid in pattern_ids:
                    neurons_per_pattern = N // 5
                    start_n = pid * neurons_per_pattern
                    end_n = start_n + neurons_per_pattern
                    label[start_n:end_n] = 1.0
            elif data_type == 'rate':
                class_id = i % num_classes
                sequence, label = generate_rate_encoded_sequence(N, T, num_classes, class_id, seed=seed)
            elif data_type == 'event':
                sequence, event_times = generate_event_driven_sequence(N, T, seed=seed)
                # Label: event density per neuron
                label = sequence.sum(axis=0) / T
            else:
                raise ValueError(f"Unknown data type: {data_type}")

            train_inputs[i] = sequence
            train_labels[i] = label

        # Generate validation data
        print("\nGenerating validation data...")
        for i in tqdm(range(num_val), desc="Val"):
            seed = num_train + i

            if data_type == 'poisson':
                sequence, rates = generate_poisson_spike_trains(N, T, seed=seed)
                label = rates
            elif data_type == 'temporal':
                sequence, pattern_ids = generate_temporal_pattern(N, T, seed=seed)
                label = np.zeros(N, dtype=np.float32)
                for pid in pattern_ids:
                    neurons_per_pattern = N // 5
                    start_n = pid * neurons_per_pattern
                    end_n = start_n + neurons_per_pattern
                    label[start_n:end_n] = 1.0
            elif data_type == 'rate':
                class_id = i % num_classes
                sequence, label = generate_rate_encoded_sequence(N, T, num_classes, class_id, seed=seed)
            elif data_type == 'event':
                sequence, event_times = generate_event_driven_sequence(N, T, seed=seed)
                label = sequence.sum(axis=0) / T
            else:
                raise ValueError(f"Unknown data type: {data_type}")

            val_inputs[i] = sequence
            val_labels[i] = label

        # Store metadata
        f.attrs['data_type'] = data_type
        f.attrs['N'] = N
        f.attrs['T'] = T
        f.attrs['num_train'] = num_train
        f.attrs['num_val'] = num_val
        f.attrs['num_classes'] = num_classes if data_type == 'rate' else 0
        f.attrs['size_gb'] = params['actual_size_gb']
        f.attrs['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

    print(f"\n✓ Dataset created: {output_path}")
    print(f"  Size: {output_path.stat().st_size / (1024**3):.2f} GB")


def inspect_dataset(path):
    """Inspect HDF5 dataset and print summary."""
    with h5py.File(path, 'r') as f:
        print(f"\nDataset: {path}")
        print(f"{'='*60}")

        # Metadata
        print("Metadata:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")

        print("\nDatasets:")
        for split in ['train', 'val']:
            if split in f:
                print(f"  {split}/")
                for key in f[split].keys():
                    ds = f[split][key]
                    print(f"    {key}: {ds.shape} {ds.dtype}")

        # Sample statistics
        print("\nSample statistics (first train sample):")
        inputs = f['train/inputs'][0]
        print(f"  Input shape: {inputs.shape}")
        print(f"  Sparsity: {(inputs == 0).mean():.2%}")
        print(f"  Mean activity: {inputs.mean():.4f}")
        print(f"  Max activity: {inputs.max():.4f}")

        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate 8GB dataset for SNN training")
    parser.add_argument('--output', type=str, default='data/snn_8gb.h5', help="Output HDF5 path")
    parser.add_argument('--type', type=str, default='rate', choices=['poisson', 'temporal', 'rate', 'event'],
                        help="Dataset type")
    parser.add_argument('--size-gb', type=float, default=8.0, help="Target size in GB")
    parser.add_argument('--N', type=int, default=4096, help="Number of neurons")
    parser.add_argument('--T', type=int, default=256, help="Time steps per sequence")
    parser.add_argument('--num-classes', type=int, default=10, help="Number of classes (for 'rate' type)")
    parser.add_argument('--chunk-size', type=int, default=16, help="HDF5 chunk size")
    parser.add_argument('--compression', type=str, default='gzip', choices=['gzip', 'lzf', None],
                        help="Compression type")
    parser.add_argument('--quick', action='store_true', help="Generate 1GB for testing")
    parser.add_argument('--inspect', action='store_true', help="Inspect existing dataset")

    args = parser.parse_args()

    if args.inspect:
        inspect_dataset(args.output)
        return

    # Adjust size for quick mode
    size_gb = 1.0 if args.quick else args.size_gb

    # Calculate parameters
    params = calculate_dataset_params(
        target_size_gb=size_gb,
        N=args.N,
        T=args.T
    )

    print(f"\nDataset Parameters:")
    print(f"{'='*60}")
    for key, value in params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*60}\n")

    # Create dataset
    create_hdf5_dataset(
        output_path=args.output,
        params=params,
        data_type=args.type,
        num_classes=args.num_classes,
        chunk_size=args.chunk_size,
        compression=args.compression
    )

    # Inspect result
    inspect_dataset(args.output)


if __name__ == '__main__':
    main()
