#!/usr/bin/env python3
"""
preprocess_traces.py

Normalize and segment raw .npy traces into a consistent format for ML.

Example transforms:
    - DC offset removal.
    - Z-score normalization.
    - Optional downsampling.

Usage:
    python sidechannel/preprocess_traces.py \
        --input-dir raw_traces \
        --output-dir processed_traces
"""

import argparse
from pathlib import Path
import numpy as np


def preprocess_trace(trace: np.ndarray) -> np.ndarray:
    """
    Preprocess a single power trace.

    Steps:
        1. Remove DC offset (subtract mean)
        2. Z-score normalization
        3. (Optional) bandpass filtering, downsampling

    Args:
        trace: Raw power trace

    Returns:
        Preprocessed trace
    """
    trace = trace.astype(float)

    # Remove DC offset
    trace = trace - trace.mean()

    # Z-score normalization
    std = trace.std() or 1.0
    trace = trace / std

    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    trace_files = list(args.input_dir.glob("*.npy"))
    print(f"Found {len(trace_files)} traces")

    for path in trace_files:
        trace = np.load(path)
        trace_p = preprocess_trace(trace)
        out_path = args.output_dir / path.name
        np.save(out_path, trace_p)

    print(f"Processed traces written to {args.output_dir}")


if __name__ == "__main__":
    main()
