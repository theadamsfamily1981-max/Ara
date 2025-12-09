"""
capture_stub.py

Abstraction layer for capturing side-channel power traces.

Goal:
    - Provide a simple Python API:
        capture_traces(num_traces, label, out_dir)
    - In the lab, this would talk to:
        * An oscilloscope (e.g. via SCPI over LAN/USB)
        * Or a custom ADC/BMC

For now:
    - Implement a dummy generator that produces synthetic traces
      (noisy sine waves) so the rest of the pipeline is testable.

Claude:
    - Add hooks for real hardware later.
"""

from pathlib import Path
import numpy as np


def capture_traces(num_traces: int, label: str, out_dir: Path, trace_len: int = 2000):
    """
    Capture power traces from the target device.

    Args:
        num_traces: Number of traces to capture
        label: Label for this batch (e.g., "clean" or "trojan")
        out_dir: Output directory for .npy files
        trace_len: Length of each trace in samples

    For now:
        Generates synthetic traces for testing.

    TODO:
        - Add real oscilloscope interface (SCPI, PyVISA, etc.)
        - Add triggering logic
        - Add averaging/filtering options
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_traces):
        # Dummy synthetic trace
        t = np.linspace(0, 1, trace_len)
        trace = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(trace_len)
        path = out_dir / f"{label}_{i:05d}.npy"
        np.save(path, trace)

    print(f"Captured {num_traces} {label} traces to {out_dir}")


if __name__ == "__main__":
    # Test generation
    capture_traces(10, "clean", Path("test_traces/clean"))
    capture_traces(10, "trojan", Path("test_traces/trojan"))
