"""
Memory scaling validation for TF-A-N 7B.

Validates memory scaling gate: α < 1.0 (sublinear scaling).

Usage:
    python scripts/memory_fit.py --seq 1024 2048 5120 10240
"""

import torch
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.models.tfan7b import TFANConfig, TFANForCausalLM


def measure_memory(model, seq_len, device="cuda"):
    """Measure peak memory usage for given sequence length."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    # Create input
    input_ids = torch.randint(0, 32768, (1, seq_len), device=device)

    # Forward pass
    with torch.no_grad():
        _ = model(input_ids)

    if device == "cuda":
        memory_mb = torch.cuda.max_memory_allocated() / 1e6
    else:
        memory_mb = 0  # Placeholder for CPU

    return memory_mb


def main(args):
    """Main function."""
    print("Memory scaling validation for TF-A-N 7B")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config = TFANConfig.from_json_file(args.config)

    # Create model
    print("Creating model...")
    model = TFANForCausalLM(config).to(device).eval()

    # Test sequence lengths
    seq_lengths = [1024, 2048, 4096, 8192, 16384]
    memories = []

    print(f"\nMeasuring memory at different sequence lengths...")

    for seq_len in seq_lengths:
        try:
            memory_mb = measure_memory(model, seq_len, device=str(device))
            memories.append(memory_mb)
            print(f"  seq_len={seq_len:>5d}: {memory_mb:.1f} MB")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  seq_len={seq_len:>5d}: OOM")
                break
            else:
                raise

    # Fit power law: Memory = a * T^α
    if len(memories) >= 3:
        log_T = np.log(seq_lengths[:len(memories)])
        log_M = np.log(memories)
        coeffs = np.polyfit(log_T, log_M, deg=1)
        alpha = coeffs[0]
        a = np.exp(coeffs[1])

        # Compute R²
        log_M_pred = coeffs[0] * log_T + coeffs[1]
        ss_res = np.sum((log_M - log_M_pred) ** 2)
        ss_tot = np.sum((log_M - np.mean(log_M)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        print(f"\nPower law fit: Memory = {a:.2f} * T^{alpha:.3f}")
        print(f"R² = {r_squared:.4f}")

        gate_pass = alpha < 1.0
        print(f"\nGate validation (α < 1.0): {gate_pass}")
        print(f"  α = {alpha:.3f} | {'✓ PASS' if gate_pass else '✗ FAIL'}")

        # Save results
        os.makedirs("artifacts/memory", exist_ok=True)
        results = {
            "alpha": float(alpha),
            "a": float(a),
            "r_squared": float(r_squared),
            "gate_pass": gate_pass,
            "measurements": [
                {"seq_len": int(s), "memory_mb": float(m)}
                for s, m in zip(seq_lengths[:len(memories)], memories)
            ],
        }

        with open("artifacts/memory/fit.json", "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to artifacts/memory/fit.json")

    else:
        print("\nNot enough data points for fitting")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="tfan/models/tfan7b/config.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
