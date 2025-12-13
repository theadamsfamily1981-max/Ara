#!/usr/bin/env python3
"""
Experiment A: ρ Calibration + Dynamic Temperature
==================================================

Calibrates the criticality parameter ρ for Ara's LM dynamics and
implements the dynamic temperature control loop.

Goals:
1. Measure how ρ behaves on real training/inference workloads
2. Wire ρ → llm_temperature control
3. Verify system sits around ρ ≈ 0.8 (tempered criticality)

GUTC Theory:
    - ρ < 0.7: COLD (subcritical) → increase temp, more exploration
    - 0.7 ≤ ρ ≤ 0.85: OPTIMAL → healthy corridor
    - ρ > 0.9: HOT (supercritical) → decrease temp, enforce grounding
    - curvature spike: CRITICAL → emergency brake

Control Law:
    llm_temp = BASE_TEMP + (TARGET_RHO - current_rho) * GAIN

Usage:
    python experiments/experiment_calibrate_rho.py --data_path <path> --steps 1000

Output:
    logs/experiment_A_rho_calibration.csv with columns:
    step, loss, rho, curvature_var, status, llm_temperature, timestamp
"""

import csv
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import numpy as np

# Import the criticality monitor
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ara.cognition.meis import GradientCriticalityMonitor


# =============================================================================
# Configuration
# =============================================================================

TARGET_RHO = 0.8        # Target branching ratio for tempered criticality
BASE_TEMP = 0.7         # Base LLM temperature
GAIN = 0.6              # Control gain (tune based on results)
MIN_TEMP = 0.2          # Minimum temperature
MAX_TEMP = 1.5          # Maximum temperature

LOG_PATH = Path("logs/experiment_A_rho_calibration.csv")


# =============================================================================
# Utility Functions
# =============================================================================

def flatten_grads(model) -> np.ndarray:
    """Flatten all gradients from model into single array."""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.view(-1))
    if not grads:
        return np.zeros(1, dtype=np.float32)

    # Handle PyTorch tensors
    import torch
    return torch.cat(grads).detach().cpu().numpy()


def get_hidden_states(outputs: Dict[str, Any]) -> np.ndarray:
    """Extract hidden states from model outputs."""
    if isinstance(outputs, dict):
        if "hidden" in outputs:
            hidden = outputs["hidden"]
        elif "hidden_states" in outputs:
            hidden = outputs["hidden_states"][-1]  # Last layer
        elif "logits" in outputs:
            hidden = outputs["logits"]
        else:
            hidden = list(outputs.values())[0]
    else:
        hidden = outputs

    # Convert to numpy
    if hasattr(hidden, "detach"):
        return hidden.detach().cpu().numpy()
    return np.asarray(hidden)


# =============================================================================
# Main Experiment
# =============================================================================

def run_experiment(
    model,
    optimizer,
    data_loader,
    loss_fn: Callable,
    n_steps: int = 1000,
    device: str = "cuda",
    log_interval: int = 50,
):
    """
    Run ρ calibration experiment with dynamic temperature control.

    Args:
        model: PyTorch model with parameters
        optimizer: PyTorch optimizer
        data_loader: Iterable yielding batches
        loss_fn: Loss function(outputs, targets) -> loss
        n_steps: Number of training steps
        device: Device to run on
        log_interval: Steps between console logs
    """
    import torch

    monitor = GradientCriticalityMonitor(history_window=500, alert_threshold=3.0)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    llm_temperature = BASE_TEMP
    critical_count = 0
    hot_count = 0
    cold_count = 0

    with LOG_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "loss", "rho", "curvature_var", "status",
            "llm_temperature", "timestamp"
        ])

        step = 0
        for batch in data_loader:
            if step >= n_steps:
                break

            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(device) if hasattr(v, "to") else v
                         for k, v in batch.items()}

            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch, temperature=llm_temperature)

            # Compute loss
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                loss = loss_fn(outputs, batch.get("targets", batch.get("labels")))

            # Backward pass
            loss.backward()

            # Collect gradients and activations
            grads_np = flatten_grads(model)
            acts_np = get_hidden_states(outputs)

            # Update criticality monitor
            state = monitor.update(grads_np, acts_np)

            # === Homeostatic Policy ===

            # Emergency brake on CRITICAL
            if state.status == "CRITICAL":
                optimizer.zero_grad()
                critical_count += 1
                print(f"[{step}] CRITICAL: rho={state.rho:.3f}, "
                      f"curv={state.curvature_var:.3e} - SKIPPING UPDATE")
                step += 1
                continue

            # Adaptive LR on HOT/COLD
            if state.status == "HOT":
                hot_count += 1
                for g in optimizer.param_groups:
                    g["lr"] *= 0.95
            elif state.status == "COLD":
                cold_count += 1
                for g in optimizer.param_groups:
                    g["lr"] *= 1.02

            # Apply gradients
            optimizer.step()

            # === Dynamic Temperature Control ===
            llm_temperature = BASE_TEMP + (TARGET_RHO - state.rho) * GAIN
            llm_temperature = float(np.clip(llm_temperature, MIN_TEMP, MAX_TEMP))

            # Log to CSV
            writer.writerow([
                step,
                float(loss.detach().cpu().item()),
                state.rho,
                state.curvature_var,
                state.status,
                llm_temperature,
                time.time(),
            ])

            # Console logging
            if step % log_interval == 0:
                print(
                    f"[{step}] loss={loss.item():.4f} "
                    f"rho={state.rho:.3f} status={state.status} "
                    f"T={llm_temperature:.3f} curv={state.curvature_var:.3e}"
                )

            step += 1

    # Summary statistics
    print("\n" + "=" * 60)
    print("Experiment A Complete")
    print("=" * 60)
    print(f"Total steps: {step}")
    print(f"CRITICAL events: {critical_count} ({100*critical_count/max(1,step):.1f}%)")
    print(f"HOT events: {hot_count} ({100*hot_count/max(1,step):.1f}%)")
    print(f"COLD events: {cold_count} ({100*cold_count/max(1,step):.1f}%)")
    print(f"Log saved to: {LOG_PATH}")

    return monitor.get_diagnostics()


# =============================================================================
# Synthetic Data Mode (for testing without real model)
# =============================================================================

def run_synthetic_experiment(n_steps: int = 500):
    """
    Run experiment with synthetic data for testing the monitoring pipeline.

    Simulates different dynamical regimes:
    - Steps 0-150: Subcritical (λ=0.7)
    - Steps 150-350: Critical (λ=1.0)
    - Steps 350-500: Supercritical (λ=1.15)
    """
    print("\n" + "=" * 60)
    print("Experiment A: Synthetic Mode")
    print("=" * 60 + "\n")

    monitor = GradientCriticalityMonitor(history_window=200, alert_threshold=3.0)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    llm_temperature = BASE_TEMP

    # Regime schedule
    def get_true_lambda(step):
        if step < 150:
            return 0.7   # Subcritical
        elif step < 350:
            return 1.0   # Critical
        else:
            return 1.15  # Supercritical

    with LOG_PATH.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step", "true_lambda", "rho", "curvature_var", "status",
            "llm_temperature", "timestamp"
        ])

        for step in range(n_steps):
            true_lambda = get_true_lambda(step)

            # Generate synthetic gradients and activations
            # Gradient direction changes more in unstable regimes
            curv_noise = 0.1 if true_lambda == 1.0 else 0.3
            grads = np.random.randn(500) * (1.0 + curv_noise * np.sin(step / 10))

            # Activations scale with lambda
            act_scale = 0.9 if step == 0 else true_lambda * prev_act_scale
            act_scale = np.clip(act_scale, 0.1, 3.0)
            acts = np.random.randn(200) * act_scale
            prev_act_scale = act_scale

            # Update monitor
            state = monitor.update(grads, acts)

            # Dynamic temperature
            llm_temperature = BASE_TEMP + (TARGET_RHO - state.rho) * GAIN
            llm_temperature = float(np.clip(llm_temperature, MIN_TEMP, MAX_TEMP))

            # Log
            writer.writerow([
                step,
                true_lambda,
                state.rho,
                state.curvature_var,
                state.status,
                llm_temperature,
                time.time(),
            ])

            if step % 50 == 0:
                print(
                    f"[{step}] true_λ={true_lambda:.2f} "
                    f"ρ={state.rho:.3f} status={state.status} "
                    f"T={llm_temperature:.3f}"
                )

    print(f"\nLog saved to: {LOG_PATH}")

    # Final diagnostics
    diag = monitor.get_diagnostics()
    print("\nFinal diagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    return diag


# =============================================================================
# Analysis Utilities
# =============================================================================

def analyze_log(log_path: Path = LOG_PATH):
    """Analyze experiment log and print statistics."""
    import pandas as pd

    df = pd.read_csv(log_path)

    print("\n" + "=" * 60)
    print("Log Analysis")
    print("=" * 60)

    print(f"\nTotal samples: {len(df)}")
    print(f"\nρ Statistics:")
    print(f"  Mean: {df['rho'].mean():.3f}")
    print(f"  Std:  {df['rho'].std():.3f}")
    print(f"  Min:  {df['rho'].min():.3f}")
    print(f"  Max:  {df['rho'].max():.3f}")

    print(f"\nTemperature Statistics:")
    print(f"  Mean: {df['llm_temperature'].mean():.3f}")
    print(f"  Std:  {df['llm_temperature'].std():.3f}")

    print(f"\nStatus Distribution:")
    for status, count in df['status'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {status}: {count} ({pct:.1f}%)")

    # Check if ρ converges to target
    last_100 = df.tail(100)
    mean_rho_last = last_100['rho'].mean()
    deviation = abs(mean_rho_last - TARGET_RHO)

    print(f"\nConvergence Check (last 100 samples):")
    print(f"  Mean ρ: {mean_rho_last:.3f}")
    print(f"  Target: {TARGET_RHO}")
    print(f"  Deviation: {deviation:.3f}")

    if deviation < 0.1:
        print("  → GOOD: ρ is near target")
    else:
        print("  → NEEDS TUNING: ρ not converging to target")

    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Experiment A: ρ Calibration")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run with synthetic data (no model needed)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of steps")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze existing log instead of running")
    parser.add_argument("--log_path", type=str, default=str(LOG_PATH),
                        help="Path to log file")

    args = parser.parse_args()

    if args.analyze:
        analyze_log(Path(args.log_path))
    elif args.synthetic:
        run_synthetic_experiment(n_steps=args.steps)
    else:
        print("Real model mode requires model/optimizer/data_loader.")
        print("Use --synthetic for testing, or import run_experiment() directly.")
        print("\nExample usage in code:")
        print("  from experiments.experiment_calibrate_rho import run_experiment")
        print("  run_experiment(model, optimizer, data_loader, loss_fn)")


if __name__ == "__main__":
    main()
