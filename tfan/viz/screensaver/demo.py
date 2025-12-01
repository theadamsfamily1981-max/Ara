#!/usr/bin/env python3
"""
Quick demo script to show T-FAN Topology Screensaver

Generates sample metrics and launches screensaver with live updates.
"""

import json
import time
import subprocess
import threading
from pathlib import Path
import numpy as np

METRICS_FILE = Path.home() / ".cache" / "tfan" / "metrics.json"

def generate_demo_metrics():
    """Generate realistic-looking metrics that change over time."""
    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("🎬 Generating demo metrics...")
    print(f"📊 Writing to: {METRICS_FILE}")

    step = 0
    while True:
        # Simulate training progress with realistic values
        t = time.time()
        step += np.random.randint(5, 15)

        # Metrics oscillate realistically
        accuracy = 0.85 + 0.10 * np.sin(t * 0.1) + 0.02 * np.random.randn()
        latency_ms = 140.0 + 20.0 * np.sin(t * 0.15) + 5.0 * np.random.randn()
        hypervolume = 45000 + 5000 * np.sin(t * 0.08) + 500 * np.random.randn()
        epr_cv = 0.10 + 0.03 * np.sin(t * 0.12) + 0.01 * np.random.randn()
        topo_gap = 0.015 + 0.005 * np.sin(t * 0.09) + 0.002 * np.random.randn()

        # Clamp to realistic ranges
        accuracy = np.clip(accuracy, 0.0, 1.0)
        latency_ms = np.clip(latency_ms, 80.0, 250.0)
        hypervolume = np.clip(hypervolume, 30000, 60000)
        epr_cv = np.clip(epr_cv, 0.05, 0.20)
        topo_gap = np.clip(topo_gap, 0.005, 0.030)

        metrics = {
            "training_active": True,
            "step": int(step),
            "accuracy": float(accuracy),
            "latency_ms": float(latency_ms),
            "hypervolume": float(hypervolume),
            "epr_cv": float(epr_cv),
            "topo_gap": float(topo_gap),
            "topo_cos": 0.93,
            "energy": 0.8,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f, indent=2)

        time.sleep(2.0)  # Update every 2 seconds

def main():
    print("=" * 60)
    print("🌌 T-FAN Topology Screensaver - Demo Mode")
    print("=" * 60)
    print()
    print("This demo will:")
    print("  1. Generate realistic metrics every 2 seconds")
    print("  2. Start metrics bridge on port 9101")
    print("  3. Launch screensaver in fullscreen")
    print()
    print("Controls:")
    print("  M or Tab  - Cycle through 4 visualization modes")
    print("  P or Space - Pause/unpause animation")
    print("  Q or Esc   - Quit demo")
    print()
    input("Press Enter to start demo...")

    # Start metrics generator thread
    metrics_thread = threading.Thread(target=generate_demo_metrics, daemon=True)
    metrics_thread.start()

    print("\n✓ Started metrics generator")
    print("✓ Metrics file:", METRICS_FILE)
    time.sleep(1)

    # Start metrics bridge
    print("\n🌐 Starting metrics bridge on http://localhost:9101/metrics...")
    bridge_proc = subprocess.Popen(
        ['python', 'metrics_bridge.py', '--port', '9101'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)
    print("✓ Metrics bridge running")

    # Launch screensaver
    print("\n🎨 Launching screensaver in fullscreen...")
    print("   (Starting with random mode, press M to cycle)")
    time.sleep(1)

    try:
        subprocess.run([
            'python', 'topo_screensaver.py',
            '--metrics', 'http://localhost:9101/metrics',
            '--fullscreen'
        ])
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\n🛑 Shutting down demo...")
        bridge_proc.terminate()
        bridge_proc.wait()
        print("✓ Demo stopped")

if __name__ == "__main__":
    main()
