#!/usr/bin/env python3
"""
HSF Demo - Hypervector Spiking Field in Action
================================================

This demo shows the complete HSF pipeline:
1. Create telemetry sources (GPU, network, service)
2. Create lanes and field
3. Run normal operation, learn baseline
4. Inject anomalies, watch detector respond
5. Show resonance-based pattern matching

Run: python -m ara.cognition.hsf.demo
"""

import numpy as np
import time
from typing import Dict

from .lanes import TelemetryLane, ItemMemory
from .field import HSField
from .detector import AnomalyDetector, AnomalySeverity, create_default_patterns
from .telemetry import (
    TelemetryMux, GPUTelemetry, NetworkTelemetry, ServiceTelemetry
)


def setup_field_and_lanes() -> tuple[HSField, Dict[str, TelemetryLane]]:
    """Create the HSF with lanes for each telemetry source."""
    field = HSField(dim=8192, stream_decay=0.95)

    # GPU lane
    gpu_lane = field.add_lane_config(
        name="gpu",
        features=["temp", "util", "mem", "power", "fan"],
        ranges={
            "temp": (30, 100),
            "util": (0, 1),
            "mem": (0, 1),
            "power": (50, 400),
            "fan": (0, 1),
        }
    )

    # Network lane
    net_lane = field.add_lane_config(
        name="network",
        features=["bps_in", "bps_out", "pps_in", "pps_out", "errors", "drops", "latency_ms"],
        ranges={
            "bps_in": (0, 10e9),
            "bps_out": (0, 10e9),
            "pps_in": (0, 1e7),
            "pps_out": (0, 1e7),
            "errors": (0, 100),
            "drops": (0, 100),
            "latency_ms": (0, 50),
        }
    )

    # Service lane
    svc_lane = field.add_lane_config(
        name="service",
        features=["request_rate", "error_rate", "latency_p50", "latency_p99",
                  "queue_depth", "cpu", "memory"],
        ranges={
            "request_rate": (0, 1000),
            "error_rate": (0, 1),
            "latency_p50": (0, 100),
            "latency_p99": (0, 500),
            "queue_depth": (0, 100),
            "cpu": (0, 1),
            "memory": (0, 1),
        }
    )

    return field, {"gpu": gpu_lane, "network": net_lane, "service": svc_lane}


def run_demo():
    """Run the complete HSF demo."""
    print("=" * 60)
    print("Hypervector Spiking Field (HSF) Demo")
    print("=" * 60)
    print()

    # Setup
    print("[1/5] Setting up field and telemetry sources...")
    field, lanes = setup_field_and_lanes()
    mux = TelemetryMux.create_default()
    detector = AnomalyDetector(field)

    # Add some default patterns
    for pattern in create_default_patterns(field):
        detector.add_pattern(pattern)

    print(f"  - Field dimension: {field.dim}")
    print(f"  - Lanes: {list(lanes.keys())}")
    print(f"  - Anomaly patterns: {len(detector.patterns)}")
    print()

    # Phase 1: Learn baseline
    print("[2/5] Learning baseline (100 samples of normal operation)...")
    for i in range(100):
        telemetry = mux.sample_all()
        field.update_all(telemetry)
        field.compute_field()

    field.compute_baseline()
    print(f"  - Baseline learned from {field.update_count} samples")
    print(f"  - Initial deviation: {field.total_deviation():.2%}")
    print()

    # Phase 2: Normal operation
    print("[3/5] Running normal operation (20 samples)...")
    for i in range(20):
        telemetry = mux.sample_all()
        field.update_all(telemetry)
        field.compute_field()

        if i % 5 == 0:
            severity, dev = detector.quick_check()
            print(f"  - Sample {i}: deviation={dev:.2%}, severity={severity.name}")

    print()

    # Phase 3: Inject anomalies
    print("[4/5] Injecting anomalies...")
    print()

    anomaly_scenarios = [
        ("gpu", "thermal_runaway", "GPU thermal runaway"),
        ("network", "congestion", "Network congestion"),
        ("service", "overload", "Service overload"),
    ]

    for source_name, anomaly_type, description in anomaly_scenarios:
        print(f"  Injecting: {description}")
        mux.inject_anomaly(source_name, anomaly_type, duration=5)

        # Run a few samples during anomaly
        for i in range(5):
            telemetry = mux.sample_all()
            field.update_all(telemetry)
            field.compute_field()

        # Full scan
        report = detector.scan()
        print(f"    - Severity: {report.severity.name}")
        print(f"    - Total deviation: {report.total_deviation:.2%}")
        print(f"    - Worst lane: {report.worst_lane}")
        if report.pattern_matches:
            for pattern, sim in report.pattern_matches[:2]:
                print(f"    - Pattern match: {pattern.name} ({sim:.2%})")
        print(f"    - Summary: {report.summary}")
        print()

        # Let it recover
        for _ in range(10):
            telemetry = mux.sample_all()
            field.update_all(telemetry)
            field.compute_field()

    # Phase 4: Multi-lane cascade
    print("[5/5] Simulating cascade failure (all systems)...")
    mux.inject_anomaly("gpu", "util_saturation", duration=10)
    mux.inject_anomaly("network", "ddos", duration=10)
    mux.inject_anomaly("service", "overload", duration=10)

    for i in range(10):
        telemetry = mux.sample_all()
        field.update_all(telemetry)
        field.compute_field()

        if i % 3 == 0:
            report = detector.scan()
            print(f"  - Tick {i}: {report.severity.name} ({report.total_deviation:.2%})")

    # Final report
    print()
    report = detector.scan()
    print("Final cascade report:")
    print(f"  - Severity: {report.severity.name}")
    print(f"  - Total deviation: {report.total_deviation:.2%}")
    print(f"  - Lane deviations:")
    for lane, dev in report.lane_deviations.items():
        print(f"      {lane}: {dev:.2%}")
    print(f"  - Summary: {report.summary}")
    print()

    # Show resonance query
    print("=" * 60)
    print("Resonance Query Demo")
    print("=" * 60)
    print()

    # Learn current bad state as a pattern
    cascade_pattern = detector.create_pattern(
        name="observed_cascade",
        description="The cascade failure we just observed",
        severity=AnomalySeverity.CRITICAL,
        threshold=0.4,
    )
    print(f"Learned pattern: '{cascade_pattern.name}'")

    # Check if it matches (should match strongly since we just created it)
    resonance = field.query_resonance(cascade_pattern.pattern_hv)
    print(f"Resonance with current field: {resonance:.2%}")

    # Now recover and check again
    print()
    print("Recovering (50 normal samples)...")
    for _ in range(50):
        telemetry = mux.sample_all()
        field.update_all(telemetry)
        field.compute_field()

    # Check resonance again
    resonance = field.query_resonance(cascade_pattern.pattern_hv)
    print(f"Resonance after recovery: {resonance:.2%}")

    # Also check stream (temporal memory)
    stream_resonance = field.query_stream_resonance(cascade_pattern.pattern_hv)
    print(f"Stream resonance (temporal memory): {stream_resonance:.2%}")

    severity, dev = detector.quick_check()
    print(f"Final state: {severity.name} ({dev:.2%} deviation)")
    print()
    print("Demo complete!")


def benchmark():
    """Quick performance benchmark."""
    print("HSF Performance Benchmark")
    print("-" * 40)

    field, _ = setup_field_and_lanes()
    mux = TelemetryMux.create_default()
    detector = AnomalyDetector(field)

    # Warmup
    for _ in range(10):
        telemetry = mux.sample_all()
        field.update_all(telemetry)
        field.compute_field()

    # Benchmark
    n_samples = 1000
    start = time.time()

    for _ in range(n_samples):
        telemetry = mux.sample_all()
        field.update_all(telemetry)
        field.compute_field()
        _ = detector.quick_check()

    elapsed = time.time() - start
    rate = n_samples / elapsed

    print(f"Processed {n_samples} samples in {elapsed:.2f}s")
    print(f"Rate: {rate:.0f} samples/sec")
    print(f"Latency: {1000/rate:.2f}ms per sample")
    print()

    # Field dimension scaling
    print("Dimension scaling:")
    for dim in [1024, 4096, 8192, 16384]:
        field = HSField(dim=dim)
        field.add_lane_config("test", ["a", "b", "c"])

        start = time.time()
        for _ in range(100):
            field.update("test", {"a": 0.5, "b": 0.3, "c": 0.7})
            field.compute_field()
        elapsed = time.time() - start

        print(f"  dim={dim}: {100/elapsed:.0f} samples/sec")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark()
    else:
        run_demo()
