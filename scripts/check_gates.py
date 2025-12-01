#!/usr/bin/env python
"""
Validate all TF-A-N hard gates.

Checks benchmark results, topology audits, and system metrics
against defined thresholds.

Usage:
    python scripts/check_gates.py --all-gates
    python scripts/check_gates.py --bench artifacts/bench --memory artifacts/memory
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


class GateValidator:
    """Validates TF-A-N hard gates."""

    def __init__(self):
        self.gates = {
            "ttw_p95_latency": {"threshold": 5.0, "unit": "ms", "operator": "<"},
            "ttw_coverage": {"threshold": 0.90, "unit": "%", "operator": ">="},
            "pgu_p95_latency": {"threshold": 200.0, "unit": "ms", "operator": "<="},
            "pgu_cache_hit": {"threshold": 0.50, "unit": "%", "operator": ">="},
            "ssa_speedup_16k": {"threshold": 3.0, "unit": "x", "operator": ">="},
            "ssa_speedup_32k": {"threshold": 3.0, "unit": "x", "operator": ">="},
            "ssa_accuracy_delta": {"threshold": 0.02, "unit": "%", "operator": "<="},
            "epr_cv": {"threshold": 0.15, "unit": "", "operator": "<="},
            "topology_wasserstein": {"threshold": 0.02, "unit": "", "operator": "<="},
            "topology_cosine": {"threshold": 0.90, "unit": "", "operator": ">="},
            "memory_alpha": {"threshold": 1.0, "unit": "", "operator": "<"},
        }

        self.results = {}
        self.violations = []

    def check_gate(self, name: str, value: float) -> bool:
        """Check a single gate."""
        if name not in self.gates:
            print(f"Warning: Unknown gate '{name}'")
            return True

        gate = self.gates[name]
        threshold = gate["threshold"]
        operator = gate["operator"]

        if operator == "<":
            passes = value < threshold
        elif operator == "<=":
            passes = value <= threshold
        elif operator == ">":
            passes = value > threshold
        elif operator == ">=":
            passes = value >= threshold
        else:
            passes = value == threshold

        self.results[name] = {
            "value": value,
            "threshold": threshold,
            "passes": passes,
            "unit": gate["unit"],
            "operator": operator,
        }

        if not passes:
            self.violations.append(name)

        return passes

    def validate_attention_bench(self, bench_path: str) -> bool:
        """Validate attention benchmark results."""
        print(f"\nValidating attention benchmark: {bench_path}")

        with open(bench_path) as f:
            data = json.load(f)

        all_pass = True

        # Check speedups at 16k and 32k
        for i, seq_len in enumerate(data["seq_lengths"]):
            speedup = data["speedups"][i]

            if seq_len == 16384:
                passes = self.check_gate("ssa_speedup_16k", speedup)
                print(f"  SSA speedup @ 16k: {speedup:.2f}× {'✓' if passes else '✗ FAIL'}")
                all_pass = all_pass and passes

            elif seq_len == 32768:
                passes = self.check_gate("ssa_speedup_32k", speedup)
                print(f"  SSA speedup @ 32k: {speedup:.2f}× {'✓' if passes else '✗ FAIL'}")
                all_pass = all_pass and passes

        return all_pass

    def validate_memory_fit(self, memory_path: str) -> bool:
        """Validate memory scaling."""
        print(f"\nValidating memory scaling: {memory_path}")

        with open(memory_path) as f:
            data = json.load(f)

        alpha = data.get("alpha", float('inf'))
        passes = self.check_gate("memory_alpha", alpha)

        print(f"  Memory α: {alpha:.3f} {'✓' if passes else '✗ FAIL'}")

        return passes

    def validate_topology_audit(self, audit_path: str) -> bool:
        """Validate topology audit."""
        print(f"\nValidating topology audit: {audit_path}")

        with open(audit_path) as f:
            data = json.load(f)

        metrics = data.get("metrics", {})
        wass = metrics.get("avg_wasserstein", float('inf'))
        cos = metrics.get("avg_cosine", 0.0)

        wass_pass = self.check_gate("topology_wasserstein", wass)
        cos_pass = self.check_gate("topology_cosine", cos)

        print(f"  Wasserstein: {wass:.4f} {'✓' if wass_pass else '✗ FAIL'}")
        print(f"  Cosine: {cos:.4f} {'✓' if cos_pass else '✗ FAIL'}")

        return wass_pass and cos_pass

    def validate_pgu_bench(self, pgu_path: str) -> bool:
        """Validate PGU benchmark."""
        print(f"\nValidating PGU benchmark: {pgu_path}")

        with open(pgu_path) as f:
            data = json.load(f)

        p95 = data.get("p95_latency_ms", float('inf'))
        hit_rate = data.get("cache_hit_rate", 0.0)

        p95_pass = self.check_gate("pgu_p95_latency", p95)
        hit_pass = self.check_gate("pgu_cache_hit", hit_rate)

        print(f"  PGU p95: {p95:.1f}ms {'✓' if p95_pass else '✗ FAIL'}")
        print(f"  Cache hit: {hit_rate:.1%} {'✓' if hit_pass else '✗ FAIL'}")

        return p95_pass and hit_pass

    def validate_ttw_bench(self, ttw_path: str) -> bool:
        """Validate TTW benchmark."""
        print(f"\nValidating TTW benchmark: {ttw_path}")

        with open(ttw_path) as f:
            data = json.load(f)

        p95 = data.get("p95_latency_ms", float('inf'))
        coverage = data.get("coverage", 0.0)

        p95_pass = self.check_gate("ttw_p95_latency", p95)
        cov_pass = self.check_gate("ttw_coverage", coverage)

        print(f"  TTW p95: {p95:.2f}ms {'✓' if p95_pass else '✗ FAIL'}")
        print(f"  Coverage: {coverage:.1%} {'✓' if cov_pass else '✗ FAIL'}")

        return p95_pass and cov_pass

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 80)
        print("GATE VALIDATION SUMMARY")
        print("=" * 80)

        for name, result in self.results.items():
            value = result["value"]
            threshold = result["threshold"]
            unit = result["unit"]
            operator = result["operator"]
            passes = result["passes"]

            status = "✓ PASS" if passes else "✗ FAIL"
            print(f"{name:25s}: {value:8.3f} {operator} {threshold:.3f} {unit:5s} {status}")

        print("-" * 80)
        total = len(self.results)
        passed = total - len(self.violations)
        print(f"Total: {passed}/{total} gates passing")

        if self.violations:
            print(f"\nViolations:")
            for v in self.violations:
                print(f"  - {v}")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Validate TF-A-N hard gates")
    parser.add_argument("--bench", type=str, help="Benchmark directory")
    parser.add_argument("--memory", type=str, help="Memory fit directory")
    parser.add_argument("--topology", type=str, help="Topology audit file")
    parser.add_argument("--all-gates", action="store_true", help="Validate all available gates")
    parser.add_argument("--weekly", action="store_true", help="Weekly validation mode")
    args = parser.parse_args()

    validator = GateValidator()

    print("=" * 80)
    print("TF-A-N Gate Validation")
    print("=" * 80)

    all_pass = True

    # Attention benchmarks
    if args.bench:
        bench_dir = Path(args.bench)
        attention_files = list(bench_dir.glob("attention*.json"))
        if attention_files:
            latest = max(attention_files, key=lambda p: p.stat().st_mtime)
            all_pass = validator.validate_attention_bench(str(latest)) and all_pass
        else:
            print(f"Warning: No attention benchmark found in {args.bench}")

        # PGU benchmarks
        pgu_files = list(bench_dir.glob("pgu*.json"))
        if pgu_files:
            latest = max(pgu_files, key=lambda p: p.stat().st_mtime)
            all_pass = validator.validate_pgu_bench(str(latest)) and all_pass

        # TTW benchmarks
        ttw_files = list(bench_dir.glob("ttw*.json"))
        if ttw_files:
            latest = max(ttw_files, key=lambda p: p.stat().st_mtime)
            all_pass = validator.validate_ttw_bench(str(latest)) and all_pass

    # Memory fit
    if args.memory:
        memory_dir = Path(args.memory)
        fit_files = list(memory_dir.glob("fit*.json"))
        if fit_files:
            latest = max(fit_files, key=lambda p: p.stat().st_mtime)
            all_pass = validator.validate_memory_fit(str(latest)) and all_pass
        else:
            print(f"Warning: No memory fit found in {args.memory}")

    # Topology audit
    if args.topology:
        all_pass = validator.validate_topology_audit(args.topology) and all_pass

    # Print summary
    validator.print_summary()

    if all_pass:
        print("\n✓ All gates PASS")
        return 0
    else:
        print(f"\n✗ {len(validator.violations)} gate(s) FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
