#!/usr/bin/env python3
"""
Generate A-Cert (Antifragility Certification) Report

Aggregates all certification artifacts into a comprehensive markdown report
for dashboard visualization and CI/CD summary.

Usage:
    python scripts/generate_acert_report.py \
        --artifacts-dir artifacts/a_cert \
        --output reports/acert_report.md
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("acert_report")


def load_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON file safely."""
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
    return None


def generate_report(artifacts_dir: Path) -> str:
    """Generate comprehensive A-Cert report."""
    lines = []

    lines.append("# A-Cert: Antifragility Certification Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.utcnow().isoformat()}Z")
    lines.append("")

    # Stage 1: Hard Gates
    lines.append("## Stage 1: Structural and Formal Audit")
    lines.append("")

    stage1_dir = artifacts_dir / "a_cert_stage1"
    all_gates = load_json_safe(stage1_dir / "all_gates.json")

    if all_gates:
        lines.append("| Gate | Value | Threshold | Status |")
        lines.append("|------|-------|-----------|--------|")

        for gate in all_gates.get("gates", []):
            status = "✅" if gate.get("passed", False) else "❌"
            value = gate.get("value", "N/A")
            if isinstance(value, float):
                value = f"{value:.4f}"
            threshold = f"{gate.get('operator', '')} {gate.get('threshold', '')}"
            lines.append(f"| {gate.get('name', 'Unknown')} | {value} | {threshold} | {status} |")

        lines.append("")
        overall = "✅ PASS" if all_gates.get("all_passed", False) else "❌ FAIL"
        lines.append(f"**Overall Stage 1:** {overall}")
    else:
        lines.append("*Stage 1 data not available*")

    lines.append("")

    # Stage 2: Performance
    lines.append("## Stage 2: Performance Certification")
    lines.append("")

    stage2_dir = artifacts_dir / "a_cert_stage2"

    pgu_bench = load_json_safe(stage2_dir / "pgu_bench.json")
    if pgu_bench:
        lines.append("### PGU Verification Latency")
        lines.append("")
        lines.append(f"- **p95 Latency:** {pgu_bench.get('p95_ms', 'N/A'):.2f}ms (target ≤120ms)")
        lines.append(f"- **Cache Hit Rate:** {pgu_bench.get('cache_hit_rate', 0):.1%} (target ≥50%)")
        lines.append("")

    cxl_bench = load_json_safe(stage2_dir / "cxl_bench.json")
    if cxl_bench:
        lines.append("### CXL Memory Latency")
        lines.append("")
        lines.append(f"- **Latency:** {cxl_bench.get('latency_us', 'N/A'):.2f}μs (target <2μs)")
        lines.append("")

    energy_bench = load_json_safe(stage2_dir / "energy_bench.json")
    if energy_bench:
        lines.append("### Energy Efficiency")
        lines.append("")
        lines.append(f"- **Savings Factor:** {energy_bench.get('savings_factor', 'N/A'):.1f}× (target 10-100×)")
        lines.append("")

    # Stage 3: Antifragility
    lines.append("## Stage 3: Antifragility Certification")
    lines.append("")

    stage3_dir = artifacts_dir / "a_cert_stage3"

    cert = load_json_safe(stage3_dir / "certification.json")
    if cert:
        lines.append("### Latency Delta (Δp99)")
        lines.append("")
        lines.append("| Condition | Baseline p99 | Adaptive p99 | Δp99 |")
        lines.append("|-----------|--------------|--------------|------|")

        baseline_normal = cert.get("baseline_normal", {})
        adaptive_normal = cert.get("adaptive_normal", {})
        delta_normal = cert.get("delta_p99_normal_ms", 0)
        lines.append(f"| Normal Load | {baseline_normal.get('p99_ms', 0):.2f}ms | {adaptive_normal.get('p99_ms', 0):.2f}ms | +{delta_normal:.2f}ms |")

        baseline_burst = cert.get("baseline_burst", {})
        adaptive_burst = cert.get("adaptive_burst", {})
        delta_burst = cert.get("delta_p99_burst_ms", 0)
        delta_pct = cert.get("delta_p99_percent", 0)
        lines.append(f"| Burst Load ({cert.get('burst_factor', 2.0)}×) | {baseline_burst.get('p99_ms', 0):.2f}ms | {adaptive_burst.get('p99_ms', 0):.2f}ms | **+{delta_burst:.2f}ms ({delta_pct:+.1f}%)** |")

        lines.append("")
        lines.append(f"### Antifragility Score: **{cert.get('antifragility_score', 0):.2f}×**")
        lines.append("")
        lines.append(f"> The adaptive system degrades {cert.get('antifragility_score', 0):.1f}× less than the baseline under stress.")
        lines.append("")

        if cert.get("certification_passed", False):
            lines.append("### ✅ ANTIFRAGILITY CERTIFIED")
            lines.append("")
            lines.append("The system demonstrates quantifiable antifragility:")
            lines.append("- Lower p99 latency under stress")
            lines.append("- Graceful degradation vs baseline")
            lines.append("- Positive Δp99 advantage")
        else:
            lines.append("### ⚠️ CERTIFICATION INCOMPLETE")

    # CLV Analysis
    clv = load_json_safe(stage3_dir / "clv_analysis.json")
    if clv:
        lines.append("")
        lines.append("### Cognitive Load Vector (CLV)")
        lines.append("")
        lines.append(f"- **Risk Level:** {clv.get('risk_level', 'N/A')}")
        lines.append(f"- **Instability:** {clv.get('instability', 0):.3f}")
        lines.append(f"- **Resource:** {clv.get('resource', 0):.3f}")
        lines.append(f"- **Structural:** {clv.get('structural', 0):.3f}")

    # Closed-loop demo
    demo = load_json_safe(stage3_dir / "closed_loop_demo.json")
    if demo:
        lines.append("")
        lines.append("### Closed-Loop Demo Results")
        lines.append("")
        lines.append(f"- **L2 Valence:** {demo.get('initial_valence', 0):.3f}")
        lines.append(f"- **L2 Arousal:** {demo.get('initial_arousal', 0):.3f}")
        lines.append(f"- **L3 Temperature Mult:** {demo.get('temperature_mult', 0):.3f}")
        lines.append(f"- **AEPO Structural Change:** {demo.get('structural_change', 0):.1%}")
        lines.append(f"- **PGU Verified:** {'✅' if demo.get('pgu_verified', False) else '❌'}")
        lines.append(f"- **Backend Selected:** `{demo.get('backend_selected', 'N/A')}`")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by A-Cert Pipeline*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate A-Cert report")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    report = generate_report(args.artifacts_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)

    print(f"Report generated: {args.output}")


if __name__ == "__main__":
    main()
