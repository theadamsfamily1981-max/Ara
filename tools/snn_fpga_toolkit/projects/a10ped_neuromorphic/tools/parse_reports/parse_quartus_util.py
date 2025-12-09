#!/usr/bin/env python3
"""
Parse Quartus .fit.summary for resource utilization

This tool extracts resource utilization metrics from Quartus Fitter reports
and outputs structured JSON for downstream tools.

Usage:
    python parse_quartus_util.py output_files/project.fit.summary

Output:
    JSON with ALMs, registers, memory blocks, DSPs, and other resources

Author: Quanta Hardware Project
License: MIT
"""

import sys
import re
import json
from pathlib import Path


def parse_utilization(summary_path):
    """Extract resource usage from Quartus .fit.summary"""
    with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    result = {
        "report_file": str(summary_path),
        "alms": {"used": 0, "available": 0, "percent": 0.0},
        "registers": {"used": 0, "available": 0, "percent": 0.0},
        "memory_bits": {"used": 0, "available": 0, "percent": 0.0},
        "memory_blocks": {"used": 0, "available": 0, "percent": 0.0},
        "dsps": {"used": 0, "available": 0, "percent": 0.0},
        "pins": {"used": 0, "available": 0, "percent": 0.0}
    }

    # Quartus fit.summary patterns (multiple variants for robustness)
    # Example format: "ALMs needed                   : 12,345 / 427,200 ( 3 % )"
    patterns = {
        "alms": [
            r'ALM.*needed\s*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*([\d.]+)\s*%\s*\)',
            r'Logic\s+utilization.*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*([\d.]+)\s*%\s*\)'
        ],
        "registers": [
            r'Total\s+registers\s*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*([\d.]+)\s*%\s*\)',
            r'Dedicated\s+logic\s+registers\s*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*([\d.]+)\s*%\s*\)'
        ],
        "memory_bits": [
            r'Total\s+block\s+memory\s+bits\s*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*([\d.]+)\s*%\s*\)',
            r'M20K\s+blocks.*bits\s*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*([\d.]+)\s*%\s*\)'
        ],
        "memory_blocks": [
            r'Total\s+block\s+memory.*blocks\s*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*([\d.]+)\s*%\s*\)',
            r'M20K\s+blocks\s*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*([\d.]+)\s*%\s*\)'
        ],
        "dsps": [
            r'DSP\s+blocks\s*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*([\d.]+)\s*%\s*\)',
            r'Variable-precision\s+multipliers\s*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*([\d.]+)\s*%\s*\)'
        ],
        "pins": [
            r'Pins\s*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*([\d.]+)\s*%\s*\)',
            r'I\/O\s+pins\s*:\s*([\d,]+)\s*/\s*([\d,]+)\s*\(\s*([\d.]+)\s*%\s*\)'
        ]
    }

    # Try each pattern for each resource type
    for resource_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                result[resource_type] = {
                    "used": int(match.group(1).replace(',', '')),
                    "available": int(match.group(2).replace(',', '')),
                    "percent": float(match.group(3))
                }
                break  # Found a match, move to next resource type

    return result


def format_utilization_table(metrics):
    """Format utilization as a readable table"""
    lines = []
    lines.append("")
    lines.append("Resource Utilization Summary")
    lines.append("=" * 60)
    lines.append(f"{'Resource':<20} {'Used':>12} {'Available':>12} {'%':>6}")
    lines.append("-" * 60)

    for resource_name, data in metrics.items():
        if resource_name == "report_file":
            continue

        name = resource_name.replace('_', ' ').title()
        used = f"{data['used']:,}"
        avail = f"{data['available']:,}"
        pct = f"{data['percent']:.1f}%"

        lines.append(f"{name:<20} {used:>12} {avail:>12} {pct:>6}")

    lines.append("=" * 60)
    return "\n".join(lines)


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: parse_quartus_util.py <fit_summary_file>", file=sys.stderr)
        print("", file=sys.stderr)
        print("Example:", file=sys.stderr)
        print("  python parse_quartus_util.py output_files/project.fit.summary", file=sys.stderr)
        sys.exit(1)

    summary_file = Path(sys.argv[1])

    if not summary_file.exists():
        print(f"❌ Error: Summary file not found: {summary_file}", file=sys.stderr)
        sys.exit(1)

    try:
        metrics = parse_utilization(summary_file)

        # Output JSON
        print(json.dumps(metrics, indent=2))

        # Also print a human-readable table to stderr
        print(format_utilization_table(metrics), file=sys.stderr)

    except Exception as e:
        print(f"❌ Error parsing utilization report: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
