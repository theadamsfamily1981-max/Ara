#!/usr/bin/env python3
"""
Parse Quartus .sta.rpt timing report and extract key metrics

This tool extracts timing analysis results from Quartus TimeQuest reports
and outputs structured JSON for downstream tools.

Usage:
    python parse_quartus_timing.py output_files/project.sta.rpt

Output:
    JSON with clocks, Fmax, slack, and timing violations

Author: Quanta Hardware Project
License: MIT
"""

import sys
import re
import json
from pathlib import Path


def parse_timing_report(rpt_path):
    """Extract timing metrics from Quartus .sta.rpt"""
    with open(rpt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    result = {
        "report_file": str(rpt_path),
        "clocks": [],
        "worst_setup_slack_ns": None,
        "worst_hold_slack_ns": None,
        "setup_violations": 0,
        "hold_violations": 0,
        "failing_paths": 0,
        "timing_met": True
    }

    # Parse clock summary
    # Pattern for Quartus clock tables:
    # Clock Name | Actual Fmax | Requirement | Slack
    clock_table_pattern = r';\s*(\S+)\s*;\s*([\d.]+)\s+MHz\s*;\s*([\d.]+)\s+MHz\s*;\s*([-\d.]+)\s+ns'

    for match in re.finditer(clock_table_pattern, content):
        clk_name = match.group(1)
        fmax_mhz = float(match.group(2))
        req_mhz = float(match.group(3))
        slack_ns = float(match.group(4))

        result["clocks"].append({
            "name": clk_name,
            "fmax_mhz": fmax_mhz,
            "requirement_mhz": req_mhz,
            "slack_ns": slack_ns,
            "margin_mhz": fmax_mhz - req_mhz,
            "timing_met": slack_ns >= 0
        })

        # Update worst slack
        if result["worst_setup_slack_ns"] is None or slack_ns < result["worst_setup_slack_ns"]:
            result["worst_setup_slack_ns"] = slack_ns
            if slack_ns < 0:
                result["setup_violations"] += 1
                result["timing_met"] = False

    # Alternative pattern: Look for explicit worst-case slack statements
    setup_slack_patterns = [
        r'Worst-case\s+setup\s+slack\s+is\s+([-\d.]+)\s*ns',
        r'Worst-case\s+slack\s+\(setup\)\s*:\s*([-\d.]+)\s*ns',
        r'Setup\s+slack\s*:\s*([-\d.]+)\s*ns'
    ]

    for pattern in setup_slack_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            slack_val = float(match.group(1))
            if result["worst_setup_slack_ns"] is None:
                result["worst_setup_slack_ns"] = slack_val
            if slack_val < 0:
                result["timing_met"] = False
            break

    # Parse hold slack
    hold_slack_patterns = [
        r'Worst-case\s+hold\s+slack\s+is\s+([-\d.]+)\s*ns',
        r'Worst-case\s+slack\s+\(hold\)\s*:\s*([-\d.]+)\s*ns',
        r'Hold\s+slack\s*:\s*([-\d.]+)\s*ns'
    ]

    for pattern in hold_slack_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            slack_val = float(match.group(1))
            result["worst_hold_slack_ns"] = slack_val
            if slack_val < 0:
                result["hold_violations"] += 1
                result["timing_met"] = False
            break

    # Count failing paths
    failing_paths_match = re.search(r'Failing\s+Paths\s*:\s*(\d+)', content, re.IGNORECASE)
    if failing_paths_match:
        result["failing_paths"] = int(failing_paths_match.group(1))

    # Summary determination
    if result["worst_setup_slack_ns"] is not None and result["worst_setup_slack_ns"] < 0:
        result["timing_met"] = False
    if result["worst_hold_slack_ns"] is not None and result["worst_hold_slack_ns"] < 0:
        result["timing_met"] = False

    return result


def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: parse_quartus_timing.py <sta_rpt_file>", file=sys.stderr)
        print("", file=sys.stderr)
        print("Example:", file=sys.stderr)
        print("  python parse_quartus_timing.py output_files/project.sta.rpt", file=sys.stderr)
        sys.exit(1)

    rpt_file = Path(sys.argv[1])

    if not rpt_file.exists():
        print(f"❌ Error: Report file not found: {rpt_file}", file=sys.stderr)
        sys.exit(1)

    try:
        metrics = parse_timing_report(rpt_file)
        print(json.dumps(metrics, indent=2))

        # Exit with error code if timing not met
        sys.exit(0 if metrics["timing_met"] else 1)

    except Exception as e:
        print(f"❌ Error parsing timing report: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
