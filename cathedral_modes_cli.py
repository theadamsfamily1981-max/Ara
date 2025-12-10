#!/usr/bin/env python3
"""
Cathedral Modes CLI

Quick-and-dirty tool to inspect Cathedral OS "fandom modes"
and the invariants each one promises.

Usage:
    python cathedral_modes_cli.py             # list modes
    python cathedral_modes_cli.py starfleet   # show one mode
"""

import sys
from pathlib import Path
import textwrap

try:
    import yaml
except ImportError:
    yaml = None


def load_modes(config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find {config_path}")
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Try: pip install pyyaml")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("modes", {})


def print_mode(name: str, mode: dict):
    print(f"\n=== {name} ===")
    print(f"  Fandom : {mode.get('fandom')}")
    print(f"  Theme  : {mode.get('theme')}")
    invariants = mode.get("invariants", {})
    print("  Invariants:")
    for k, v in invariants.items():
        print(f"    - {k}: {v}")
    notes = mode.get("invariants", {}).get("notes") or mode.get("notes")
    if notes:
        print("\n  Notes:")
        print(textwrap.indent(textwrap.fill(str(notes), width=78), "    "))


def main():
    config_path = Path(__file__).parent / "config" / "cathedral_modes.yaml"
    try:
        modes = load_modes(config_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if len(sys.argv) == 1:
        # List all modes
        print("Available Cathedral OS modes:\n")
        for name, mode in modes.items():
            print(f"  - {name:14s} ({mode.get('fandom')})")
        print("\nUsage: python cathedral_modes_cli.py <mode_name>\n")
        sys.exit(0)

    mode_name = sys.argv[1]
    mode = modes.get(mode_name)
    if not mode:
        print(f"[ERROR] Unknown mode '{mode_name}'. Known modes:")
        for name in modes.keys():
            print(f"  - {name}")
        sys.exit(1)

    print_mode(mode_name, mode)


if __name__ == "__main__":
    main()
