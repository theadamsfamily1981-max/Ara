#!/usr/bin/env python3
"""
run_gui.py

Launcher script for the FPGA ML Bitstream Analysis GUI.

Usage:
    python run_gui.py

Or make executable:
    chmod +x run_gui.py
    ./run_gui.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import PyQt5
    except ImportError:
        missing.append("PyQt5")

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import torch
    except ImportError:
        missing.append("torch")

    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nInstall with:")
        print(f"   pip install {' '.join(missing)}")
        print("\nOr install all dependencies:")
        print("   pip install -r requirements.txt")
        return False

    return True


def main():
    """Main entry point."""
    print("=" * 60)
    print("FPGA ML Bitstream Analysis Toolkit - GUI")
    print("=" * 60)
    print()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    print("✅ All dependencies found")
    print("🚀 Launching GUI...")
    print()

    # Import and run GUI
    from gui.main_window import main as gui_main
    gui_main()


if __name__ == "__main__":
    main()
