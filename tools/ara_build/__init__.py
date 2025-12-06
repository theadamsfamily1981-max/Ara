"""
Ara Build Tools - Immune System for Build Failures.

This package provides:
- ara-build: Build command wrapper with failure tracking
- ara-doctor: System health scanner

The tools learn from every failure and suggest fixes based on past experience.
"""

from pathlib import Path

__version__ = "1.0.0"
__all__ = ["ara_build", "ara_doctor"]

# Package paths
PACKAGE_DIR = Path(__file__).parent
LOG_DIR = Path.home() / ".ara" / "build_logs"
PATTERN_FILE = Path.home() / ".ara" / "build_patterns.json"
