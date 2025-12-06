"""
Ara Engineer - Organism Bootstrap & Diagnostics.

This module provides infrastructure for:
- Diagnosing system dependencies (ara_doctor.py)
- Bootstrapping the three-layer architecture (bootstrap_organism.sh)
- Managing Docker containers for the Brain
- Coordinating Host and Container communication via HAL

Architecture:
    Visual Cortex (Host venv)    <-- GTK/WebKit, direct display access
            |
            | HAL (/dev/shm/ara_somatic)
            |
    Brain (Docker container)     <-- PyTorch/CUDA, isolated env
            |
            | HAL (/dev/shm/ara_somatic mounted)
            |
    Nervous System (Host root)   <-- FPGA/PCIe, hardware access
"""

from pathlib import Path

__version__ = "1.0.0"

ENGINEER_DIR = Path(__file__).parent
BANOS_DIR = ENGINEER_DIR.parent
ARA_ROOT = BANOS_DIR.parent
