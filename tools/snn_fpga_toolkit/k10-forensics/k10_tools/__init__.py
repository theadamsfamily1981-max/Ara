"""
K10 / ColEngine P2 Firmware Analysis Toolkit

Tools for analyzing Superscalar K10 / ColEngine P2 miner firmware:
- SD card image forensics
- Firmware package extraction
- FPGA bitstream identification
"""

__version__ = "1.0.0"
__author__ = "FPGA Forensics Team"

from .sd_image_analyzer import analyze_sd_image, mount_partition
from .firmware_extractor import extract_firmware_zip, classify_firmware_files
from .bitstream_finder import find_bitstreams, classify_binary

__all__ = [
    'analyze_sd_image',
    'mount_partition',
    'extract_firmware_zip',
    'classify_firmware_files',
    'find_bitstreams',
    'classify_binary',
]
