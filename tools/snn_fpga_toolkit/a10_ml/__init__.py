"""
A10 ML Bitstream Pipeline

Arria 10 FPGA bitstream preprocessing tools for ML training.

Core functionality:
- Load Intel .sof/.rbf bitstreams
- Strip headers and extract raw configuration data
- Autocorrelation-based width detection
- Convert bitstreams to 2D images for CNN training
"""

__version__ = "1.0.0"

from .preprocess import load_bitstream, strip_intel_headers
from .width_detection import guess_width, autocorrelation_scan
from .image_encoder import bitstream_to_image, save_image_dataset

__all__ = [
    'load_bitstream',
    'strip_intel_headers',
    'guess_width',
    'autocorrelation_scan',
    'bitstream_to_image',
    'save_image_dataset',
]
