"""
Hypervector encoding module.

Lightweight HDC for encoding tabular data into high-dimensional vectors.
"""

from .hypervector import (
    DIM,
    random_hv,
    bind,
    bundle,
    similarity,
    normalize,
)
from .encode_tabular import encode_dataframe, encode_value

__all__ = [
    "DIM",
    "random_hv",
    "bind",
    "bundle",
    "similarity",
    "normalize",
    "encode_dataframe",
    "encode_value",
]
