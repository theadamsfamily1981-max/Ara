"""
Encode tabular data (DataFrames) into hypervectors.

This enables "geometric" operations on structured data:
- Similarity between rows
- Clustering in HD space
- Pattern matching via binding/bundling
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from .hypervector import random_hv, bind, bundle, DIM


# Cache for column and value hypervectors
_col_hv_cache: Dict[str, np.ndarray] = {}
_val_hv_cache: Dict[str, np.ndarray] = {}


def _get_col_hv(col_name: str) -> np.ndarray:
    """Get or create hypervector for a column name."""
    if col_name not in _col_hv_cache:
        seed = hash(f"col:{col_name}") % (2**31 - 1)
        _col_hv_cache[col_name] = random_hv(seed)
    return _col_hv_cache[col_name]


def _get_val_hv(col_name: str, value: Any) -> np.ndarray:
    """Get or create hypervector for a column-value pair."""
    key = f"{col_name}={value}"
    if key not in _val_hv_cache:
        seed = hash(key) % (2**31 - 1)
        _val_hv_cache[key] = random_hv(seed)
    return _val_hv_cache[key]


def encode_value(col_name: str, value: Any) -> np.ndarray:
    """
    Encode a single column-value pair as a hypervector.

    The encoding binds the column HV with the value HV:
    hv = bind(col_hv, val_hv)

    Args:
        col_name: Column name
        value: Value to encode

    Returns:
        Hypervector representing this column-value pair
    """
    col_hv = _get_col_hv(col_name)
    val_hv = _get_val_hv(col_name, value)
    return bind(col_hv, val_hv)


def encode_row(row: pd.Series) -> np.ndarray:
    """
    Encode a single row as a hypervector.

    The row HV is the bundle of all column-value pair HVs.

    Args:
        row: Pandas Series representing a row

    Returns:
        Hypervector representing the entire row
    """
    pair_hvs = []
    for col, val in row.items():
        if pd.notna(val):
            pair_hvs.append(encode_value(col, val))

    if not pair_hvs:
        return np.zeros(DIM, dtype=np.int8)

    return bundle(pair_hvs)


def encode_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Encode an entire DataFrame into hypervectors.

    Returns:
        Dictionary with:
        - 'rows': Dict[index, np.ndarray] - HV for each row
        - 'cols': Dict[col_name, np.ndarray] - HV for each column
        - 'shape': Tuple of (n_rows, n_cols)
    """
    col_hvs = {col: _get_col_hv(col) for col in df.columns}
    row_hvs = {}

    for idx, row in df.iterrows():
        row_hvs[idx] = encode_row(row)

    return {
        "rows": row_hvs,
        "cols": col_hvs,
        "shape": df.shape,
    }


def find_similar_rows(
    target_hv: np.ndarray,
    row_hvs: Dict[Any, np.ndarray],
    top_k: int = 5,
) -> list:
    """
    Find rows most similar to a target hypervector.

    Args:
        target_hv: Query hypervector
        row_hvs: Dictionary of row index -> row HV
        top_k: Number of results to return

    Returns:
        List of (index, similarity) tuples, sorted by similarity
    """
    from .hypervector import similarity

    similarities = []
    for idx, row_hv in row_hvs.items():
        sim = similarity(target_hv, row_hv)
        similarities.append((idx, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def clear_cache():
    """Clear the hypervector caches."""
    global _col_hv_cache, _val_hv_cache
    _col_hv_cache = {}
    _val_hv_cache = {}
