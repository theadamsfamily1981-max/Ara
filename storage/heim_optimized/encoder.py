"""
Heim Encoder - 16k ⇄ 173 Compression
=====================================

The encoder transforms between the mythic 16k-dimensional soul
and the physical 173-bit compressed representation.

Key Operations:
    heim_compress(h_full) → h_173: Project and sparsify
    heim_decompress(h_173, delta) → h_approx: Reconstruct for rerank
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from .config import HEIM_CONFIG, PROJECTION_SEED


# =============================================================================
# Projection Matrix (Lazy Initialization)
# =============================================================================

_PROJECTION_W: Optional[np.ndarray] = None
_DECODER_W: Optional[np.ndarray] = None


def _get_projection_matrix() -> np.ndarray:
    """
    Get or create the random projection matrix W: (D_full → D_compressed).

    Uses a fixed seed for reproducibility across sessions.
    """
    global _PROJECTION_W

    if _PROJECTION_W is None:
        rng = np.random.default_rng(seed=PROJECTION_SEED)
        D_full = HEIM_CONFIG.D_full
        D_comp = HEIM_CONFIG.D_compressed

        # Random Gaussian projection (normalized)
        _PROJECTION_W = rng.standard_normal((D_comp, D_full)).astype(np.float32)
        _PROJECTION_W /= np.sqrt(D_full)  # Scale for unit variance preservation

    return _PROJECTION_W


def _get_decoder_matrix() -> np.ndarray:
    """
    Get or create the decoder matrix (pseudo-inverse of projection).

    For reconstruction: h_approx = DECODER @ h_compressed
    """
    global _DECODER_W

    if _DECODER_W is None:
        W = _get_projection_matrix()
        # Moore-Penrose pseudo-inverse
        _DECODER_W = np.linalg.pinv(W).astype(np.float32)

    return _DECODER_W


# =============================================================================
# Core Encoding
# =============================================================================

def sparse_binary_encode(h_projected: np.ndarray) -> np.ndarray:
    """
    Apply sparsity and binarization to projected HV.

    Args:
        h_projected: D_compressed dimensional real vector

    Returns:
        Sparse binary HV (uint8 array of {0,1})
    """
    D = len(h_projected)
    sparsity = HEIM_CONFIG.sparsity

    # Number of active bits (top k by magnitude)
    k = int(D * (1.0 - sparsity))

    if k <= 0:
        return np.zeros(D, dtype=np.uint8)

    # Find top-k indices by absolute magnitude
    idx = np.argpartition(-np.abs(h_projected), k)[:k]

    # Create sparse binary HV
    hv = np.zeros(D, dtype=np.uint8)
    hv[idx] = (h_projected[idx] > 0).astype(np.uint8)  # Sign → {0,1}

    return hv


def heim_compress(h_full: np.ndarray) -> np.ndarray:
    """
    Compress a full 16k HV to D=173 sparse binary.

    Args:
        h_full: (D_full,) array, bipolar {-1,+1} or real

    Returns:
        (D_compressed,) uint8 array of {0,1}
    """
    W = _get_projection_matrix()

    # Project to reduced dimension
    h_projected = W @ h_full.astype(np.float32)

    # Sparsify and binarize
    h_compressed = sparse_binary_encode(h_projected)

    return h_compressed


def heim_decompress(
    h_compressed: np.ndarray,
    delta_info: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Decompress a 173-bit HV back to 16k approximation.

    This is used during Stage 2 reranking when we need full-precision
    similarity computation.

    Args:
        h_compressed: (D_compressed,) uint8 {0,1}
        delta_info: Optional small correction (sparse indices + signs)

    Returns:
        (D_full,) float32 bipolar approximation
    """
    DECODER = _get_decoder_matrix()

    # Convert {0,1} to bipolar {-1,+1} for reconstruction
    h_bipolar = 2.0 * h_compressed.astype(np.float32) - 1.0

    # Reconstruct via pseudo-inverse
    h_approx = DECODER @ h_bipolar

    # Apply delta correction if provided
    if delta_info is not None:
        h_approx = _apply_delta(h_approx, delta_info)

    # Normalize to bipolar
    return np.sign(h_approx).astype(np.float32)


def _apply_delta(h_approx: np.ndarray, delta_info: np.ndarray) -> np.ndarray:
    """
    Apply small delta correction to reconstructed HV.

    Delta format: packed array of (index, sign) pairs
    """
    # Delta is small - typically <10 corrections
    if len(delta_info) == 0:
        return h_approx

    # Unpack and apply
    n_corrections = len(delta_info) // 2
    for i in range(n_corrections):
        idx = int(delta_info[2*i])
        sign = delta_info[2*i + 1]
        if 0 <= idx < len(h_approx):
            h_approx[idx] = sign

    return h_approx


# =============================================================================
# Similarity Functions
# =============================================================================

def hv_hamming_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Hamming similarity between two binary HVs.

    Returns fraction of matching bits ∈ [0, 1].
    """
    matches = np.count_nonzero(a == b)
    return matches / len(a)


def hv_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two HVs.

    Works for both binary {0,1} and bipolar {-1,+1}.
    """
    # Convert to float
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)

    # Handle binary → bipolar
    if np.all((a == 0) | (a == 1)):
        a_f = 2.0 * a_f - 1.0
    if np.all((b == 0) | (b == 1)):
        b_f = 2.0 * b_f - 1.0

    # Cosine
    dot = np.dot(a_f, b_f)
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)

    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0

    return float(dot / (norm_a * norm_b))


def hv_xnor_popcount(a: np.ndarray, b: np.ndarray) -> int:
    """
    XNOR + popcount for binary HVs.

    This is what the FPGA computes in hardware.
    Returns count of matching bits.
    """
    # For binary {0,1}: match = ~(a XOR b) = (a == b)
    return int(np.count_nonzero(a == b))


# =============================================================================
# Batch Operations
# =============================================================================

def batch_compress(hvs_full: np.ndarray) -> np.ndarray:
    """
    Compress multiple HVs efficiently.

    Args:
        hvs_full: (N, D_full) array

    Returns:
        (N, D_compressed) uint8 array
    """
    W = _get_projection_matrix()
    D_comp = HEIM_CONFIG.D_compressed
    N = hvs_full.shape[0]

    # Batch projection
    projected = hvs_full.astype(np.float32) @ W.T

    # Batch sparsify + binarize
    result = np.zeros((N, D_comp), dtype=np.uint8)
    k = int(D_comp * (1.0 - HEIM_CONFIG.sparsity))

    for i in range(N):
        idx = np.argpartition(-np.abs(projected[i]), k)[:k]
        result[i, idx] = (projected[i, idx] > 0).astype(np.uint8)

    return result


def batch_decompress(hvs_compressed: np.ndarray) -> np.ndarray:
    """
    Decompress multiple HVs efficiently.

    Args:
        hvs_compressed: (N, D_compressed) uint8 array

    Returns:
        (N, D_full) float32 array
    """
    DECODER = _get_decoder_matrix()

    # Convert to bipolar
    hvs_bipolar = 2.0 * hvs_compressed.astype(np.float32) - 1.0

    # Batch reconstruction
    hvs_approx = hvs_bipolar @ DECODER.T

    # Normalize to bipolar
    return np.sign(hvs_approx).astype(np.float32)


# =============================================================================
# Validation Helpers
# =============================================================================

def compression_fidelity(h_full: np.ndarray) -> Tuple[float, float]:
    """
    Measure compression quality for a single HV.

    Returns:
        (reconstruction_sim, active_bit_ratio)
    """
    h_comp = heim_compress(h_full)
    h_recon = heim_decompress(h_comp)

    # Similarity between original and reconstructed
    sim = hv_cosine_sim(h_full, h_recon)

    # Sparsity check
    active_ratio = np.count_nonzero(h_comp) / len(h_comp)

    return sim, active_ratio


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'heim_compress',
    'heim_decompress',
    'sparse_binary_encode',
    'hv_hamming_sim',
    'hv_cosine_sim',
    'hv_xnor_popcount',
    'batch_compress',
    'batch_decompress',
    'compression_fidelity',
]
