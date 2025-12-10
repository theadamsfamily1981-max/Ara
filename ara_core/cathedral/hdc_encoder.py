"""
Cathedral HDC Encoder
=====================

Event → Hypervector → Chunk encoding for the cathedral soul substrate.

Event Schema (JSONL):
    {
        "ts": "2025-12-10T03:14:15.926Z",
        "host": "threadripper-01",
        "module": "ara_voice",
        "phase": "inference",
        "chunk_id": "2025-12-10T03:14:00Z",
        "T_s": 0.943,
        "A_g": 0.017,
        "H_s": 0.978,
        "power_w": 612.3,
        "yield_per_dollar": 3.2,
        "sigma": 0.10,
        "tags": ["ara", "cathedral_os", "prod"]
    }

Usage:
    encoder = HDCEncoder()
    hv = encoder.encode_event(event_dict)
    H_chunks, meta = build_chunks(events, encoder)
"""

import numpy as np
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import json
from pathlib import Path
from datetime import datetime

# Soul substrate dimensions
DIM = 16384
RNG = np.random.default_rng(42)


def random_hv() -> np.ndarray:
    """Generate random bipolar {+1, -1} hypervector."""
    return RNG.choice([-1, 1], size=DIM, replace=True).astype(np.float32)


def normalize(hv: np.ndarray) -> np.ndarray:
    """Normalize hypervector to unit length."""
    norm = np.linalg.norm(hv)
    if norm < 1e-8:
        return hv
    return hv / norm


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two hypervectors (elementwise product)."""
    return a * b


def bundle(hvs: List[np.ndarray]) -> np.ndarray:
    """Bundle multiple hypervectors (normalized sum)."""
    if not hvs:
        return np.zeros(DIM, dtype=np.float32)
    return normalize(np.sum(hvs, axis=0))


def permute(hv: np.ndarray, shifts: int = 1) -> np.ndarray:
    """Permute hypervector (circular shift for sequence encoding)."""
    return np.roll(hv, shifts)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between hypervectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


@dataclass
class HDCEncoder:
    """
    Hyperdimensional Computing encoder for cathedral events.

    Encodes categorical and scalar features into hypervectors.
    """
    dim: int = DIM
    symbol_hv: Dict[str, np.ndarray] = field(default_factory=dict)
    scalar_hv: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)

    def _sym(self, key: str) -> np.ndarray:
        """Get or create symbol hypervector."""
        if key not in self.symbol_hv:
            # Deterministic from key
            seed = hash(key) % (2**32)
            rng = np.random.default_rng(seed)
            self.symbol_hv[key] = rng.choice([-1, 1], size=self.dim).astype(np.float32)
        return self.symbol_hv[key]

    def _scalar_bin(self, name: str, bin_id: int) -> np.ndarray:
        """Get or create scalar bin hypervector."""
        if name not in self.scalar_hv:
            self.scalar_hv[name] = {}
        if bin_id not in self.scalar_hv[name]:
            seed = hash(f"{name}_{bin_id}") % (2**32)
            rng = np.random.default_rng(seed)
            self.scalar_hv[name][bin_id] = rng.choice([-1, 1], size=self.dim).astype(np.float32)
        return self.scalar_hv[name][bin_id]

    def _bin_scalar(self, value: float, n_bins: int = 10,
                    lo: float = 0.0, hi: float = 1.0) -> int:
        """Bin a scalar value."""
        v = np.clip(value, lo, hi)
        return int(np.floor((v - lo) / (hi - lo + 1e-9) * n_bins))

    def encode_event(self, ev: Dict[str, Any]) -> np.ndarray:
        """
        Encode a cathedral event as a hypervector.

        H_e = bind(time, bind(module, bind(location, pack(metrics))))
        """
        hvs = []

        # Categorical features
        hvs.append(self._sym(f"module:{ev.get('module', 'unknown')}"))
        hvs.append(self._sym(f"phase:{ev.get('phase', 'unknown')}"))
        hvs.append(self._sym(f"host:{ev.get('host', 'unknown')}"))

        # T_s in [0, 1]
        ts = float(ev.get("T_s", 0.0))
        ts_bin = self._bin_scalar(ts, n_bins=10, lo=0.0, hi=1.0)
        hvs.append(bind(self._sym("T_s"), self._scalar_bin("T_s", ts_bin)))

        # A_g in [-0.1, +0.1] roughly
        ag = float(ev.get("A_g", 0.0))
        ag_norm = (ag + 0.1) / 0.2  # map [-0.1, 0.1] -> [0, 1]
        ag_bin = self._bin_scalar(ag_norm, n_bins=10, lo=0.0, hi=1.0)
        hvs.append(bind(self._sym("A_g"), self._scalar_bin("A_g", ag_bin)))

        # H_s in [0, 1]
        hs = float(ev.get("H_s", 0.977))
        hs_bin = self._bin_scalar(hs, n_bins=10, lo=0.0, hi=1.0)
        hvs.append(bind(self._sym("H_s"), self._scalar_bin("H_s", hs_bin)))

        # Power (assume 0-1200W)
        pw = float(ev.get("power_w", 0.0))
        pw_norm = np.clip(pw / 1200.0, 0.0, 1.0)
        pw_bin = self._bin_scalar(pw_norm, n_bins=10, lo=0.0, hi=1.0)
        hvs.append(bind(self._sym("power_w"), self._scalar_bin("power_w", pw_bin)))

        # Sigma (stress level) in [0, 0.3]
        sigma = float(ev.get("sigma", 0.10))
        sigma_norm = np.clip(sigma / 0.3, 0.0, 1.0)
        sigma_bin = self._bin_scalar(sigma_norm, n_bins=10, lo=0.0, hi=1.0)
        hvs.append(bind(self._sym("sigma"), self._scalar_bin("sigma", sigma_bin)))

        # Yield/$ in [0, 20] roughly
        ypd = float(ev.get("yield_per_dollar", 0.0))
        ypd_norm = np.clip(ypd / 20.0, 0.0, 1.0)
        ypd_bin = self._bin_scalar(ypd_norm, n_bins=10, lo=0.0, hi=1.0)
        hvs.append(bind(self._sym("yield_per_dollar"), self._scalar_bin("yield_per_dollar", ypd_bin)))

        # Optional tags
        for tag in ev.get("tags", []):
            hvs.append(self._sym(f"tag:{tag}"))

        # Time encoding (cyclical)
        if "ts" in ev:
            try:
                dt = datetime.fromisoformat(ev["ts"].replace("Z", "+00:00"))
                hour = dt.hour
                dow = dt.weekday()
                # Cyclical encoding via dedicated HVs
                hvs.append(self._sym(f"hour:{hour}"))
                hvs.append(self._sym(f"dow:{dow}"))
            except (ValueError, AttributeError):
                pass

        return bundle(hvs)


@dataclass
class ChunkStats:
    """Statistics for a time chunk."""
    chunk_id: str
    T_s_sum: float = 0.0
    A_g_sum: float = 0.0
    H_s_sum: float = 0.0
    power_sum: float = 0.0
    yield_sum: float = 0.0
    count: int = 0

    @property
    def T_s_mean(self) -> float:
        return self.T_s_sum / max(self.count, 1)

    @property
    def A_g_mean(self) -> float:
        return self.A_g_sum / max(self.count, 1)

    @property
    def H_s_mean(self) -> float:
        return self.H_s_sum / max(self.count, 1)

    @property
    def power_mean(self) -> float:
        return self.power_sum / max(self.count, 1)

    @property
    def yield_mean(self) -> float:
        return self.yield_sum / max(self.count, 1)

    def to_dict(self) -> Dict[str, float]:
        return {
            "T_s_mean": self.T_s_mean,
            "A_g_mean": self.A_g_mean,
            "H_s_mean": self.H_s_mean,
            "power_mean": self.power_mean,
            "yield_mean": self.yield_mean,
            "count": self.count,
        }


def build_chunks(
    events: List[Dict[str, Any]],
    encoder: HDCEncoder,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """
    Build chunk hypervectors from events.

    Returns:
        H_chunks: dict of chunk_id -> 16kD hypervector
        meta: dict of chunk_id -> {T_s_mean, A_g_mean, power_mean, ...}
    """
    chunk_hvs: Dict[str, List[np.ndarray]] = defaultdict(list)
    chunk_stats: Dict[str, ChunkStats] = {}

    for ev in events:
        cid = ev.get("chunk_id", "default")

        # Encode event
        hv_e = encoder.encode_event(ev)
        chunk_hvs[cid].append(hv_e)

        # Accumulate stats
        if cid not in chunk_stats:
            chunk_stats[cid] = ChunkStats(chunk_id=cid)

        cs = chunk_stats[cid]
        cs.T_s_sum += float(ev.get("T_s", 0.0))
        cs.A_g_sum += float(ev.get("A_g", 0.0))
        cs.H_s_sum += float(ev.get("H_s", 0.977))
        cs.power_sum += float(ev.get("power_w", 0.0))
        cs.yield_sum += float(ev.get("yield_per_dollar", 0.0))
        cs.count += 1

    # Bundle each chunk
    H_chunks = {}
    meta = {}

    for cid, hvs in chunk_hvs.items():
        # Sequence-encode events within chunk
        sequenced = [permute(hv, i) for i, hv in enumerate(hvs)]
        H_chunks[cid] = bundle(sequenced)
        meta[cid] = chunk_stats[cid].to_dict()

    return H_chunks, meta


def load_events_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load events from JSONL file."""
    events = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def save_chunks(
    H_chunks: Dict[str, np.ndarray],
    meta: Dict[str, Dict[str, float]],
    vec_path: str,
    meta_path: str,
):
    """Save chunks to files."""
    np.save(vec_path, H_chunks, allow_pickle=True)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def load_chunks(
    vec_path: str,
    meta_path: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """Load chunks from files."""
    H_chunks = np.load(vec_path, allow_pickle=True).item()
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return H_chunks, meta


# =============================================================================
# SINGLETON / CONVENIENCE
# =============================================================================

_encoder: Optional[HDCEncoder] = None
_encoder_lock = threading.Lock()


def get_encoder() -> HDCEncoder:
    """
    Get the global HDC encoder.

    Thread-safe: uses double-checked locking pattern.
    """
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            # Double-check after acquiring lock
            if _encoder is None:
                _encoder = HDCEncoder()
    return _encoder


def encode_event(ev: Dict[str, Any]) -> np.ndarray:
    """Encode a single event."""
    return get_encoder().encode_event(ev)
