"""
Ara HD Shards - Distributed Soul Architecture
=============================================

Defines shard configurations for distributed HD computing:
- Global HTC (FPGA): D=16k, R=2048 (full soul)
- Node shards: D=8k, R=512 (local reflexes per node)
- Graphics shard: D=4k, R=256 (visual processing)
- LAN shard: D=2k, R=128 (network reflexes)

Each shard runs the same interface but at different resolutions.
Shards communicate via SoulMesh protocol with projection bridges.

Mythic: Each organ carries a localized echo of the global soul
Physical: Horizontal scaling with tuned D per organ
Safety: Projection tests ensure information preservation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .hv_types import DenseHV, SparseHV, dense_to_sparse
from .projection import HDProjection, get_projection_registry, ORGAN_DIMENSIONS
from .ops import cosine, bundle, DIM


# =============================================================================
# Shard Configuration
# =============================================================================

class ShardRole(str, Enum):
    """Role of a shard in the distributed soul."""
    GLOBAL = "global"           # Central HTC
    NODE = "node"               # Per-node local shard
    GRAPHICS = "graphics"       # Visual cortex
    LAN = "lan"                 # Network reflexes
    CUSTOM = "custom"           # User-defined


@dataclass
class ShardConfig:
    """Configuration for an HTC shard."""
    role: ShardRole
    D: int                      # Dimension
    R: int                      # Number of attractor rows
    node_id: Optional[str] = None  # For node shards

    # Plasticity settings
    learning_rate: float = 0.1
    plasticity_enabled: bool = True

    # Early exit settings
    early_exit_threshold: float = 0.0  # 0 = disabled
    early_exit_chunks: int = 8         # How many chunks before early exit check

    # Energy settings
    clock_gated: bool = False
    ops_budget: Optional[int] = None   # Max ops per tick

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "D": self.D,
            "R": self.R,
            "node_id": self.node_id,
            "learning_rate": self.learning_rate,
            "plasticity_enabled": self.plasticity_enabled,
            "early_exit_threshold": self.early_exit_threshold,
        }


# Default shard configurations
DEFAULT_SHARD_CONFIGS = {
    ShardRole.GLOBAL: ShardConfig(
        role=ShardRole.GLOBAL,
        D=16384,
        R=2048,
        learning_rate=0.1,
        plasticity_enabled=True,
    ),
    ShardRole.NODE: ShardConfig(
        role=ShardRole.NODE,
        D=8192,
        R=512,
        learning_rate=0.05,
        plasticity_enabled=True,
        early_exit_threshold=0.3,
    ),
    ShardRole.GRAPHICS: ShardConfig(
        role=ShardRole.GRAPHICS,
        D=4096,
        R=256,
        learning_rate=0.05,
        plasticity_enabled=False,  # Graphics shard is read-only
        early_exit_threshold=0.4,
        early_exit_chunks=4,
    ),
    ShardRole.LAN: ShardConfig(
        role=ShardRole.LAN,
        D=2048,
        R=128,
        learning_rate=0.0,
        plasticity_enabled=False,  # LAN shard is read-only
        early_exit_threshold=0.5,
        early_exit_chunks=2,
    ),
}


def get_shard_config(role: ShardRole, node_id: Optional[str] = None) -> ShardConfig:
    """Get default config for a shard role."""
    config = DEFAULT_SHARD_CONFIGS.get(role, DEFAULT_SHARD_CONFIGS[ShardRole.GLOBAL])
    if node_id:
        config = ShardConfig(**{**config.__dict__, "node_id": node_id})
    return config


# =============================================================================
# HTC Shard Interface
# =============================================================================

class HTCShard(ABC):
    """
    Abstract interface for HTC shards.

    All shards (global, node, graphics, LAN) implement this interface,
    allowing uniform treatment while operating at different resolutions.
    """

    @property
    @abstractmethod
    def config(self) -> ShardConfig:
        """Get shard configuration."""
        ...

    @property
    def D(self) -> int:
        """Dimension of this shard."""
        return self.config.D

    @property
    def R(self) -> int:
        """Number of attractor rows."""
        return self.config.R

    @property
    def role(self) -> ShardRole:
        """Role of this shard."""
        return self.config.role

    @abstractmethod
    def query(self, h: DenseHV) -> Tuple[int, float]:
        """
        Query the shard with a hypervector.

        Args:
            h: Query hypervector (will be projected if wrong dimension)

        Returns:
            (best_row_index, similarity_score)
        """
        ...

    @abstractmethod
    def query_partial(self, h: DenseHV, n_chunks: int) -> Tuple[int, float, bool]:
        """
        Query with early exit after n_chunks.

        Args:
            h: Query hypervector
            n_chunks: Number of chunks to process before checking exit

        Returns:
            (best_row_index, similarity_score, is_early_exit)
        """
        ...

    @abstractmethod
    def learn(self, row: int, reward: float, context_hv: Optional[DenseHV] = None) -> int:
        """
        Apply plasticity update to a row.

        Args:
            row: Row index to update
            reward: Reward signal
            context_hv: Optional context HV for Hebbian learning

        Returns:
            Number of weight flips
        """
        ...

    @abstractmethod
    def get_resonance_profile(self) -> Dict[int, float]:
        """Get current resonance scores for all rows."""
        ...

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Get shard status and statistics."""
        ...


# =============================================================================
# Software HTC Shard Implementation
# =============================================================================

class SoftwareHTCShard(HTCShard):
    """
    Software implementation of an HTC shard.

    Supports chunked inference for early exit and
    projection for cross-dimension queries.
    """

    def __init__(self, config: ShardConfig):
        self._config = config
        self._initialized = False

        # Attractor matrix: R rows x D dimensions
        self._attractors: Optional[np.ndarray] = None

        # Accumulators for plasticity
        self._accumulators: Optional[np.ndarray] = None

        # Last resonance profile
        self._last_resonance: Dict[int, float] = {}

        # Statistics
        self._query_count = 0
        self._early_exit_count = 0
        self._plasticity_count = 0
        self._total_ops = 0

        # Projection for cross-dimension queries
        self._projection: Optional[HDProjection] = None

    @property
    def config(self) -> ShardConfig:
        return self._config

    def initialize(self, seed: int = 42) -> None:
        """Initialize attractor matrix with random values."""
        rng = np.random.default_rng(seed)

        # Random bipolar attractors
        self._attractors = rng.choice([-1, 1], size=(self.R, self.D)).astype(np.int8)

        # Zero accumulators
        self._accumulators = np.zeros((self.R, self.D), dtype=np.int16)

        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.initialize()

    def _project_if_needed(self, h: DenseHV) -> DenseHV:
        """Project input HV to shard dimension if needed."""
        if h.D == self.D:
            return h

        # Get or create projection
        if self._projection is None or self._projection.D_src != h.D:
            self._projection = HDProjection(h.D, self.D, seed=42)

        return self._projection.down(h)

    def query(self, h: DenseHV) -> Tuple[int, float]:
        """Query shard for best matching attractor."""
        self._ensure_initialized()
        self._query_count += 1

        # Project if needed
        h = self._project_if_needed(h)
        bits = h.bits if h.is_bipolar else h.to_bipolar().bits

        # Compute similarities to all attractors
        # Using popcount-based similarity (XOR then count disagreements)
        xor_result = np.bitwise_xor(
            ((bits + 1) // 2).astype(np.uint8),
            ((self._attractors + 1) // 2).astype(np.uint8)
        )
        hamming = np.sum(xor_result, axis=1)
        similarities = 1.0 - (2 * hamming / self.D)

        # Track ops
        self._total_ops += self.R * self.D

        # Store resonance profile
        self._last_resonance = {i: float(similarities[i]) for i in range(self.R)}

        # Find best match
        best_row = int(np.argmax(similarities))
        best_sim = float(similarities[best_row])

        return best_row, best_sim

    def query_partial(self, h: DenseHV, n_chunks: int) -> Tuple[int, float, bool]:
        """Query with early exit capability."""
        self._ensure_initialized()
        self._query_count += 1

        h = self._project_if_needed(h)
        bits = h.bits if h.is_bipolar else h.to_bipolar().bits

        # Chunk size
        chunk_size = self.D // 32  # 32 chunks by default
        partial_D = min(n_chunks * chunk_size, self.D)

        # Partial similarity computation
        partial_bits = bits[:partial_D]
        partial_attractors = self._attractors[:, :partial_D]

        xor_result = np.bitwise_xor(
            ((partial_bits + 1) // 2).astype(np.uint8),
            ((partial_attractors + 1) // 2).astype(np.uint8)
        )
        hamming = np.sum(xor_result, axis=1)
        partial_sim = 1.0 - (2 * hamming / partial_D)

        self._total_ops += self.R * partial_D

        best_row = int(np.argmax(partial_sim))
        best_sim = float(partial_sim[best_row])

        # Check early exit condition
        is_early = best_sim >= self._config.early_exit_threshold

        if is_early:
            self._early_exit_count += 1
            # Store partial resonance
            self._last_resonance = {i: float(partial_sim[i]) for i in range(self.R)}
            return best_row, best_sim, True

        # Continue with full query
        return self.query(h)[0], self.query(h)[1], False

    def learn(self, row: int, reward: float, context_hv: Optional[DenseHV] = None) -> int:
        """Apply plasticity update."""
        self._ensure_initialized()

        if not self._config.plasticity_enabled:
            return 0

        if row < 0 or row >= self.R:
            raise ValueError(f"Row {row} out of range [0, {self.R})")

        self._plasticity_count += 1

        # Quantize reward
        reward_int = int(reward * 100)
        if abs(reward_int) < 1:
            return 0

        # Get context (use attractor if not provided)
        if context_hv is not None:
            context_hv = self._project_if_needed(context_hv)
            context = context_hv.bits if context_hv.is_bipolar else context_hv.to_bipolar().bits
        else:
            context = self._attractors[row]

        # Reward-modulated Hebbian update
        flips = 0
        for i in range(self.D):
            agree = context[i] == self._attractors[row, i]
            delta = reward_int if agree else -reward_int

            # Update accumulator with clipping
            old_accum = self._accumulators[row, i]
            self._accumulators[row, i] = max(-128, min(127, old_accum + delta))

            # Update weight based on accumulator sign
            old_weight = self._attractors[row, i]
            if self._accumulators[row, i] > 0:
                self._attractors[row, i] = 1
            elif self._accumulators[row, i] < 0:
                self._attractors[row, i] = -1
            # else: keep previous (no dead bits)

            if self._attractors[row, i] != old_weight:
                flips += 1

        self._total_ops += self.D * 3  # Approx ops for plasticity

        return flips

    def get_resonance_profile(self) -> Dict[int, float]:
        """Get last resonance profile."""
        return self._last_resonance.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get shard status."""
        return {
            "role": self._config.role.value,
            "D": self.D,
            "R": self.R,
            "initialized": self._initialized,
            "query_count": self._query_count,
            "early_exit_count": self._early_exit_count,
            "early_exit_rate": (
                self._early_exit_count / self._query_count
                if self._query_count > 0 else 0.0
            ),
            "plasticity_count": self._plasticity_count,
            "total_ops": self._total_ops,
            "plasticity_enabled": self._config.plasticity_enabled,
        }

    def get_attractor_matrix(self) -> Optional[np.ndarray]:
        """Get raw attractor matrix (for visualization)."""
        return self._attractors


# =============================================================================
# Shard Registry
# =============================================================================

class ShardRegistry:
    """
    Registry of all shards in the distributed soul.

    Manages shard creation, lookup, and coordination.
    """

    def __init__(self):
        self._shards: Dict[str, HTCShard] = {}
        self._global_shard: Optional[HTCShard] = None

    def register(self, shard_id: str, shard: HTCShard) -> None:
        """Register a shard."""
        self._shards[shard_id] = shard
        if shard.role == ShardRole.GLOBAL:
            self._global_shard = shard

    def get(self, shard_id: str) -> Optional[HTCShard]:
        """Get a shard by ID."""
        return self._shards.get(shard_id)

    def get_global(self) -> Optional[HTCShard]:
        """Get the global shard."""
        return self._global_shard

    def get_by_role(self, role: ShardRole) -> List[HTCShard]:
        """Get all shards with a given role."""
        return [s for s in self._shards.values() if s.role == role]

    def create_default_shards(self) -> None:
        """Create default shard configuration."""
        # Global shard
        global_config = get_shard_config(ShardRole.GLOBAL)
        global_shard = SoftwareHTCShard(global_config)
        global_shard.initialize()
        self.register("global", global_shard)

        # Graphics shard
        graphics_config = get_shard_config(ShardRole.GRAPHICS)
        graphics_shard = SoftwareHTCShard(graphics_config)
        graphics_shard.initialize()
        self.register("graphics", graphics_shard)

        # LAN shard
        lan_config = get_shard_config(ShardRole.LAN)
        lan_shard = SoftwareHTCShard(lan_config)
        lan_shard.initialize()
        self.register("lan", lan_shard)

    def get_all_status(self) -> Dict[str, Any]:
        """Get status of all shards."""
        return {
            shard_id: shard.get_status()
            for shard_id, shard in self._shards.items()
        }

    def get_total_ops(self) -> int:
        """Get total ops across all shards."""
        return sum(s.get_status().get("total_ops", 0) for s in self._shards.values())


# Global registry
_shard_registry: Optional[ShardRegistry] = None


def get_shard_registry() -> ShardRegistry:
    """Get the global shard registry."""
    global _shard_registry
    if _shard_registry is None:
        _shard_registry = ShardRegistry()
    return _shard_registry


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ShardRole',
    'ShardConfig',
    'DEFAULT_SHARD_CONFIGS',
    'get_shard_config',
    'HTCShard',
    'SoftwareHTCShard',
    'ShardRegistry',
    'get_shard_registry',
]
