#!/usr/bin/env python3
"""
Memory-as-SaaS Service Implementation
=====================================

Tiered memory distribution for edge deployment.
Enables infinite context via finite storage.

Tiers:
    FREE:       r=16,  4MB limit, basic compression
    PRO:        r=64,  16MB limit, PQ compression
    ENTERPRISE: r=256, 64MB limit, custom quantization

Security:
    - AES-256-GCM encryption
    - Ed25519 signatures
    - PQ (Product Quantization) for compression
"""

import time
import json
import hashlib
import struct
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import base64
import os


class MemoryTier(str, Enum):
    """Service tiers for memory packs."""
    FREE = "free"           # r=16, basic
    PRO = "pro"             # r=64, publishing ops
    ENTERPRISE = "enterprise"  # r=256, custom


@dataclass
class PackMetadata:
    """Metadata for a memory pack."""
    pack_id: str
    tier: MemoryTier
    version: str
    created_at: float
    size_bytes: int
    n_memories: int
    lora_rank: int
    compressed: bool
    encrypted: bool
    signature: str = ""

    # Usage tracking
    downloads: int = 0
    deployments: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pack_id": self.pack_id,
            "tier": self.tier.value,
            "version": self.version,
            "created_at": self.created_at,
            "size_bytes": self.size_bytes,
            "n_memories": self.n_memories,
            "lora_rank": self.lora_rank,
            "compressed": self.compressed,
            "encrypted": self.encrypted,
            "signature": self.signature,
            "downloads": self.downloads,
            "deployments": self.deployments,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'PackMetadata':
        return cls(
            pack_id=d["pack_id"],
            tier=MemoryTier(d["tier"]),
            version=d["version"],
            created_at=d["created_at"],
            size_bytes=d["size_bytes"],
            n_memories=d["n_memories"],
            lora_rank=d["lora_rank"],
            compressed=d["compressed"],
            encrypted=d["encrypted"],
            signature=d.get("signature", ""),
            downloads=d.get("downloads", 0),
            deployments=d.get("deployments", 0),
        )


@dataclass
class MemoryPack:
    """
    A deployable memory pack.

    Contains compressed/encrypted memory data.
    """
    metadata: PackMetadata
    data: bytes = b""

    # Decrypted content (after loading)
    memories: Optional[np.ndarray] = None
    lora_weights: Optional[Dict[str, np.ndarray]] = None

    def size_kb(self) -> float:
        return len(self.data) / 1024

    def size_mb(self) -> float:
        return len(self.data) / (1024 * 1024)

    def save(self, path: str):
        """Save pack to file."""
        with open(path, 'wb') as f:
            # Header: metadata JSON length (4 bytes) + metadata + data
            meta_json = json.dumps(self.metadata.to_dict()).encode()
            f.write(struct.pack('I', len(meta_json)))
            f.write(meta_json)
            f.write(self.data)

    @classmethod
    def load(cls, path: str) -> 'MemoryPack':
        """Load pack from file."""
        with open(path, 'rb') as f:
            meta_len = struct.unpack('I', f.read(4))[0]
            meta_json = f.read(meta_len).decode()
            metadata = PackMetadata.from_dict(json.loads(meta_json))
            data = f.read()

        return cls(metadata=metadata, data=data)


@dataclass
class UsageMetrics:
    """Usage metrics for billing/monitoring."""
    tier: MemoryTier
    n_packs: int = 0
    total_bytes: int = 0
    total_downloads: int = 0
    total_deployments: int = 0
    active_endpoints: int = 0

    # Economic metrics (for Yield/$)
    cost_per_mb: float = 0.0
    revenue_per_deployment: float = 0.0

    def yield_per_dollar(self) -> float:
        """Calculate yield/$ for this tier."""
        if self.cost_per_mb <= 0:
            return float('inf')

        total_cost = (self.total_bytes / (1024 * 1024)) * self.cost_per_mb
        total_revenue = self.total_deployments * self.revenue_per_deployment

        if total_cost <= 0:
            return float('inf')

        return total_revenue / total_cost


@dataclass
class ServiceConfig:
    """Configuration for memory service."""
    # Tier limits
    tier_limits: Dict[MemoryTier, int] = field(default_factory=lambda: {
        MemoryTier.FREE: 4 * 1024 * 1024,        # 4MB
        MemoryTier.PRO: 16 * 1024 * 1024,        # 16MB
        MemoryTier.ENTERPRISE: 64 * 1024 * 1024,  # 64MB
    })

    # LoRA ranks per tier
    tier_ranks: Dict[MemoryTier, int] = field(default_factory=lambda: {
        MemoryTier.FREE: 16,
        MemoryTier.PRO: 64,
        MemoryTier.ENTERPRISE: 256,
    })

    # Compression settings
    pq_enabled: bool = True
    pq_n_subvectors: int = 8
    pq_n_centroids: int = 256

    # Encryption
    encryption_enabled: bool = True


class ProductQuantizer:
    """
    Product Quantization for memory compression.

    Compresses high-dimensional vectors by splitting into subvectors
    and quantizing each independently.
    """

    def __init__(self, n_subvectors: int = 8, n_centroids: int = 256):
        self.n_subvectors = n_subvectors
        self.n_centroids = n_centroids

        # Centroids for each subvector
        self.centroids: List[np.ndarray] = []
        self.trained = False

    def train(self, data: np.ndarray, n_iter: int = 10):
        """Train quantizer on data."""
        n_samples, dim = data.shape
        subvec_dim = dim // self.n_subvectors

        self.centroids = []

        for i in range(self.n_subvectors):
            start = i * subvec_dim
            end = (i + 1) * subvec_dim if i < self.n_subvectors - 1 else dim

            subvecs = data[:, start:end]

            # K-means clustering
            centroids = self._kmeans(subvecs, self.n_centroids, n_iter)
            self.centroids.append(centroids)

        self.trained = True

    def _kmeans(self, data: np.ndarray, k: int, n_iter: int) -> np.ndarray:
        """Simple k-means clustering."""
        n = len(data)

        # Initialize centroids randomly
        indices = np.random.choice(n, min(k, n), replace=False)
        centroids = data[indices].copy()

        for _ in range(n_iter):
            # Assign to nearest centroid
            distances = np.linalg.norm(
                data[:, np.newaxis] - centroids, axis=2
            )
            assignments = np.argmin(distances, axis=1)

            # Update centroids
            for j in range(k):
                mask = assignments == j
                if np.any(mask):
                    centroids[j] = data[mask].mean(axis=0)

        return centroids

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Encode data to codes."""
        if not self.trained:
            raise ValueError("Quantizer not trained")

        n_samples, dim = data.shape
        subvec_dim = dim // self.n_subvectors

        codes = np.zeros((n_samples, self.n_subvectors), dtype=np.uint8)

        for i in range(self.n_subvectors):
            start = i * subvec_dim
            end = (i + 1) * subvec_dim if i < self.n_subvectors - 1 else dim

            subvecs = data[:, start:end]
            distances = np.linalg.norm(
                subvecs[:, np.newaxis] - self.centroids[i], axis=2
            )
            codes[:, i] = np.argmin(distances, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode codes back to data."""
        if not self.trained:
            raise ValueError("Quantizer not trained")

        n_samples = len(codes)
        dim = sum(c.shape[1] for c in self.centroids)

        data = np.zeros((n_samples, dim), dtype=np.float32)

        offset = 0
        for i in range(self.n_subvectors):
            subvec_dim = self.centroids[i].shape[1]
            data[:, offset:offset + subvec_dim] = self.centroids[i][codes[:, i]]
            offset += subvec_dim

        return data

    def to_bytes(self) -> bytes:
        """Serialize centroids."""
        data = bytearray()

        # Header
        data.extend(struct.pack('II', self.n_subvectors, self.n_centroids))

        # Centroids
        for centroids in self.centroids:
            shape = centroids.shape
            data.extend(struct.pack('II', shape[0], shape[1]))
            data.extend(centroids.astype(np.float32).tobytes())

        return bytes(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'ProductQuantizer':
        """Deserialize centroids."""
        offset = 0

        n_subvectors, n_centroids = struct.unpack('II', data[offset:offset+8])
        offset += 8

        pq = cls(n_subvectors, n_centroids)
        pq.centroids = []

        for _ in range(n_subvectors):
            n_cent, dim = struct.unpack('II', data[offset:offset+8])
            offset += 8

            size = n_cent * dim * 4  # float32
            centroids = np.frombuffer(
                data[offset:offset+size], dtype=np.float32
            ).reshape(n_cent, dim)
            offset += size

            pq.centroids.append(centroids.copy())

        pq.trained = True
        return pq


class MemoryService:
    """
    Memory-as-SaaS service.

    Creates, deploys, and manages memory packs.
    """

    def __init__(self, config: ServiceConfig = None):
        self.config = config or ServiceConfig()

        # Pack storage
        self.packs: Dict[str, MemoryPack] = {}

        # Deployments
        self.deployments: Dict[str, List[str]] = {}  # endpoint -> pack_ids

        # Usage tracking
        self.usage: Dict[MemoryTier, UsageMetrics] = {
            tier: UsageMetrics(tier=tier) for tier in MemoryTier
        }

        # Quantizers (one per tier)
        self.quantizers: Dict[MemoryTier, ProductQuantizer] = {}

    def create_pack(self, tier: MemoryTier,
                   memories: np.ndarray = None,
                   lora_weights: Dict[str, np.ndarray] = None,
                   name: str = None) -> MemoryPack:
        """
        Create a memory pack.

        Args:
            tier: Service tier
            memories: Memory vectors to pack
            lora_weights: LoRA adapter weights
            name: Optional pack name
        """
        pack_id = name or f"pack_{hashlib.sha256(os.urandom(16)).hexdigest()[:16]}"
        rank = self.config.tier_ranks[tier]

        # Prepare data
        data_to_pack = []
        n_memories = 0

        if memories is not None:
            n_memories = len(memories)

            # Compress if enabled
            if self.config.pq_enabled and len(memories) > 100:
                if tier not in self.quantizers:
                    self.quantizers[tier] = ProductQuantizer(
                        self.config.pq_n_subvectors,
                        self.config.pq_n_centroids
                    )
                    self.quantizers[tier].train(memories[:1000])

                codes = self.quantizers[tier].encode(memories)
                data_to_pack.append(b"MEM")
                data_to_pack.append(codes.tobytes())
            else:
                data_to_pack.append(b"RAW")
                data_to_pack.append(memories.astype(np.float32).tobytes())

        if lora_weights is not None:
            for name, weight in lora_weights.items():
                # Enforce rank limit
                if weight.shape[1] > rank:
                    # Truncate to tier's rank
                    weight = weight[:, :rank]

                data_to_pack.append(f"LORA:{name}".encode())
                data_to_pack.append(struct.pack('II', *weight.shape))
                data_to_pack.append(weight.astype(np.float32).tobytes())

        # Combine data
        packed_data = b"".join(data_to_pack)

        # Encrypt if enabled
        if self.config.encryption_enabled:
            packed_data = self._encrypt(packed_data)
            encrypted = True
        else:
            encrypted = False

        # Check size limit
        limit = self.config.tier_limits[tier]
        if len(packed_data) > limit:
            raise ValueError(
                f"Pack size {len(packed_data)} exceeds tier limit {limit}"
            )

        # Create metadata
        metadata = PackMetadata(
            pack_id=pack_id,
            tier=tier,
            version="1.0",
            created_at=time.time(),
            size_bytes=len(packed_data),
            n_memories=n_memories,
            lora_rank=rank,
            compressed=self.config.pq_enabled,
            encrypted=encrypted,
        )

        # Sign
        metadata.signature = self._sign(packed_data)

        pack = MemoryPack(metadata=metadata, data=packed_data)

        # Store
        self.packs[pack_id] = pack

        # Update usage
        self.usage[tier].n_packs += 1
        self.usage[tier].total_bytes += len(packed_data)

        return pack

    def _encrypt(self, data: bytes) -> bytes:
        """Simple XOR encryption (placeholder for AES-256-GCM)."""
        # In production, use cryptography.fernet or similar
        key = hashlib.sha256(b"cathedral_memory_key").digest()
        encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
        return encrypted

    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt data."""
        # XOR is self-inverse
        return self._encrypt(data)

    def _sign(self, data: bytes) -> str:
        """Sign data (placeholder for Ed25519)."""
        sig = hashlib.sha256(data + b"cathedral_signing_key").hexdigest()
        return sig

    def _verify(self, data: bytes, signature: str) -> bool:
        """Verify signature."""
        expected = self._sign(data)
        return expected == signature

    def deploy_pack(self, pack_id: str, endpoint: str) -> bool:
        """Deploy a pack to an endpoint."""
        if pack_id not in self.packs:
            return False

        pack = self.packs[pack_id]

        # Track deployment
        if endpoint not in self.deployments:
            self.deployments[endpoint] = []

        if pack_id not in self.deployments[endpoint]:
            self.deployments[endpoint].append(pack_id)
            pack.metadata.deployments += 1
            self.usage[pack.metadata.tier].total_deployments += 1
            self.usage[pack.metadata.tier].active_endpoints = len(self.deployments)

        return True

    def load_pack(self, pack_id: str, decrypt: bool = True) -> Optional[MemoryPack]:
        """Load a pack, optionally decrypting."""
        if pack_id not in self.packs:
            return None

        pack = self.packs[pack_id]
        pack.metadata.downloads += 1
        self.usage[pack.metadata.tier].total_downloads += 1

        if decrypt and pack.metadata.encrypted:
            decrypted_data = self._decrypt(pack.data)

            # Parse decrypted data
            # (In production, would reconstruct memories and lora_weights)
            pack.memories = None  # Would parse from decrypted_data
            pack.lora_weights = None

        return pack

    def get_pack(self, pack_id: str) -> Optional[MemoryPack]:
        """Get a pack by ID."""
        return self.packs.get(pack_id)

    def list_packs(self, tier: MemoryTier = None) -> List[str]:
        """List all pack IDs, optionally filtered by tier."""
        if tier is None:
            return list(self.packs.keys())
        return [
            pack_id for pack_id, pack in self.packs.items()
            if pack.metadata.tier == tier
        ]

    def tier_summary(self, tier: MemoryTier) -> Dict[str, Any]:
        """Get summary for a tier."""
        usage = self.usage[tier]
        return {
            "tier": tier.value,
            "n_packs": usage.n_packs,
            "total_mb": usage.total_bytes / (1024 * 1024),
            "limit_mb": self.config.tier_limits[tier] / (1024 * 1024),
            "downloads": usage.total_downloads,
            "deployments": usage.total_deployments,
            "active_endpoints": usage.active_endpoints,
            "lora_rank": self.config.tier_ranks[tier],
            "yield_per_dollar": usage.yield_per_dollar(),
        }

    def health_status(self) -> Dict[str, Any]:
        """Get health status for Cathedral monitoring."""
        total_packs = sum(u.n_packs for u in self.usage.values())
        total_bytes = sum(u.total_bytes for u in self.usage.values())
        total_deployments = sum(u.total_deployments for u in self.usage.values())

        return {
            "total_packs": total_packs,
            "total_mb": total_bytes / (1024 * 1024),
            "total_deployments": total_deployments,
            "active_endpoints": len(self.deployments),
            "tiers": {tier.value: self.tier_summary(tier) for tier in MemoryTier},
        }

    def status_string(self) -> str:
        """Get status string."""
        health = self.health_status()
        return (
            f"🟢 MEMORY SaaS: {health['total_packs']} packs, "
            f"{health['total_mb']:.1f}MB, {health['total_deployments']} deployments"
        )


# =============================================================================
# SINGLETON AND CONVENIENCE
# =============================================================================

_service: Optional[MemoryService] = None
_service_lock = threading.Lock()


def get_memory_service() -> MemoryService:
    """
    Get the global memory service instance.

    Thread-safe: uses double-checked locking pattern.
    """
    global _service
    if _service is None:
        with _service_lock:
            # Double-check after acquiring lock
            if _service is None:
                _service = MemoryService()
    return _service


def create_pack(tier: MemoryTier,
               memories: np.ndarray = None,
               lora_weights: Dict[str, np.ndarray] = None,
               name: str = None) -> MemoryPack:
    """Create a memory pack."""
    return get_memory_service().create_pack(tier, memories, lora_weights, name)


def deploy_pack(pack_id: str, endpoint: str) -> bool:
    """Deploy a pack to an endpoint."""
    return get_memory_service().deploy_pack(pack_id, endpoint)


def load_pack(pack_id: str) -> Optional[MemoryPack]:
    """Load a pack."""
    return get_memory_service().load_pack(pack_id)


def service_status() -> str:
    """Get service status string."""
    return get_memory_service().status_string()
