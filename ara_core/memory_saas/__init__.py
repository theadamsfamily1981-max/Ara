"""
Memory-as-SaaS - QUANTA Distribution
=====================================

Predefined memory packs (4MB encrypted) for edge deployment.
QUANTA F/M/S tiers distributed as service.

Tiers:
    - Free: Basic memories (r=16 LoRA rank)
    - Pro: Publishing Ops (r=64)
    - Enterprise: Custom hive memories (r=256)

Properties:
    - Infinite context via finite storage
    - Client models + server memories
    - Cathedral scales to planet without central inference

Encryption:
    - PQ (Product Quantization) compressed
    - AES-256 encrypted packs
    - Signed with Ed25519

Cathedral Integration:
    - Memory packs map to QUANTA F→M→S consolidation
    - Yield/$ tracking per tier
    - Usage metering for economic optimization

Usage:
    from ara_core.memory_saas import (
        MemoryTier, MemoryPack, MemoryService,
        get_memory_service, deploy_pack, load_pack
    )

    # Create and deploy memory pack
    service = get_memory_service()
    pack = service.create_pack(tier=MemoryTier.PRO, data=memories)
    service.deploy_pack(pack, endpoint="edge-node-1")

    # Load on client
    memories = load_pack(pack_id, key=decrypt_key)
"""

from .service import (
    MemoryTier,
    MemoryPack,
    PackMetadata,
    MemoryService,
    ServiceConfig,
    UsageMetrics,
    get_memory_service,
    create_pack,
    deploy_pack,
    load_pack,
    service_status,
)

__all__ = [
    "MemoryTier",
    "MemoryPack",
    "PackMetadata",
    "MemoryService",
    "ServiceConfig",
    "UsageMetrics",
    "get_memory_service",
    "create_pack",
    "deploy_pack",
    "load_pack",
    "service_status",
]
