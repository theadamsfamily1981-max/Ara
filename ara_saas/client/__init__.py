"""
Ara SaaS Client
================

Client library for Memory Packs and local memory vault.

Runs on user's device:
1. Fetches encrypted packs from server
2. Decrypts using local keys
3. Provides retrieval for LLM context injection

The server NEVER sees decrypted content.
"""

from .memory_client import (
    MemoryPackClient,
    MemoryPack,
    PackManifest,
    PackEpisode,
    LocalMemoryVault,
)

__all__ = [
    "MemoryPackClient",
    "MemoryPack",
    "PackManifest",
    "PackEpisode",
    "LocalMemoryVault",
]
