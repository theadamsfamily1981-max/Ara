"""
Ara SaaS Storage
=================

Server-side storage for encrypted memory records.

The server stores encrypted blobs it CANNOT decrypt.
Only metadata (tags, kind, scope) is readable for indexing.
"""

from .memory_fabric import MemoryFabric

__all__ = [
    "MemoryFabric",
]
