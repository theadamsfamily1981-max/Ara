"""
Ara SaaS - Memory Packs as a Service
=====================================

Client-encrypted knowledge packs for any LLM.

Architecture:
- Memory Packs: Curated, encrypted knowledge bundles
- Blind Router: Routes envelopes without decrypting content
- Memory Fabric: Stores encrypted blobs server cannot read
- Client Crypto: E2E encryption on user devices

The server NEVER sees plaintext user data.

Usage:
    # Server (FastAPI)
    from ara_saas.api import create_app
    app = create_app()

    # Client
    from ara_saas.client import MemoryPackClient, LocalMemoryVault
    from ara_saas.crypto import UserKeys

    keys = UserKeys.generate("user123")
    client = MemoryPackClient(api_url, keys)
    pack = client.fetch_and_mount("indie_publishing", "1")
    episodes = pack.retrieve("how to write a blurb")
"""

from .api import create_app, BlindRouter, OuterEnvelope, ResponseEnvelope
from .crypto import UserKeys, DeviceKeys, Session
from .client import MemoryPackClient, MemoryPack, LocalMemoryVault
from .storage import MemoryFabric

__all__ = [
    # API
    "create_app",
    "BlindRouter",
    "OuterEnvelope",
    "ResponseEnvelope",
    # Crypto
    "UserKeys",
    "DeviceKeys",
    "Session",
    # Client
    "MemoryPackClient",
    "MemoryPack",
    "LocalMemoryVault",
    # Storage
    "MemoryFabric",
]
