"""
Memory Pack Client
===================

Client library for fetching and using Memory Packs.

This runs on the user's device:
1. Fetches encrypted packs from the server
2. Decrypts using local keys
3. Provides retrieval for LLM context

The server NEVER sees decrypted content.
"""

from __future__ import annotations

import json
import time
import uuid
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from ..crypto.keys import (
    UserKeys,
    SymmetricKey,
    Session,
    aead_encrypt,
    aead_decrypt,
    wrap_key,
    unwrap_key,
    WrappedKey,
)
from ..api.wire_protocol import (
    OuterEnvelope,
    InnerPayload,
    MemoryRecord,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Memory Pack
# =============================================================================

@dataclass
class PackManifest:
    """Manifest for a Memory Pack."""
    pack_id: str
    version: str
    domain: str
    capabilities: List[str]
    created_at: str
    encryption: Dict[str, str]
    signatures: Dict[str, str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PackManifest:
        return cls(
            pack_id=data.get("pack_id", ""),
            version=data.get("version", "1"),
            domain=data.get("domain", ""),
            capabilities=data.get("capabilities", []),
            created_at=data.get("created_at", ""),
            encryption=data.get("encryption", {}),
            signatures=data.get("signatures", {}),
        )


@dataclass
class PackEpisode:
    """A single episode/chunk from a pack."""
    id: str
    type: str  # checklist, prompt_example, procedure, etc.
    tags: List[str]
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PackEpisode:
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "episode"),
            tags=data.get("tags", []),
            text=data.get("text", ""),
            embedding=data.get("embedding"),
            metadata=data.get("meta", {}),
        )


class MemoryPack:
    """
    A decrypted, ready-to-use Memory Pack.

    Provides retrieval for LLM context injection.
    """

    def __init__(
        self,
        manifest: PackManifest,
        episodes: List[PackEpisode],
    ) -> None:
        self.manifest = manifest
        self.episodes = episodes

        # Build index
        self._by_id: Dict[str, PackEpisode] = {ep.id: ep for ep in episodes}
        self._by_tag: Dict[str, List[PackEpisode]] = {}
        for ep in episodes:
            for tag in ep.tags:
                self._by_tag.setdefault(tag, []).append(ep)

    @property
    def pack_id(self) -> str:
        return self.manifest.pack_id

    @property
    def capabilities(self) -> List[str]:
        return self.manifest.capabilities

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        tags: Optional[List[str]] = None,
    ) -> List[PackEpisode]:
        """
        Retrieve relevant episodes for a query.

        v0.1: Simple keyword matching.
        Later: Vector similarity with embeddings.
        """
        candidates = self.episodes

        # Filter by tags if specified
        if tags:
            tag_set = set(tags)
            candidates = [
                ep for ep in candidates
                if tag_set & set(ep.tags)
            ]

        # Score by keyword overlap
        query_words = set(query.lower().split())
        scored = []
        for ep in candidates:
            ep_words = set(ep.text.lower().split())
            overlap = len(query_words & ep_words)
            scored.append((overlap, ep))

        # Sort by score, return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:top_k]]

    def build_prompt(
        self,
        user_query: str,
        context_episodes: List[PackEpisode],
    ) -> str:
        """
        Build a prompt with retrieved context.
        """
        parts = []

        # Add context
        if context_episodes:
            parts.append("# Relevant Knowledge")
            for ep in context_episodes:
                parts.append(f"\n## [{ep.type}] {ep.id}")
                parts.append(ep.text)

        # Add user query
        parts.append("\n# User Question")
        parts.append(user_query)

        return "\n".join(parts)

    def get_by_tag(self, tag: str) -> List[PackEpisode]:
        """Get all episodes with a specific tag."""
        return self._by_tag.get(tag, [])

    def get_by_id(self, episode_id: str) -> Optional[PackEpisode]:
        """Get a specific episode by ID."""
        return self._by_id.get(episode_id)


# =============================================================================
# Memory Pack Client
# =============================================================================

class MemoryPackClient:
    """
    Client for fetching and managing Memory Packs.

    Handles:
    - Fetching encrypted packs from server
    - Decrypting with user keys
    - Caching locally
    """

    def __init__(
        self,
        api_url: str,
        user_keys: UserKeys,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.user_keys = user_keys
        self.cache_dir = cache_dir or Path.home() / ".ara" / "packs"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._loaded_packs: Dict[str, MemoryPack] = {}

    def list_available(self) -> List[Dict[str, Any]]:
        """
        List available packs from server.

        Returns metadata only (pack_id, version, capabilities).
        """
        # In real impl: HTTP GET to /api/packs
        # For now: return stub
        return [
            {
                "pack_id": "indie_publishing",
                "version": "1",
                "domain": "publishing",
                "capabilities": ["kdp_workflow", "blurb_writing", "launch_checklist"],
            },
        ]

    def fetch_and_mount(
        self,
        pack_id: str,
        version: str = "latest",
    ) -> MemoryPack:
        """
        Fetch a pack from server, decrypt, and mount for use.
        """
        cache_key = f"{pack_id}_v{version}"

        # Check cache first
        if cache_key in self._loaded_packs:
            logger.info(f"Using cached pack: {pack_id}")
            return self._loaded_packs[cache_key]

        # Fetch from server (stub - would be HTTP in real impl)
        encrypted_pack = self._fetch_pack(pack_id, version)

        # Decrypt
        pack = self._decrypt_pack(encrypted_pack)

        # Cache
        self._loaded_packs[cache_key] = pack
        logger.info(f"Mounted pack: {pack_id} ({len(pack.episodes)} episodes)")

        return pack

    def _fetch_pack(self, pack_id: str, version: str) -> Dict[str, Any]:
        """
        Fetch encrypted pack from server.

        Returns raw encrypted data.
        """
        # Stub: In real impl, HTTP GET /api/packs/{pack_id}/{version}
        # For dev, load from local file if exists

        local_path = self.cache_dir / f"{pack_id}_v{version}.json"
        if local_path.exists():
            return json.loads(local_path.read_text())

        # Return stub pack
        return {
            "manifest": {
                "pack_id": pack_id,
                "version": version,
                "domain": "publishing",
                "capabilities": ["kdp_workflow"],
                "created_at": "2025-12-09T00:00:00Z",
                "encryption": {"algorithm": "aes-256-gcm"},
                "signatures": {},
            },
            "content": [
                {"id": "ep001", "type": "checklist", "tags": ["kdp", "setup"],
                 "text": "KDP account setup checklist:\n1. Go to kdp.amazon.com\n2. Sign in with Amazon account\n3. Complete tax interview\n4. Set up payment method"},
                {"id": "ep002", "type": "procedure", "tags": ["launch"],
                 "text": "Book launch sequence:\n- T-30 days: Announce to email list\n- T-14 days: Send ARCs to reviewers\n- T-7 days: Set up pre-order\n- T-1 day: Final check\n- Launch day: Promote!"},
            ],
        }

    def _decrypt_pack(self, encrypted: Dict[str, Any]) -> MemoryPack:
        """
        Decrypt a pack using user keys.

        In v0.1: Content might not actually be encrypted.
        The structure is what matters for now.
        """
        manifest = PackManifest.from_dict(encrypted.get("manifest", {}))

        # Decrypt content (stub - content is plaintext in v0.1)
        content = encrypted.get("content", [])
        episodes = [PackEpisode.from_dict(ep) for ep in content]

        return MemoryPack(manifest=manifest, episodes=episodes)


# =============================================================================
# Local Memory Vault
# =============================================================================

class LocalMemoryVault:
    """
    Client-side memory storage.

    Stores:
    - User's episodic memories (encrypted)
    - Local index for retrieval
    - Sync state with server
    """

    def __init__(
        self,
        user_keys: UserKeys,
        vault_path: Path,
    ) -> None:
        self.user_keys = user_keys
        self.vault_path = vault_path
        self.vault_path.mkdir(parents=True, exist_ok=True)

        self._episodes_file = vault_path / "episodes.jsonl"
        self._index_file = vault_path / "index.json"

    def add_episode(
        self,
        kind: str,
        text: str,
        tags: Optional[List[str]] = None,
        scope: str = "private",
    ) -> str:
        """
        Add a new episode to the vault.

        Returns the episode ID.
        """
        episode_id = f"ep_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        record = {
            "id": episode_id,
            "kind": kind,
            "text": text,
            "tags": tags or [],
            "scope": scope,
            "timestamp": time.time(),
        }

        # Encrypt before storing
        plaintext = json.dumps(record).encode()
        dek = SymmetricKey.generate()
        nonce, ciphertext = aead_encrypt(dek.key, plaintext)

        # Wrap DEK with user's mem_kek
        wrapped_dek = wrap_key(self.user_keys.mem_kek.key, dek.key)

        # Store encrypted
        import base64
        encrypted_record = {
            "id": episode_id,
            "kind": kind,  # Metadata can be plaintext for indexing
            "tags": tags or [],
            "scope": scope,
            "dek_wrapped": wrapped_dek.to_b64(),
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ciphertext).decode(),
        }

        with open(self._episodes_file, "a") as f:
            f.write(json.dumps(encrypted_record) + "\n")

        logger.debug(f"Added episode {episode_id}")
        return episode_id

    def list_episodes(
        self,
        kind: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        List episode metadata (not decrypted content).
        """
        if not self._episodes_file.exists():
            return []

        results = []
        with open(self._episodes_file) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)

                if kind and record.get("kind") != kind:
                    continue

                if tags:
                    record_tags = set(record.get("tags", []))
                    if not (set(tags) & record_tags):
                        continue

                # Return metadata only
                results.append({
                    "id": record["id"],
                    "kind": record.get("kind"),
                    "tags": record.get("tags", []),
                    "scope": record.get("scope"),
                })

        return results

    def decrypt_episode(self, episode_id: str) -> Optional[Dict[str, Any]]:
        """
        Decrypt and return a specific episode.
        """
        import base64

        if not self._episodes_file.exists():
            return None

        with open(self._episodes_file) as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["id"] != episode_id:
                    continue

                # Unwrap DEK
                wrapped_dek = WrappedKey.from_b64(record["dek_wrapped"])
                dek = unwrap_key(self.user_keys.mem_kek.key, wrapped_dek)

                # Decrypt content
                nonce = base64.b64decode(record["nonce"])
                ciphertext = base64.b64decode(record["ciphertext"])
                plaintext = aead_decrypt(dek, nonce, ciphertext)

                return json.loads(plaintext)

        return None

    def prepare_for_sync(self) -> List[MemoryRecord]:
        """
        Prepare encrypted records for syncing to server.

        The server will receive encrypted blobs it cannot decrypt.
        """
        import base64

        if not self._episodes_file.exists():
            return []

        records = []
        with open(self._episodes_file) as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)

                record = MemoryRecord(
                    record_id=data["id"],
                    kind=data.get("kind", "episode"),
                    scope=data.get("scope", "private"),
                    tags=data.get("tags", []),
                    dek_wrapped=data["dek_wrapped"],
                    ciphertext=data["ciphertext"],
                    metadata={"synced_at": time.time()},
                )
                records.append(record)

        return records
