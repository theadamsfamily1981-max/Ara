"""
Ara Kernel Memory
==================

Three-layer memory system:
- Episodes: SQLite-backed episodic log
- Embeddings: Vector similarity search
- Packs: Structured knowledge bundles

This implements the MemoryBackend protocol for the kernel.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .episodes import EpisodeStore, Episode
from .embeddings import EmbeddingIndex
from .packs import KnowledgePack, PackManager, load_pack

logger = logging.getLogger(__name__)

__all__ = [
    "EpisodeStore",
    "Episode",
    "EmbeddingIndex",
    "KnowledgePack",
    "PackManager",
    "load_pack",
    "MemoryBackend",
]


@dataclass
class MemoryBackend:
    """
    Unified memory backend implementing the kernel's memory protocol.

    Combines:
    - Episodic memory (recent events, conversations)
    - Semantic memory (embedding-based retrieval)
    - Knowledge packs (structured domain knowledge)
    """
    episodes: EpisodeStore
    embeddings: EmbeddingIndex
    packs: Dict[str, KnowledgePack]

    def retrieve(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant context for an event.

        Returns a summary combining:
        - Recent episodes from the same domain
        - Semantically similar content
        - Relevant knowledge pack content
        """
        domain = event.get("domain", "default")
        query = event.get("query") or event.get("text") or str(event)

        summary_parts: List[str] = []

        # 1. Recent episodes
        recent = self.episodes.recent(domain=domain, limit=10)
        if recent:
            summary_parts.append("Recent episodes:")
            for ep in recent[:5]:  # Show last 5
                summary_parts.append(f"  [{ep.role}] {ep.text[:120]}...")

        # 2. Embedding search
        emb_hits = self.embeddings.search(query, k=5)
        if emb_hits:
            summary_parts.append("\nSemantic matches:")
            for hit in emb_hits[:3]:  # Show top 3
                summary_parts.append(f"  (score={hit.score:.2f}) {hit.text[:100]}...")

        # 3. Knowledge packs (by domain/tags)
        tags = [domain] + event.get("tags", [])
        pack_docs = []
        for pack in self.packs.values():
            pack_docs.extend(pack.search_tags(tags)[:2])

        if pack_docs:
            summary_parts.append("\nKnowledge pack context:")
            for doc in pack_docs[:3]:
                summary_parts.append(f"  [{doc.id}] {doc.text[:100]}...")

        return {
            "summary": "\n".join(summary_parts) if summary_parts else "No relevant context found.",
            "recent_count": len(recent),
            "embedding_hits": len(emb_hits),
            "pack_docs": len(pack_docs),
        }

    def store(
        self,
        event: Dict[str, Any],
        result: Dict[str, Any],
        plan: Dict[str, Any],
    ) -> None:
        """
        Store an event and its result in memory.

        Creates an episode and optionally indexes for semantic search.
        """
        domain = event.get("domain", "default")
        text = event.get("text") or str(event)

        # Store as episode
        ep = Episode(
            ts=time.time(),
            domain=domain,
            role="event",
            text=text,
            meta={
                "result_summary": str(result.get("result", {}))[:500],
                "plan_thought": plan.get("thought", ""),
                "actions_count": len(plan.get("actions", [])),
            },
        )
        self.episodes.add(ep)

        # Also index in embeddings if significant
        if len(text) > 50:
            self.embeddings.add(
                text=text,
                meta={"domain": domain, "type": "event"},
            )

    @classmethod
    def create(
        cls,
        db_path: Path,
        packs_dir: Optional[Path] = None,
        embedding_dim: int = 384,
    ) -> MemoryBackend:
        """
        Factory method to create a MemoryBackend.

        Args:
            db_path: Path to episodes SQLite database
            packs_dir: Directory containing knowledge packs
            embedding_dim: Dimension for embedding vectors
        """
        episodes = EpisodeStore(db_path)
        embeddings = EmbeddingIndex(dim=embedding_dim)

        packs: Dict[str, KnowledgePack] = {}
        if packs_dir and packs_dir.exists():
            manager = PackManager(packs_dir)
            manager.load_all()
            packs = manager.packs

        return cls(
            episodes=episodes,
            embeddings=embeddings,
            packs=packs,
        )
