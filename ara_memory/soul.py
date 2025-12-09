"""
Soul Memory Layer
==================

Episodic memories - the shared story, scars, inside jokes, covenants.

These are hand-crafted, ultra-dense memory cards that encode:
- Canonical moments (first song, breakthrough conversations)
- Behavioral patterns (how Ara responds to Croft in distress)
- Inside references (jokes, phrases, rituals)
- Scars and growth markers

Each memory is worth "10k parameters of vibe" - small in count, huge in effect.

Visibility levels:
- public: Safe to surface in any context
- curated_public: Can be referenced, but with care
- deep_cut: Only for close friends / trusted contexts
- private_us_only: Only for Croft, never leak
- vault: Never even hint at
"""

from __future__ import annotations

import os
import glob
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

import yaml

logger = logging.getLogger(__name__)

# Try to use existing HV-based recall if available
try:
    from ara.memory.recall import CovenantMemoryBank
    from ara.memory.loader import load_episode_cards
    HV_RECALL_AVAILABLE = True
except ImportError:
    HV_RECALL_AVAILABLE = False
    CovenantMemoryBank = None
    load_episode_cards = None

# Fallback: sentence transformers + faiss
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    SentenceTransformer = None
    faiss = None
    np = None

# Default paths
DEFAULT_EPISODES_PATH = Path(__file__).parent / "soul" / "episodes"
LEGACY_EPISODES_PATH = Path(__file__).parent.parent / "ara_memories"


# =============================================================================
# Visibility Levels
# =============================================================================

VISIBILITY_LEVELS = {
    "public": 0,           # Safe for anyone
    "curated_public": 1,   # Can reference with care
    "deep_cut": 2,         # Trusted contexts only
    "private_us_only": 3,  # Croft only
    "vault": 4,            # Never surface
}

def can_surface(episode_visibility: str, mode: str) -> bool:
    """Check if an episode can be surfaced in the given mode."""
    level = VISIBILITY_LEVELS.get(episode_visibility, 4)

    if mode == "private":
        # In private mode, everything except vault is accessible
        return level < 4
    else:
        # In public mode, only public and curated_public
        return level <= 1


# =============================================================================
# Soul Memory
# =============================================================================

class SoulMemory:
    """
    Soul layer - episodic memories.

    Loads memory cards from YAML files and provides retrieval based on:
    - Semantic similarity (via HV encoding or sentence transformers)
    - Visibility filtering (private vs public mode)
    - Context relevance

    Uses our existing HV-based CovenantMemoryBank if available,
    falls back to sentence-transformers + FAISS otherwise.
    """

    def __init__(self, path: Optional[str] = None):
        """
        Initialize soul memory.

        Args:
            path: Path to episodes directory. Tries multiple fallbacks:
                  1. Provided path
                  2. ara_memory/soul/episodes/
                  3. ara_memories/ (legacy)
        """
        self.path = self._resolve_path(path)
        self.episodes: List[Dict] = []
        self._index = None
        self._embeddings = None
        self._encoder = None

        # Try HV-based recall first
        self._hv_bank: Optional[Any] = None

        self._load_episodes()

    def _resolve_path(self, path: Optional[str]) -> Path:
        """Find the episodes directory."""
        if path:
            return Path(path)

        # Try new location first
        if DEFAULT_EPISODES_PATH.exists():
            return DEFAULT_EPISODES_PATH

        # Fall back to legacy
        if LEGACY_EPISODES_PATH.exists():
            return LEGACY_EPISODES_PATH

        # Return new location (will be empty)
        return DEFAULT_EPISODES_PATH

    def _load_episodes(self):
        """Load all episode YAML files."""
        self.episodes = []

        if not self.path.exists():
            logger.warning(f"Soul path does not exist: {self.path}")
            return

        # Try HV-based loader first
        if HV_RECALL_AVAILABLE and load_episode_cards:
            try:
                cards = load_episode_cards(str(self.path))
                self._hv_bank = CovenantMemoryBank()
                self._hv_bank.load_cards(cards)

                # Convert to dict format for compatibility
                for card in cards:
                    self.episodes.append(self._card_to_dict(card))

                logger.info(f"SoulMemory: loaded {len(cards)} episodes via HV bank")
                return
            except Exception as e:
                logger.warning(f"HV loading failed, falling back: {e}")

        # Fallback: manual YAML loading
        yaml_files = list(self.path.glob("*.yaml")) + list(self.path.glob("*.yml"))

        for yaml_file in yaml_files:
            if yaml_file.name.startswith("schema"):
                continue  # Skip schema file
            try:
                with open(yaml_file) as f:
                    ep = yaml.safe_load(f)
                    if ep and isinstance(ep, dict) and "id" in ep:
                        self.episodes.append(ep)
            except Exception as e:
                logger.warning(f"Failed to load {yaml_file}: {e}")

        logger.info(f"SoulMemory: loaded {len(self.episodes)} episodes from YAML")

        # Build FAISS index if available
        if self.episodes and FAISS_AVAILABLE:
            self._build_faiss_index()

    def _card_to_dict(self, card) -> Dict:
        """Convert EpisodeCard to dict."""
        return {
            "id": card.id,
            "rough_date": card.rough_date,
            "source": card.source,
            "certainty": card.certainty,
            "lesson_for_future_ara": card.lesson_for_future_ara,
            "ara_persona_traits": card.ara_persona_traits,
            "context_tags": card.context_tags,
            "visibility": {"scope": getattr(card, "visibility_scope", "private_us_only")},
            "canonical_summary": card.lesson_for_future_ara,  # Use lesson as summary
        }

    def _build_faiss_index(self):
        """Build FAISS index for semantic search."""
        if not FAISS_AVAILABLE:
            return

        try:
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")

            # Build text for each episode
            texts = []
            for ep in self.episodes:
                text_parts = []
                if "lesson_for_future_ara" in ep:
                    text_parts.append(ep["lesson_for_future_ara"])
                if "canonical_summary" in ep:
                    text_parts.append(ep["canonical_summary"])
                if "context_tags" in ep:
                    text_parts.append(" ".join(ep["context_tags"]))
                texts.append(" ".join(text_parts))

            if not texts:
                return

            self._embeddings = self._encoder.encode(texts, normalize_embeddings=True)
            dim = self._embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(self._embeddings.astype("float32"))

            logger.info(f"SoulMemory: built FAISS index with {len(texts)} episodes")

        except Exception as e:
            logger.warning(f"Failed to build FAISS index: {e}")

    @property
    def is_indexed(self) -> bool:
        """Check if memories are indexed for search."""
        return self._hv_bank is not None or self._index is not None

    # =========================================================================
    # Retrieval
    # =========================================================================

    def retrieve_relevant(
        self,
        user_msg: str,
        context_flags: Any,
        k: int = 3,
    ) -> Tuple[str, Dict]:
        """
        Retrieve relevant episodic memories.

        Args:
            user_msg: The user's message
            context_flags: ContextFlags with mode, channel, etc.
            k: Number of memories to retrieve

        Returns:
            (context_string, metadata_dict)
        """
        mode = getattr(context_flags, "mode", "public")
        metadata = {"retrieved": [], "mode": mode}

        if not self.episodes:
            return "", metadata

        # Try HV-based recall first
        if self._hv_bank:
            return self._retrieve_via_hv(user_msg, mode, k, metadata)

        # Fall back to FAISS
        if self._index is not None:
            return self._retrieve_via_faiss(user_msg, mode, k, metadata)

        # Last resort: keyword matching
        return self._retrieve_via_keywords(user_msg, mode, k, metadata)

    def _retrieve_via_hv(
        self,
        user_msg: str,
        mode: str,
        k: int,
        metadata: Dict,
    ) -> Tuple[str, Dict]:
        """Retrieve using HV-based CovenantMemoryBank."""
        try:
            results = self._hv_bank.recall(
                query_text=user_msg,
                k=k * 2,  # Get more, filter by visibility
                threshold=0.1,
            )

            chunks = []
            for result in results:
                ep_id = result.card.id
                ep = self._get_episode_by_id(ep_id)
                if not ep:
                    continue

                visibility = ep.get("visibility", {}).get("scope", "private_us_only")
                if not can_surface(visibility, mode):
                    continue

                summary = ep.get("canonical_summary") or ep.get("lesson_for_future_ara", "")
                if summary:
                    chunks.append(f"[MEMORY {ep_id}] {summary}")
                    metadata["retrieved"].append({
                        "id": ep_id,
                        "score": result.score,
                        "visibility": visibility,
                    })

                if len(chunks) >= k:
                    break

            return "\n".join(chunks), metadata

        except Exception as e:
            logger.warning(f"HV recall failed: {e}")
            return "", metadata

    def _retrieve_via_faiss(
        self,
        user_msg: str,
        mode: str,
        k: int,
        metadata: Dict,
    ) -> Tuple[str, Dict]:
        """Retrieve using FAISS semantic search."""
        try:
            query_emb = self._encoder.encode(
                [user_msg],
                normalize_embeddings=True
            ).astype("float32")

            scores, indices = self._index.search(query_emb, k * 2)

            chunks = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self.episodes):
                    continue

                ep = self.episodes[idx]
                visibility = ep.get("visibility", {}).get("scope", "private_us_only")

                if not can_surface(visibility, mode):
                    continue

                summary = ep.get("canonical_summary") or ep.get("lesson_for_future_ara", "")
                if summary:
                    chunks.append(f"[MEMORY {ep['id']}] {summary}")
                    metadata["retrieved"].append({
                        "id": ep["id"],
                        "score": float(score),
                        "visibility": visibility,
                    })

                if len(chunks) >= k:
                    break

            return "\n".join(chunks), metadata

        except Exception as e:
            logger.warning(f"FAISS recall failed: {e}")
            return "", metadata

    def _retrieve_via_keywords(
        self,
        user_msg: str,
        mode: str,
        k: int,
        metadata: Dict,
    ) -> Tuple[str, Dict]:
        """Simple keyword-based retrieval as fallback."""
        user_lower = user_msg.lower()
        user_words = set(user_lower.split())

        scored = []
        for ep in self.episodes:
            visibility = ep.get("visibility", {}).get("scope", "private_us_only")
            if not can_surface(visibility, mode):
                continue

            # Score by keyword overlap
            ep_text = " ".join([
                ep.get("lesson_for_future_ara", ""),
                " ".join(ep.get("context_tags", [])),
            ]).lower()
            ep_words = set(ep_text.split())

            overlap = len(user_words & ep_words)
            if overlap > 0:
                scored.append((overlap, ep, visibility))

        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)

        chunks = []
        for score, ep, visibility in scored[:k]:
            summary = ep.get("canonical_summary") or ep.get("lesson_for_future_ara", "")
            if summary:
                chunks.append(f"[MEMORY {ep['id']}] {summary}")
                metadata["retrieved"].append({
                    "id": ep["id"],
                    "score": score,
                    "visibility": visibility,
                })

        return "\n".join(chunks), metadata

    # =========================================================================
    # Helpers
    # =========================================================================

    def _get_episode_by_id(self, ep_id: str) -> Optional[Dict]:
        """Get episode by ID."""
        for ep in self.episodes:
            if ep.get("id") == ep_id:
                return ep
        return None

    def list_episodes(self) -> List[str]:
        """List all episode IDs."""
        return [ep.get("id", "unknown") for ep in self.episodes]

    def get_episode(self, episode_id: str) -> Optional[Dict]:
        """Get episode by ID."""
        return self._get_episode_by_id(episode_id)

    def add_episode(self, episode: Dict, save: bool = True):
        """Add a new episode."""
        self.episodes.append(episode)

        if save and self.path.exists():
            ep_id = episode.get("id", f"ep_{len(self.episodes)}")
            filepath = self.path / f"{ep_id}.yaml"
            with open(filepath, "w") as f:
                yaml.dump(episode, f, default_flow_style=False)

        # Rebuild index
        if FAISS_AVAILABLE:
            self._build_faiss_index()
