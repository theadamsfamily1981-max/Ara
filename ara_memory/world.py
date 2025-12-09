"""
World Memory Layer
===================

Compressed knowledge, trend oracles, and external context.

This layer provides:
- Embeddings over docs, trends, APIs
- FAISS/HNSW indexes for semantic search
- Publishing workflow graphs
- Service schemas and rate limits

The "4MB = 100B param equivalent" is poetic, not literal, but the effect is real:
each MB encodes hundreds of thousands of facts and relationships that the base
model can query by embedding.

Index types:
- trend_oracle: Reddit, HN, X trends and patterns
- croft_dna: Internal docs, brand notes, repo content
- publishing_graph: Workflow DAG for content pipelines
- service_schemas: API shapes, endpoints, limits
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Try FAISS
try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    np = None

# Try sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    ENCODER_AVAILABLE = True
except ImportError:
    ENCODER_AVAILABLE = False
    SentenceTransformer = None

# Default paths
DEFAULT_WORLD_PATH = Path(__file__).parent / "world"


# =============================================================================
# World Memory
# =============================================================================

class WorldMemory:
    """
    World layer - external knowledge embeddings.

    Manages multiple FAISS indexes:
    - trend_oracle: Trend patterns and insights
    - croft_dna: Internal documentation and notes
    - publishing_graph: Workflow information
    - service_schemas: API documentation

    For v1, this is largely stubbed - indexes will be built over time.
    """

    def __init__(self, path: Optional[str] = None):
        """
        Initialize world memory.

        Args:
            path: Path to world indexes directory
        """
        self.path = Path(path) if path else DEFAULT_WORLD_PATH

        # Indexes and metadata
        self._indexes: Dict[str, Any] = {}
        self._metadata: Dict[str, List[Dict]] = {}

        # Encoder (shared across indexes)
        self._encoder = None

        self._load()

    def _load(self):
        """Load available indexes."""
        if not self.path.exists():
            logger.info(f"WorldMemory: path {self.path} does not exist, running empty")
            return

        # Try to load encoder
        if ENCODER_AVAILABLE:
            try:
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"Failed to load encoder: {e}")

        # Try to load each index type
        for index_name in ["trend_oracle", "croft_dna", "publishing_graph", "service_schemas"]:
            self._load_index(index_name)

        logger.info(f"WorldMemory: loaded {len(self._indexes)} indexes")

    def _load_index(self, name: str):
        """Load a specific index and its metadata."""
        if not FAISS_AVAILABLE:
            return

        index_path = self.path / f"{name}.faiss"
        meta_path = self.path / f"{name}_meta.jsonl"

        # Load FAISS index
        if index_path.exists():
            try:
                self._indexes[name] = faiss.read_index(str(index_path))
            except Exception as e:
                logger.warning(f"Failed to load {name} index: {e}")
                return

        # Load metadata
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    self._metadata[name] = [json.loads(line) for line in f if line.strip()]
            except Exception as e:
                logger.warning(f"Failed to load {name} metadata: {e}")
                self._metadata[name] = []

    @property
    def index_count(self) -> int:
        """Number of loaded indexes."""
        return len(self._indexes)

    # =========================================================================
    # Retrieval
    # =========================================================================

    def retrieve_knowledge(
        self,
        user_msg: str,
        context_flags: Any,
        k: int = 5,
        indexes: Optional[List[str]] = None,
    ) -> Tuple[str, Dict]:
        """
        Retrieve relevant world knowledge.

        Args:
            user_msg: User's message
            context_flags: ContextFlags
            k: Number of items to retrieve per index
            indexes: Specific indexes to query (default: all)

        Returns:
            (context_string, metadata)
        """
        metadata = {"retrieved": [], "indexes_queried": []}

        if not self._indexes:
            return "", metadata

        if not self._encoder:
            return "", metadata

        # Determine which indexes to query
        target_indexes = indexes if indexes else list(self._indexes.keys())
        metadata["indexes_queried"] = target_indexes

        # Encode query
        try:
            query_emb = self._encoder.encode(
                [user_msg],
                normalize_embeddings=True
            ).astype("float32")
        except Exception as e:
            logger.warning(f"Failed to encode query: {e}")
            return "", metadata

        # Query each index
        all_results = []
        for index_name in target_indexes:
            results = self._query_index(index_name, query_emb, k)
            all_results.extend(results)

        # Sort by score and take top k
        all_results.sort(key=lambda x: x["score"], reverse=True)
        top_results = all_results[:k]

        # Format output
        if not top_results:
            return "", metadata

        lines = ["[WORLD CONTEXT]"]
        for r in top_results:
            lines.append(f"- {r['summary']} (source={r['source']}, score={r['score']:.2f})")
            metadata["retrieved"].append({
                "source": r["source"],
                "score": r["score"],
                "index": r["index"],
            })

        return "\n".join(lines), metadata

    def _query_index(
        self,
        index_name: str,
        query_emb: Any,
        k: int,
    ) -> List[Dict]:
        """Query a specific index."""
        if index_name not in self._indexes:
            return []

        index = self._indexes[index_name]
        meta = self._metadata.get(index_name, [])

        try:
            scores, indices = index.search(query_emb, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(meta):
                    continue

                item = meta[idx]
                results.append({
                    "summary": item.get("summary", item.get("title", "Unknown")),
                    "source": item.get("source", index_name),
                    "score": float(score),
                    "index": index_name,
                    "metadata": item,
                })

            return results

        except Exception as e:
            logger.warning(f"Failed to query {index_name}: {e}")
            return []

    # =========================================================================
    # Index Management
    # =========================================================================

    def build_index(
        self,
        name: str,
        items: List[Dict],
        text_field: str = "text",
        save: bool = True,
    ) -> bool:
        """
        Build a new index from items.

        Args:
            name: Index name
            items: List of items (must have text_field)
            text_field: Field to embed
            save: Whether to save to disk

        Returns:
            True if successful
        """
        if not FAISS_AVAILABLE or not self._encoder:
            logger.error("Cannot build index: FAISS or encoder not available")
            return False

        try:
            # Extract texts
            texts = [item.get(text_field, "") for item in items]
            if not texts:
                return False

            # Encode
            embeddings = self._encoder.encode(texts, normalize_embeddings=True)

            # Build index
            dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings.astype("float32"))

            # Store
            self._indexes[name] = index
            self._metadata[name] = items

            # Save if requested
            if save and self.path.exists():
                faiss.write_index(index, str(self.path / f"{name}.faiss"))
                with open(self.path / f"{name}_meta.jsonl", "w") as f:
                    for item in items:
                        f.write(json.dumps(item) + "\n")

            logger.info(f"Built index {name} with {len(items)} items")
            return True

        except Exception as e:
            logger.error(f"Failed to build index {name}: {e}")
            return False

    def add_to_index(
        self,
        name: str,
        items: List[Dict],
        text_field: str = "text",
    ) -> bool:
        """
        Add items to an existing index.

        Note: For simplicity, this rebuilds the index. For production,
        use IVF indexes with add_with_ids.
        """
        existing = self._metadata.get(name, [])
        all_items = existing + items
        return self.build_index(name, all_items, text_field, save=True)

    # =========================================================================
    # Specialized Retrievers
    # =========================================================================

    def get_trends(self, query: str, k: int = 5) -> List[Dict]:
        """Get relevant trends for a query."""
        if "trend_oracle" not in self._indexes:
            return []

        results, _ = self.retrieve_knowledge(
            user_msg=query,
            context_flags=None,
            k=k,
            indexes=["trend_oracle"],
        )
        return self._metadata.get("trend_oracle", [])[:k]

    def get_workflow(self, task: str) -> Optional[Dict]:
        """Get workflow for a task from publishing_graph."""
        # For now, return from metadata directly
        workflows = self._metadata.get("publishing_graph", [])
        for w in workflows:
            if w.get("task") == task or task in w.get("tags", []):
                return w
        return None

    def get_service_schema(self, service: str) -> Optional[Dict]:
        """Get API schema for a service."""
        schemas = self._metadata.get("service_schemas", [])
        for s in schemas:
            if s.get("service") == service or s.get("name") == service:
                return s
        return None


# =============================================================================
# Seed Data Builder
# =============================================================================

def build_seed_world(world_path: Optional[Path] = None):
    """
    Build initial world indexes with seed data.

    This creates starter indexes that can be expanded over time.
    """
    path = world_path or DEFAULT_WORLD_PATH
    path.mkdir(parents=True, exist_ok=True)

    world = WorldMemory(str(path))

    # Seed trend oracle with some starter items
    trend_seeds = [
        {
            "text": "AI agents and autonomous systems gaining traction",
            "summary": "AI agents trending in tech",
            "source": "tech_trends",
            "tags": ["ai", "agents", "automation"],
        },
        {
            "text": "Spiking neural networks for low-latency inference",
            "summary": "SNNs for edge AI",
            "source": "research",
            "tags": ["snn", "neuromorphic", "edge"],
        },
        {
            "text": "Self-publishing and creator economy tools",
            "summary": "Creator tools rising",
            "source": "business",
            "tags": ["publishing", "creator", "tools"],
        },
    ]

    # Seed publishing graph
    publishing_seeds = [
        {
            "text": "Twitter thread publishing workflow",
            "task": "twitter_thread",
            "steps": ["research", "outline", "draft", "review", "post"],
            "tags": ["twitter", "social", "thread"],
        },
        {
            "text": "Newsletter publishing workflow",
            "task": "newsletter",
            "steps": ["research", "draft", "edit", "schedule", "send"],
            "tags": ["email", "newsletter"],
        },
        {
            "text": "Blog post publishing workflow",
            "task": "blog_post",
            "steps": ["research", "outline", "draft", "edit", "publish"],
            "tags": ["blog", "content"],
        },
    ]

    # Seed service schemas
    service_seeds = [
        {
            "text": "GitHub API for repository management",
            "service": "github",
            "name": "GitHub API",
            "base_url": "https://api.github.com",
            "rate_limit": "5000/hour authenticated",
            "tags": ["git", "code", "repos"],
        },
        {
            "text": "Twitter/X API for posting and reading",
            "service": "twitter",
            "name": "Twitter API v2",
            "base_url": "https://api.twitter.com/2",
            "rate_limit": "varies by endpoint",
            "tags": ["social", "posting"],
        },
    ]

    # Build indexes
    if FAISS_AVAILABLE and ENCODER_AVAILABLE:
        world.build_index("trend_oracle", trend_seeds)
        world.build_index("publishing_graph", publishing_seeds)
        world.build_index("service_schemas", service_seeds)
        logger.info("Built seed world indexes")
    else:
        # Just save metadata without FAISS
        for name, items in [
            ("trend_oracle", trend_seeds),
            ("publishing_graph", publishing_seeds),
            ("service_schemas", service_seeds),
        ]:
            with open(path / f"{name}_meta.jsonl", "w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")
        logger.info("Saved seed metadata (no FAISS)")
