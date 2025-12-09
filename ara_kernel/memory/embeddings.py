"""
Embedding Index
================

Vector similarity search for Ara's memory.
Stubbed for v0.1; ready for FAISS/Qdrant integration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """A single embedding search result."""
    text: str
    score: float
    meta: Dict[str, Any]
    doc_id: Optional[str] = None


class EmbeddingIndex:
    """
    Vector embedding index for semantic search.

    v0.1 implementation: Simple in-memory list with stub similarity.
    Later: Back with FAISS, Qdrant, or similar.
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._items: List[Tuple[str, List[float], Dict[str, Any]]] = []
        self._id_counter = 0

    def add(
        self,
        text: str,
        embedding: Optional[List[float]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a text to the index.

        Args:
            text: The text to index
            embedding: Pre-computed embedding (optional, will be computed if None)
            meta: Metadata to store

        Returns:
            Document ID
        """
        if embedding is None:
            # Stub: use text hash as fake embedding
            embedding = self._stub_embed(text)

        doc_id = f"doc_{self._id_counter}"
        self._id_counter += 1

        self._items.append((text, embedding, meta or {"doc_id": doc_id}))
        logger.debug(f"Added embedding for doc {doc_id}: {text[:50]}...")
        return doc_id

    def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        k: int = 5,
    ) -> List[EmbeddingResult]:
        """
        Search for similar documents.

        Args:
            query: Query text
            query_embedding: Pre-computed query embedding (optional)
            k: Number of results

        Returns:
            List of EmbeddingResult
        """
        if not self._items:
            return []

        if query_embedding is None:
            query_embedding = self._stub_embed(query)

        # Compute similarities
        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for text, emb, meta in self._items:
            score = self._cosine_similarity(query_embedding, emb)
            scored.append((score, text, meta))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top k
        results = []
        for score, text, meta in scored[:k]:
            results.append(EmbeddingResult(
                text=text,
                score=score,
                meta=meta,
                doc_id=meta.get("doc_id"),
            ))

        return results

    def _stub_embed(self, text: str) -> List[float]:
        """
        Stub embedding function.
        In production, use sentence-transformers or similar.
        """
        import hashlib

        # Use hash to create a deterministic fake embedding
        h = hashlib.sha256(text.encode()).hexdigest()
        embedding = []
        for i in range(0, min(len(h), self.dim * 2), 2):
            val = int(h[i:i+2], 16) / 255.0 - 0.5
            embedding.append(val)

        # Pad or truncate to dim
        while len(embedding) < self.dim:
            embedding.append(0.0)
        embedding = embedding[:self.dim]

        return embedding

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def count(self) -> int:
        """Number of items in the index."""
        return len(self._items)

    def save(self, path: Path) -> None:
        """Save index to file (JSON for stub, binary for FAISS later)."""
        data = {
            "dim": self.dim,
            "items": [
                {"text": text, "embedding": emb, "meta": meta}
                for text, emb, meta in self._items
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        logger.info(f"Saved embedding index to {path}")

    def load(self, path: Path) -> None:
        """Load index from file."""
        if not path.exists():
            logger.warning(f"Embedding index not found: {path}")
            return

        with open(path) as f:
            data = json.load(f)

        self.dim = data.get("dim", 384)
        self._items = [
            (item["text"], item["embedding"], item["meta"])
            for item in data.get("items", [])
        ]
        logger.info(f"Loaded embedding index from {path} ({len(self._items)} items)")


class FAISSIndex:
    """
    FAISS-backed embedding index for production use.
    Requires: pip install faiss-cpu (or faiss-gpu)
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._index = None
        self._texts: List[str] = []
        self._metas: List[Dict[str, Any]] = []
        self._initialized = False

        try:
            import faiss
            self._faiss = faiss
            self._index = faiss.IndexFlatIP(dim)  # Inner product (cosine with normalized vectors)
            self._initialized = True
            logger.info(f"FAISSIndex initialized with dim={dim}")
        except ImportError:
            logger.warning("FAISS not available, using stub index")

    def add(
        self,
        text: str,
        embedding: List[float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Add embedding to FAISS index."""
        if not self._initialized:
            return -1

        import numpy as np

        # Normalize for cosine similarity
        emb_array = np.array([embedding], dtype=np.float32)
        self._faiss.normalize_L2(emb_array)

        self._index.add(emb_array)
        self._texts.append(text)
        self._metas.append(meta or {})

        return len(self._texts) - 1

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
    ) -> List[EmbeddingResult]:
        """Search FAISS index."""
        if not self._initialized or self._index.ntotal == 0:
            return []

        import numpy as np

        q_array = np.array([query_embedding], dtype=np.float32)
        self._faiss.normalize_L2(q_array)

        scores, indices = self._index.search(q_array, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(EmbeddingResult(
                text=self._texts[idx],
                score=float(score),
                meta=self._metas[idx],
                doc_id=f"faiss_{idx}",
            ))

        return results

    def count(self) -> int:
        """Number of vectors in index."""
        if not self._initialized:
            return 0
        return self._index.ntotal
