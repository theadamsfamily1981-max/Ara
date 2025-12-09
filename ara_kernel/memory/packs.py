"""
Knowledge Packs
================

Structured knowledge bundles for Ara.
Packs contain documents, schemas, and metadata.
"""

from __future__ import annotations

import json
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document within a knowledge pack."""
    id: str
    text: str
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgePack:
    """
    A knowledge pack containing related documents.

    Packs can be:
    - Publishing knowledge (KDP rules, market trends)
    - Technical knowledge (API docs, code patterns)
    - Domain knowledge (quantum computing, hardware)
    """
    name: str
    version: str
    description: str
    documents: List[Document]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def search_tags(self, tags: List[str]) -> List[Document]:
        """Find documents matching any of the given tags."""
        tag_set = set(tags)
        return [doc for doc in self.documents if tag_set & set(doc.tags)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metadata": self.metadata,
            "documents": [
                {
                    "id": doc.id,
                    "text": doc.text,
                    "tags": doc.tags,
                    "meta": doc.meta,
                }
                for doc in self.documents
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgePack:
        """Create from dictionary."""
        documents = [
            Document(
                id=doc["id"],
                text=doc["text"],
                tags=doc.get("tags", []),
                meta=doc.get("meta", {}),
            )
            for doc in data.get("documents", [])
        ]
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            documents=documents,
            metadata=data.get("metadata", {}),
        )


def load_pack(path: Path) -> KnowledgePack:
    """
    Load a knowledge pack from file.

    Supports:
    - .json files (single JSON object)
    - .zip files (meta.json + docs.json)
    """
    if path.suffix == ".zip":
        return _load_pack_zip(path)
    elif path.suffix == ".json":
        return _load_pack_json(path)
    else:
        raise ValueError(f"Unsupported pack format: {path.suffix}")


def _load_pack_json(path: Path) -> KnowledgePack:
    """Load pack from JSON file."""
    with open(path) as f:
        data = json.load(f)
    pack = KnowledgePack.from_dict(data)
    logger.info(f"Loaded pack '{pack.name}' v{pack.version} ({len(pack.documents)} docs)")
    return pack


def _load_pack_zip(path: Path) -> KnowledgePack:
    """Load pack from ZIP archive."""
    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("meta.json"))
        docs_data = json.loads(zf.read("docs.json"))

    documents = [
        Document(
            id=doc["id"],
            text=doc["text"],
            tags=doc.get("tags", []),
            meta=doc.get("meta", {}),
        )
        for doc in docs_data
    ]

    pack = KnowledgePack(
        name=meta["name"],
        version=meta.get("version", "1.0.0"),
        description=meta.get("description", ""),
        documents=documents,
        metadata=meta,
    )
    logger.info(f"Loaded pack '{pack.name}' v{pack.version} ({len(pack.documents)} docs)")
    return pack


def save_pack(pack: KnowledgePack, path: Path, format: str = "json") -> None:
    """
    Save a knowledge pack to file.

    Args:
        pack: The pack to save
        path: Output path
        format: "json" or "zip"
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(path, "w") as f:
            json.dump(pack.to_dict(), f, indent=2)
    elif format == "zip":
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            meta = {
                "name": pack.name,
                "version": pack.version,
                "description": pack.description,
                **pack.metadata,
            }
            zf.writestr("meta.json", json.dumps(meta, indent=2))

            docs = [
                {"id": doc.id, "text": doc.text, "tags": doc.tags, "meta": doc.meta}
                for doc in pack.documents
            ]
            zf.writestr("docs.json", json.dumps(docs, indent=2))
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"Saved pack '{pack.name}' to {path}")


class PackManager:
    """
    Manages multiple knowledge packs.
    """

    def __init__(self, packs_dir: Path) -> None:
        self.packs_dir = packs_dir
        self.packs: Dict[str, KnowledgePack] = {}

    def load_all(self) -> int:
        """Load all packs from the packs directory."""
        if not self.packs_dir.exists():
            logger.warning(f"Packs directory not found: {self.packs_dir}")
            return 0

        count = 0
        for path in self.packs_dir.glob("*.json"):
            try:
                pack = load_pack(path)
                self.packs[pack.name] = pack
                count += 1
            except Exception as e:
                logger.error(f"Failed to load pack {path}: {e}")

        for path in self.packs_dir.glob("*.zip"):
            try:
                pack = load_pack(path)
                self.packs[pack.name] = pack
                count += 1
            except Exception as e:
                logger.error(f"Failed to load pack {path}: {e}")

        logger.info(f"Loaded {count} knowledge packs")
        return count

    def get(self, name: str) -> Optional[KnowledgePack]:
        """Get a pack by name."""
        return self.packs.get(name)

    def search_all(self, tags: List[str]) -> List[Document]:
        """Search all packs for documents with given tags."""
        results = []
        for pack in self.packs.values():
            results.extend(pack.search_tags(tags))
        return results

    def list_packs(self) -> List[str]:
        """List all loaded pack names."""
        return list(self.packs.keys())
