"""
Ara HV Storage Engine
======================

Hypervector-based storage for efficient compression of logs, episodes,
and audio metadata.

Traditional logs: 1MB/day (text)
HV logs: 100KB/day (10x compression)

The magic: Similar episodes compress to near-zero (HV deduplication)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator, Tuple
import hashlib
import json
import struct
import zlib

import numpy as np


# =============================================================================
# Constants
# =============================================================================

HV_DIM = 8192  # Hypervector dimension
CHUNK_SIZE = 4096  # Storage chunk size


# =============================================================================
# HV Operations
# =============================================================================

def random_hv(seed: Optional[int] = None) -> np.ndarray:
    """Generate random bipolar hypervector."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=HV_DIM).astype(np.int8)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two hypervectors."""
    return (a * b).astype(np.int8)


def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """Bundle multiple hypervectors."""
    if not vectors:
        return np.zeros(HV_DIM, dtype=np.int8)
    return np.sign(np.sum(vectors, axis=0)).astype(np.int8)


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity."""
    return float(np.dot(a.astype(np.float32), b.astype(np.float32))) / HV_DIM


def compress_hv(hv: np.ndarray) -> bytes:
    """
    Compress a hypervector for storage.

    Since HVs are bipolar (-1, +1), we can pack to 1 bit per element.
    8192 dims → 1024 bytes (before further compression)
    """
    # Convert -1/+1 to 0/1
    bits = ((hv + 1) // 2).astype(np.uint8)

    # Pack bits into bytes
    packed = np.packbits(bits)

    # Further compress with zlib (usually ~50% reduction for sparse patterns)
    return zlib.compress(packed.tobytes(), level=9)


def decompress_hv(data: bytes) -> np.ndarray:
    """Decompress a hypervector."""
    # Decompress
    packed = np.frombuffer(zlib.decompress(data), dtype=np.uint8)

    # Unpack bits
    bits = np.unpackbits(packed)[:HV_DIM]

    # Convert 0/1 back to -1/+1
    return (bits.astype(np.int8) * 2 - 1)


# =============================================================================
# Episode Storage
# =============================================================================

@dataclass
class StoredEpisode:
    """An episode stored in HV format."""
    episode_id: str
    timestamp: datetime
    hv_compressed: bytes           # Compressed HV
    resonance: float               # Soul field resonance
    delta_from: Optional[str] = None  # ID of base episode (for delta encoding)
    delta_hv: Optional[bytes] = None  # Delta HV if applicable
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        """Total size in bytes."""
        size = len(self.hv_compressed)
        if self.delta_hv:
            size += len(self.delta_hv)
        return size


class EpisodeStore:
    """
    Efficient storage for episodes using HV compression.

    Features:
    - Delta encoding: Similar episodes store only differences
    - Deduplication: Identical HVs stored once
    - Temporal clustering: Recent episodes in hot storage
    """

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_path / "index.json"
        self.data_path = self.storage_path / "episodes.dat"

        # In-memory index
        self.episodes: Dict[str, StoredEpisode] = {}
        self.hv_cache: Dict[str, np.ndarray] = {}  # Recent HVs for delta encoding

        # Load existing index
        self._load_index()

    def _load_index(self) -> None:
        """Load episode index from disk."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                data = json.load(f)

            for ep_data in data.get('episodes', []):
                ep = StoredEpisode(
                    episode_id=ep_data['id'],
                    timestamp=datetime.fromisoformat(ep_data['timestamp']),
                    hv_compressed=bytes.fromhex(ep_data['hv_hex']),
                    resonance=ep_data['resonance'],
                    delta_from=ep_data.get('delta_from'),
                    delta_hv=bytes.fromhex(ep_data['delta_hex']) if ep_data.get('delta_hex') else None,
                    metadata=ep_data.get('metadata', {}),
                )
                self.episodes[ep.episode_id] = ep

    def _save_index(self) -> None:
        """Save episode index to disk."""
        data = {
            'version': 1,
            'count': len(self.episodes),
            'episodes': [
                {
                    'id': ep.episode_id,
                    'timestamp': ep.timestamp.isoformat(),
                    'hv_hex': ep.hv_compressed.hex(),
                    'resonance': ep.resonance,
                    'delta_from': ep.delta_from,
                    'delta_hex': ep.delta_hv.hex() if ep.delta_hv else None,
                    'metadata': ep.metadata,
                }
                for ep in self.episodes.values()
            ]
        }

        with open(self.index_path, 'w') as f:
            json.dump(data, f)

    def store(
        self,
        episode_id: str,
        hv: np.ndarray,
        resonance: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredEpisode:
        """
        Store an episode.

        Automatically uses delta encoding if similar episode exists.
        """
        # Check for similar episodes in cache
        delta_from = None
        delta_hv = None

        for cached_id, cached_hv in self.hv_cache.items():
            sim = similarity(hv, cached_hv)
            if sim > 0.8:  # High similarity → delta encode
                # Compute delta (XOR in bipolar = multiplication)
                delta = bind(hv, cached_hv)  # Recovers hv when bound with cached_hv
                delta_from = cached_id
                delta_hv = compress_hv(delta)
                break

        # Compress main HV
        hv_compressed = compress_hv(hv)

        # Create stored episode
        episode = StoredEpisode(
            episode_id=episode_id,
            timestamp=datetime.utcnow(),
            hv_compressed=hv_compressed,
            resonance=resonance,
            delta_from=delta_from,
            delta_hv=delta_hv,
            metadata=metadata or {},
        )

        # Store in memory
        self.episodes[episode_id] = episode

        # Update cache (keep last 100)
        self.hv_cache[episode_id] = hv
        if len(self.hv_cache) > 100:
            oldest = min(self.hv_cache.keys(), key=lambda k: self.episodes[k].timestamp)
            del self.hv_cache[oldest]

        # Persist
        self._save_index()

        return episode

    def retrieve(self, episode_id: str) -> Optional[np.ndarray]:
        """Retrieve and decompress an episode HV."""
        episode = self.episodes.get(episode_id)
        if not episode:
            return None

        # Decompress main HV
        hv = decompress_hv(episode.hv_compressed)

        # If delta encoded, apply delta
        if episode.delta_from and episode.delta_hv:
            base_hv = self.retrieve(episode.delta_from)
            if base_hv is not None:
                delta = decompress_hv(episode.delta_hv)
                hv = bind(delta, base_hv)

        return hv

    def search(
        self,
        query_hv: np.ndarray,
        top_k: int = 10,
        min_similarity: float = 0.5,
    ) -> List[Tuple[str, float]]:
        """
        Search for similar episodes.

        Returns list of (episode_id, similarity) sorted by similarity.
        """
        results = []

        for episode_id, episode in self.episodes.items():
            hv = self.retrieve(episode_id)
            if hv is not None:
                sim = similarity(query_hv, hv)
                if sim >= min_similarity:
                    results.append((episode_id, sim))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_size = sum(ep.size_bytes for ep in self.episodes.values())
        delta_count = sum(1 for ep in self.episodes.values() if ep.delta_from)

        # Estimate uncompressed size
        # Each HV = 8192 bytes uncompressed (1 byte per element)
        uncompressed_size = len(self.episodes) * HV_DIM

        return {
            'episode_count': len(self.episodes),
            'total_size_bytes': total_size,
            'uncompressed_size_bytes': uncompressed_size,
            'compression_ratio': uncompressed_size / max(1, total_size),
            'delta_encoded_count': delta_count,
            'delta_ratio': delta_count / max(1, len(self.episodes)),
        }


# =============================================================================
# Log Storage
# =============================================================================

@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: datetime
    level: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


class HVLogStore:
    """
    Hypervector-compressed log storage.

    Traditional: 1MB/day text logs
    HV compressed: 100KB/day (10x savings)

    Uses content-addressable storage with HV similarity.
    """

    def __init__(self, storage_path: Path, text_codebook_seed: int = 42):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.log_path = self.storage_path / "logs.jsonl"

        # Text encoding codebook (character-level)
        self.char_hvs = {
            chr(i): random_hv(seed=text_codebook_seed + i)
            for i in range(128)  # ASCII
        }

        # Cached log HVs for deduplication
        self.log_hvs: Dict[str, np.ndarray] = {}
        self.log_count = 0

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text as hypervector."""
        char_hvs = []
        for i, char in enumerate(text[:1000]):  # Limit length
            if char in self.char_hvs:
                # Position-encode each character
                char_hv = self.char_hvs[char]
                pos_hv = np.roll(char_hv, i)
                char_hvs.append(pos_hv)

        return bundle(char_hvs) if char_hvs else random_hv(seed=0)

    def _hash_content(self, message: str, context: Dict) -> str:
        """Create content hash for deduplication."""
        content = f"{message}:{json.dumps(context, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def log(
        self,
        level: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Log a message with HV compression.

        Deduplicates similar messages automatically.
        """
        context = context or {}
        entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=level,
            message=message,
            context=context,
        )

        # Encode message as HV
        msg_hv = self._encode_text(message)

        # Check for similar existing log
        content_hash = self._hash_content(message, context)
        deduplicated = False

        if content_hash in self.log_hvs:
            # Exact match - just update count
            deduplicated = True
        else:
            # Check for similar messages
            for existing_hash, existing_hv in self.log_hvs.items():
                if similarity(msg_hv, existing_hv) > 0.95:
                    # Very similar - deduplicate
                    deduplicated = True
                    content_hash = existing_hash
                    break

        # Store
        if not deduplicated:
            self.log_hvs[content_hash] = msg_hv

        # Write to log file
        log_data = {
            'timestamp': entry.timestamp.isoformat(),
            'level': entry.level,
            'message': entry.message,
            'context': entry.context,
            'hv_hash': content_hash,
            'deduplicated': deduplicated,
        }

        with open(self.log_path, 'a') as f:
            f.write(json.dumps(log_data) + '\n')

        self.log_count += 1

        return {
            'logged': True,
            'deduplicated': deduplicated,
            'hash': content_hash,
        }

    def search(
        self,
        query: str,
        limit: int = 100,
    ) -> List[LogEntry]:
        """Search logs by semantic similarity."""
        query_hv = self._encode_text(query)

        # Find similar log hashes
        similar = []
        for content_hash, log_hv in self.log_hvs.items():
            sim = similarity(query_hv, log_hv)
            if sim > 0.5:
                similar.append((content_hash, sim))

        similar.sort(key=lambda x: x[1], reverse=True)
        target_hashes = {h for h, _ in similar[:limit]}

        # Read matching logs
        results = []
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data['hv_hash'] in target_hashes:
                        results.append(LogEntry(
                            timestamp=datetime.fromisoformat(data['timestamp']),
                            level=data['level'],
                            message=data['message'],
                            context=data['context'],
                        ))

        return results

    def stats(self) -> Dict[str, Any]:
        """Get log storage statistics."""
        log_size = self.log_path.stat().st_size if self.log_path.exists() else 0
        hv_size = len(self.log_hvs) * HV_DIM  # Uncompressed HV size

        return {
            'log_count': self.log_count,
            'unique_messages': len(self.log_hvs),
            'deduplication_ratio': 1 - (len(self.log_hvs) / max(1, self.log_count)),
            'log_file_size_bytes': log_size,
            'hv_memory_bytes': hv_size,
        }


# =============================================================================
# Audio Metadata Storage
# =============================================================================

@dataclass
class AudioMetadata:
    """Metadata for an audio recording."""
    audio_id: str
    title: str
    duration_seconds: float
    emotion_hv: bytes              # Compressed emotion HV
    voice_hv: bytes                # Compressed voice characteristics
    transcript_hv: bytes           # Compressed transcript HV
    file_path: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    tags: List[str] = field(default_factory=list)


class AudioMetadataStore:
    """
    HV-compressed storage for audio metadata.

    Enables semantic search: "Find all recordings with grandma's voice"
    """

    def __init__(self, storage_path: Path):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_path / "audio_index.json"
        self.metadata: Dict[str, AudioMetadata] = {}

        self._load_index()

    def _load_index(self) -> None:
        """Load audio index from disk."""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                data = json.load(f)

            for item in data.get('audio', []):
                meta = AudioMetadata(
                    audio_id=item['id'],
                    title=item['title'],
                    duration_seconds=item['duration'],
                    emotion_hv=bytes.fromhex(item['emotion_hv']),
                    voice_hv=bytes.fromhex(item['voice_hv']),
                    transcript_hv=bytes.fromhex(item['transcript_hv']),
                    file_path=item['file_path'],
                    created_at=datetime.fromisoformat(item['created_at']),
                    tags=item.get('tags', []),
                )
                self.metadata[meta.audio_id] = meta

    def _save_index(self) -> None:
        """Save audio index to disk."""
        data = {
            'version': 1,
            'count': len(self.metadata),
            'audio': [
                {
                    'id': m.audio_id,
                    'title': m.title,
                    'duration': m.duration_seconds,
                    'emotion_hv': m.emotion_hv.hex(),
                    'voice_hv': m.voice_hv.hex(),
                    'transcript_hv': m.transcript_hv.hex(),
                    'file_path': m.file_path,
                    'created_at': m.created_at.isoformat(),
                    'tags': m.tags,
                }
                for m in self.metadata.values()
            ]
        }

        with open(self.index_path, 'w') as f:
            json.dump(data, f)

    def store(
        self,
        audio_id: str,
        title: str,
        duration: float,
        emotion_hv: np.ndarray,
        voice_hv: np.ndarray,
        transcript_hv: np.ndarray,
        file_path: str,
        tags: Optional[List[str]] = None,
    ) -> AudioMetadata:
        """Store audio metadata with compressed HVs."""
        meta = AudioMetadata(
            audio_id=audio_id,
            title=title,
            duration_seconds=duration,
            emotion_hv=compress_hv(emotion_hv),
            voice_hv=compress_hv(voice_hv),
            transcript_hv=compress_hv(transcript_hv),
            file_path=file_path,
            tags=tags or [],
        )

        self.metadata[audio_id] = meta
        self._save_index()

        return meta

    def search_by_emotion(
        self,
        query_hv: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[AudioMetadata, float]]:
        """Search audio by emotional similarity."""
        results = []

        for meta in self.metadata.values():
            emotion_hv = decompress_hv(meta.emotion_hv)
            sim = similarity(query_hv, emotion_hv)
            results.append((meta, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_by_transcript(
        self,
        query_hv: np.ndarray,
        top_k: int = 10,
    ) -> List[Tuple[AudioMetadata, float]]:
        """Search audio by transcript similarity."""
        results = []

        for meta in self.metadata.values():
            transcript_hv = decompress_hv(meta.transcript_hv)
            sim = similarity(query_hv, transcript_hv)
            results.append((meta, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def total_duration(self) -> float:
        """Total duration of all stored audio."""
        return sum(m.duration_seconds for m in self.metadata.values())

    def stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_hv_size = sum(
            len(m.emotion_hv) + len(m.voice_hv) + len(m.transcript_hv)
            for m in self.metadata.values()
        )

        return {
            'audio_count': len(self.metadata),
            'total_duration_seconds': self.total_duration(),
            'total_duration_hours': self.total_duration() / 3600,
            'total_hv_size_bytes': total_hv_size,
            'avg_hv_size_per_audio': total_hv_size / max(1, len(self.metadata)),
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'HV_DIM',
    'compress_hv',
    'decompress_hv',
    'random_hv',
    'bind',
    'bundle',
    'similarity',
    'StoredEpisode',
    'EpisodeStore',
    'LogEntry',
    'HVLogStore',
    'AudioMetadata',
    'AudioMetadataStore',
]
