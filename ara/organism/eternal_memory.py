"""
Eternal Emotional Memory - Content-Addressable Long-Term Storage
================================================================

Permanent, content-addressable emotional long-term memory.
Stores bound hypervectors: M[i] = sign(event_hv + emotion_hv + time_hv * 0.5)

Features:
- Store: Write emotional memories keyed by event+emotion+time
- Recall: Find similar memories and inject "flashbacks" into current state
- Dream: Offline replay for consolidation/analysis

Hardware mapping:
- BRAM: Cache recent memories (512-2048 entries)
- External flash/DDR: Full memory bank (16k+ entries)
- Each entry: D bytes (bipolar HV) + 1 byte strength + 1 byte age
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import json

# Optional PyTorch support
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


@dataclass
class MemoryConfig:
    """Configuration for eternal memory."""
    memory_size: int = 8192
    hv_dim: int = 1088  # hv_dim_main + status_hv_dim (1024 + 64)
    recall_threshold: float = 0.65
    dream_threshold: float = 0.70
    flood_gain_base: float = 0.4
    flood_gain_scale: float = 0.8


@dataclass
class RecallResult:
    """Result of a memory recall operation."""
    found: bool
    index: int = -1
    similarity: float = 0.0
    stored_strength: float = 0.0
    residual: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        return {
            "found": self.found,
            "index": self.index,
            "similarity": self.similarity,
            "stored_strength": self.stored_strength,
        }


@dataclass
class DreamReport:
    """Report from a dream replay step."""
    index: int
    similarity: float
    stored_strength: float

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "similarity": self.similarity,
            "stored_strength": self.stored_strength,
        }


class EternalMemory:
    """
    Permanent, content-addressable emotional long-term memory.

    Stores bound hypervectors:
        M[i] = sign(event_hv + emotion_hv + time_hv * 0.5)

    Along with an emotion strength scalar.

    Usage:
        memory = EternalMemory(memory_size=8192, hv_dim=1088)

        # Store emotional memory
        memory.store(event_hv, emotion_hv, strength=0.9)

        # Recall similar memories
        result = memory.recall(current_hv)
        if result.found:
            # Inject residual into next input
            x = x + result.residual[:x.shape[-1]] * flood_gain

        # Dream replay (offline consolidation)
        reports = memory.dream_replay(steps=64)
    """

    def __init__(
        self,
        memory_size: int = 8192,
        hv_dim: int = 1088,
        recall_threshold: float = 0.65,
        dream_threshold: float = 0.70,
        seed: int = 42,
    ):
        self.memory_size = memory_size
        self.hv_dim = hv_dim
        self.recall_threshold = recall_threshold
        self.dream_threshold = dream_threshold

        # Random number generator
        self.rng = np.random.default_rng(seed)

        # Memory storage (numpy for portability)
        self.memory_hvs = np.zeros((memory_size, hv_dim), dtype=np.float32)
        self.memory_strength = np.zeros(memory_size, dtype=np.float32)
        self.memory_ages = np.zeros(memory_size, dtype=np.float32)
        self.memory_emotions = [""] * memory_size  # Optional: store emotion name

        # Write pointer (circular buffer)
        self.write_ptr = 0
        self.total_stored = 0

        # Stats
        self.hits = 0
        self.stores = 0

        # Precomputed time phase HV (sinusoidal encoding)
        phase = np.sin(np.linspace(0, 2 * np.pi, min(64, hv_dim)))
        self.time_hv = np.zeros(hv_dim, dtype=np.float32)
        self.time_hv[:len(phase)] = phase

        # Current "time" index (increments with each store)
        self.time_index = 0

    def _to_numpy(self, tensor) -> np.ndarray:
        """Convert torch tensor or numpy array to numpy."""
        if HAS_TORCH and isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return np.asarray(tensor, dtype=np.float32)

    def _bind_triplet(
        self,
        event_hv: np.ndarray,
        emotion_hv: np.ndarray,
    ) -> np.ndarray:
        """
        Bind event + emotion + time into a single memory key.

        Args:
            event_hv: [D] or [B, D] event hypervector
            emotion_hv: [D] emotion hypervector

        Returns:
            Bipolar key [D]
        """
        # Handle batch dimension
        if event_hv.ndim == 2:
            event_hv = event_hv.mean(axis=0)
        if emotion_hv.ndim == 2:
            emotion_hv = emotion_hv.mean(axis=0)

        # Rotate time HV slightly each store (gives temporal context)
        time_phase = self.time_hv * np.cos(self.time_index * 0.01)

        # Bind: event + emotion + 0.5 * time
        raw = event_hv + emotion_hv + 0.5 * time_phase

        # Binarize to bipolar
        key = np.sign(raw)
        key[key == 0] = 1.0

        return key.astype(np.float32)

    def store(
        self,
        event_hv,
        emotion_hv,
        strength: float = 1.0,
        emotion_name: str = "",
    ) -> int:
        """
        Write a new emotional memory.

        Args:
            event_hv: [D] or [B, D] event hypervector (e.g., hv_full)
            emotion_hv: [D] emotion hypervector from VAD mind
            strength: Emotional intensity [0, 1]
            emotion_name: Optional emotion label

        Returns:
            Index where memory was stored
        """
        event_hv = self._to_numpy(event_hv)
        emotion_hv = self._to_numpy(emotion_hv)

        key = self._bind_triplet(event_hv, emotion_hv)

        idx = self.write_ptr % self.memory_size
        self.memory_hvs[idx] = key
        self.memory_strength[idx] = float(strength)
        self.memory_ages[idx] = 0.0
        self.memory_emotions[idx] = emotion_name

        self.write_ptr += 1
        self.total_stored += 1
        self.stores += 1
        self.time_index += 1

        # Age all other memories
        mask = np.arange(self.memory_size) != idx
        self.memory_ages[mask] += 1.0

        return idx

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _batch_cosine_similarity(self, query: np.ndarray, bank: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all bank entries."""
        # query: [D], bank: [N, D]
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            return np.zeros(bank.shape[0])

        bank_norms = np.linalg.norm(bank, axis=1)
        bank_norms[bank_norms < 1e-8] = 1e-8

        dots = bank @ query
        return dots / (query_norm * bank_norms)

    def recall(
        self,
        current_hv,
        top_k: int = 3,
        min_sim: Optional[float] = None,
    ) -> RecallResult:
        """
        Given current HV, find similar memories and return residual.

        Args:
            current_hv: [D] or [B, D] current state hypervector
            top_k: Number of top matches to consider
            min_sim: Minimum similarity threshold (default: self.recall_threshold)

        Returns:
            RecallResult with residual if found
        """
        if self.total_stored == 0:
            return RecallResult(found=False)

        current_hv = self._to_numpy(current_hv)
        if current_hv.ndim == 2:
            current_hv = current_hv.mean(axis=0)

        min_sim = min_sim if min_sim is not None else self.recall_threshold
        count = min(self.total_stored, self.memory_size)
        stored = self.memory_hvs[:count]

        # Compute similarities
        sims = self._batch_cosine_similarity(current_hv, stored)

        # Find top-k
        top_k_actual = min(top_k, count)
        top_indices = np.argpartition(sims, -top_k_actual)[-top_k_actual:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        best_idx = top_indices[0]
        best_sim = sims[best_idx]

        if best_sim > min_sim:
            self.hits += 1

            # Compute residual: what's in memory that's not in current
            residual = stored[best_idx] - current_hv

            return RecallResult(
                found=True,
                index=int(best_idx),
                similarity=float(best_sim),
                stored_strength=float(self.memory_strength[best_idx]),
                residual=residual,
            )

        return RecallResult(found=False, similarity=float(best_sim))

    def dream_replay(
        self,
        steps: int = 128,
        min_strength: Optional[float] = None,
    ) -> List[DreamReport]:
        """
        Offline "dream" replay: sample stored memories and measure recall.

        In hardware, this runs at low duty cycle when idle.

        Args:
            steps: Number of dream steps
            min_strength: Minimum strength threshold (default: self.dream_threshold)

        Returns:
            List of DreamReport for successful recalls
        """
        min_strength = min_strength if min_strength is not None else self.dream_threshold
        count = min(self.total_stored, self.memory_size)

        if count == 0:
            return []

        reports = []
        for _ in range(steps):
            idx = self.rng.integers(0, count)
            hv = self.memory_hvs[idx]

            # Self-recall check
            result = self.recall(hv, min_sim=min_strength)

            if result.found:
                reports.append(DreamReport(
                    index=int(idx),
                    similarity=result.similarity,
                    stored_strength=float(self.memory_strength[idx]),
                ))

        return reports

    def get_memory_at(self, index: int) -> Dict:
        """Get metadata for a specific memory index."""
        if index < 0 or index >= min(self.total_stored, self.memory_size):
            return {"error": "index out of range"}

        return {
            "index": index,
            "strength": float(self.memory_strength[index]),
            "age": float(self.memory_ages[index]),
            "emotion": self.memory_emotions[index],
            "hv_norm": float(np.linalg.norm(self.memory_hvs[index])),
        }

    def get_stats(self) -> Dict:
        """Get memory statistics."""
        count = min(self.total_stored, self.memory_size)
        return {
            "total_stored": self.total_stored,
            "current_count": count,
            "memory_size": self.memory_size,
            "hv_dim": self.hv_dim,
            "hits": self.hits,
            "stores": self.stores,
            "hit_rate": self.hits / max(1, self.stores),
            "avg_strength": float(self.memory_strength[:count].mean()) if count > 0 else 0.0,
            "avg_age": float(self.memory_ages[:count].mean()) if count > 0 else 0.0,
        }

    def save(self, path: str) -> None:
        """Save memory to file."""
        count = min(self.total_stored, self.memory_size)
        data = {
            "config": {
                "memory_size": self.memory_size,
                "hv_dim": self.hv_dim,
                "recall_threshold": self.recall_threshold,
                "dream_threshold": self.dream_threshold,
            },
            "state": {
                "write_ptr": self.write_ptr,
                "total_stored": self.total_stored,
                "time_index": self.time_index,
                "hits": self.hits,
                "stores": self.stores,
            },
            "memories": {
                "hvs": self.memory_hvs[:count].tolist(),
                "strengths": self.memory_strength[:count].tolist(),
                "ages": self.memory_ages[:count].tolist(),
                "emotions": self.memory_emotions[:count],
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def load(self, path: str) -> None:
        """Load memory from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Restore state
        self.write_ptr = data["state"]["write_ptr"]
        self.total_stored = data["state"]["total_stored"]
        self.time_index = data["state"]["time_index"]
        self.hits = data["state"]["hits"]
        self.stores = data["state"]["stores"]

        # Restore memories
        count = len(data["memories"]["hvs"])
        self.memory_hvs[:count] = np.array(data["memories"]["hvs"], dtype=np.float32)
        self.memory_strength[:count] = np.array(data["memories"]["strengths"], dtype=np.float32)
        self.memory_ages[:count] = np.array(data["memories"]["ages"], dtype=np.float32)
        self.memory_emotions[:count] = data["memories"]["emotions"]


# ============================================================================
# PyTorch-compatible version (if torch available)
# ============================================================================

if HAS_TORCH:

    class EternalMemoryTorch(EternalMemory):
        """
        PyTorch-accelerated version of EternalMemory.

        Uses torch tensors for GPU acceleration of similarity search.
        """

        def __init__(
            self,
            memory_size: int = 8192,
            hv_dim: int = 1088,
            recall_threshold: float = 0.65,
            dream_threshold: float = 0.70,
            device: str = "cuda",
            seed: int = 42,
        ):
            super().__init__(
                memory_size=memory_size,
                hv_dim=hv_dim,
                recall_threshold=recall_threshold,
                dream_threshold=dream_threshold,
                seed=seed,
            )

            self.device = torch.device(device if torch.cuda.is_available() else "cpu")

            # Convert to torch tensors
            self.memory_hvs_t = torch.zeros(
                memory_size, hv_dim, device=self.device, dtype=torch.float32
            )
            self.memory_strength_t = torch.zeros(
                memory_size, device=self.device, dtype=torch.float32
            )

            # Time HV
            self.time_hv_t = torch.from_numpy(self.time_hv).to(self.device)

        def store(
            self,
            event_hv,
            emotion_hv,
            strength: float = 1.0,
            emotion_name: str = "",
        ) -> int:
            """Store with torch tensors."""
            # Convert to torch
            if isinstance(event_hv, np.ndarray):
                event_hv = torch.from_numpy(event_hv).to(self.device)
            if isinstance(emotion_hv, np.ndarray):
                emotion_hv = torch.from_numpy(emotion_hv).to(self.device)

            # Handle batch
            if event_hv.dim() == 2:
                event_hv = event_hv.mean(0)
            if emotion_hv.dim() == 2:
                emotion_hv = emotion_hv.mean(0)

            # Bind
            time_phase = self.time_hv_t * torch.cos(
                torch.tensor(self.time_index * 0.01, device=self.device)
            )
            raw = event_hv + emotion_hv + 0.5 * time_phase
            key = torch.sign(raw)
            key = torch.where(key == 0, torch.ones_like(key), key)

            idx = self.write_ptr % self.memory_size
            self.memory_hvs_t[idx] = key
            self.memory_strength_t[idx] = strength

            # Also update numpy arrays for save/load
            self.memory_hvs[idx] = key.cpu().numpy()
            self.memory_strength[idx] = strength
            self.memory_ages[idx] = 0.0
            self.memory_emotions[idx] = emotion_name

            self.write_ptr += 1
            self.total_stored += 1
            self.stores += 1
            self.time_index += 1

            return idx

        def recall(
            self,
            current_hv,
            top_k: int = 3,
            min_sim: Optional[float] = None,
        ) -> RecallResult:
            """Recall with torch acceleration."""
            if self.total_stored == 0:
                return RecallResult(found=False)

            # Convert to torch
            if isinstance(current_hv, np.ndarray):
                current_hv = torch.from_numpy(current_hv).to(self.device)

            if current_hv.dim() == 2:
                current_hv = current_hv.mean(0)

            min_sim = min_sim if min_sim is not None else self.recall_threshold
            count = min(self.total_stored, self.memory_size)
            stored = self.memory_hvs_t[:count]

            # Batch cosine similarity
            sims = torch.nn.functional.cosine_similarity(
                current_hv.unsqueeze(0), stored, dim=1
            )

            top_vals, top_idx = torch.topk(sims, k=min(top_k, count))

            if top_vals[0].item() > min_sim:
                self.hits += 1
                best_idx = top_idx[0].item()

                # Residual
                residual = stored[best_idx] - current_hv

                return RecallResult(
                    found=True,
                    index=int(best_idx),
                    similarity=float(top_vals[0].item()),
                    stored_strength=float(self.memory_strength_t[best_idx].item()),
                    residual=residual.cpu().numpy(),
                )

            return RecallResult(found=False, similarity=float(top_vals[0].item()))


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate eternal memory."""
    print("=" * 60)
    print("Eternal Emotional Memory Demo")
    print("=" * 60)

    memory = EternalMemory(memory_size=1024, hv_dim=256)
    rng = np.random.default_rng(42)

    # Create some "normal" and "anomaly" patterns
    normal_proto = rng.choice([-1, 1], size=256).astype(np.float32)
    anomaly_proto = rng.choice([-1, 1], size=256).astype(np.float32)

    # Emotion HVs
    joy_hv = rng.choice([-1, 1], size=256).astype(np.float32)
    fear_hv = rng.choice([-1, 1], size=256).astype(np.float32)

    print("\n--- Storing memories ---")

    # Store some joyful normal experiences
    for i in range(10):
        event = normal_proto + rng.standard_normal(256) * 0.1
        memory.store(event, joy_hv, strength=0.6 + rng.random() * 0.3, emotion_name="joy")
        print(f"  Stored joyful memory {i}")

    # Store some fearful anomaly experiences
    for i in range(5):
        event = anomaly_proto + rng.standard_normal(256) * 0.1
        memory.store(event, fear_hv, strength=0.7 + rng.random() * 0.2, emotion_name="fear")
        print(f"  Stored fearful memory {i}")

    print(f"\n--- Stats ---")
    stats = memory.get_stats()
    print(f"  Total stored: {stats['total_stored']}")
    print(f"  Avg strength: {stats['avg_strength']:.2f}")

    print("\n--- Recall test ---")

    # Try recalling with a normal-ish pattern
    test_normal = normal_proto + rng.standard_normal(256) * 0.15
    result = memory.recall(test_normal)
    print(f"  Normal pattern recall: found={result.found}, sim={result.similarity:.3f}")
    if result.found:
        print(f"    Index: {result.index}, strength: {result.stored_strength:.2f}")

    # Try recalling with an anomaly-ish pattern
    test_anomaly = anomaly_proto + rng.standard_normal(256) * 0.15
    result = memory.recall(test_anomaly)
    print(f"  Anomaly pattern recall: found={result.found}, sim={result.similarity:.3f}")
    if result.found:
        print(f"    Index: {result.index}, strength: {result.stored_strength:.2f}")

    print("\n--- Dream replay ---")
    reports = memory.dream_replay(steps=32, min_strength=0.5)
    print(f"  Dream hits: {len(reports)}/32")
    for r in reports[:3]:
        print(f"    idx={r.index}, sim={r.similarity:.2f}, strength={r.stored_strength:.2f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
