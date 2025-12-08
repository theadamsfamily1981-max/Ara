"""
Hyperdimensional Probe (HDP)
============================

Query state hypervectors against a codebook of known concepts.

The probe is the "pattern recognition" layer of the correlation engine:
- Maintains a codebook Φ of concept HPVs
- Computes similarity of state to all concepts
- Detects anomalies (low similarity to all known concepts)
- Provides ranked concept activations for decision-making

This is what decides whether to handle locally or escalate to LLM.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class HDProbeConfig:
    """Configuration for the HD probe."""
    dim: int = 1024                      # Hypervector dimension
    anomaly_threshold: float = 0.3       # Below this = anomaly
    novelty_threshold: float = 0.5       # Below this = novel (needs learning)
    top_k: int = 5                       # Number of top concepts to return
    min_similarity: float = 0.1          # Ignore concepts below this


@dataclass
class ProbeResult:
    """Result of probing state against codebook."""
    top_concepts: List[Tuple[str, float]]  # [(concept_name, similarity), ...]
    max_similarity: float                   # Highest similarity found
    mean_similarity: float                  # Mean similarity to all concepts
    is_anomaly: bool                        # Below anomaly threshold?
    is_novel: bool                          # Below novelty threshold?
    anomaly_score: float                    # 1 - max_similarity


class HDProbe:
    """
    Probe state against a codebook of known concepts.

    The codebook can be:
    - Pre-defined (known patterns, policies)
    - Learned (from LLM feedback via NEW_POLICY_HDC)
    - Hybrid (both)
    """

    def __init__(self, config: Optional[HDProbeConfig] = None):
        self.cfg = config or HDProbeConfig()

        # Codebook: concept_name → HPV
        self._codebook: Dict[str, np.ndarray] = {}

        # Metadata for each concept
        self._concept_meta: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self._probe_count = 0
        self._anomaly_count = 0

    @property
    def num_concepts(self) -> int:
        """Number of concepts in codebook."""
        return len(self._codebook)

    def add_concept(self, name: str, hv: np.ndarray,
                    meta: Optional[Dict[str, Any]] = None):
        """
        Add a concept to the codebook.

        Args:
            name: Concept identifier
            hv: Hypervector for this concept
            meta: Optional metadata (source, timestamp, etc.)
        """
        self._codebook[name] = hv.astype(np.int8)
        self._concept_meta[name] = meta or {}

    def remove_concept(self, name: str) -> bool:
        """Remove a concept from the codebook."""
        if name in self._codebook:
            del self._codebook[name]
            del self._concept_meta[name]
            return True
        return False

    def get_concept(self, name: str) -> Optional[np.ndarray]:
        """Get a concept's hypervector."""
        return self._codebook.get(name)

    def list_concepts(self) -> List[str]:
        """List all concept names."""
        return list(self._codebook.keys())

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity (for bipolar vectors)."""
        return float(np.dot(a.astype(np.float32), b.astype(np.float32))) / len(a)

    def probe(self, state_hv: np.ndarray) -> ProbeResult:
        """
        Probe state against all concepts in codebook.

        Returns ranked similarities and anomaly/novelty flags.
        """
        self._probe_count += 1

        if not self._codebook:
            # No concepts yet = everything is novel
            return ProbeResult(
                top_concepts=[],
                max_similarity=0.0,
                mean_similarity=0.0,
                is_anomaly=True,
                is_novel=True,
                anomaly_score=1.0,
            )

        # Compute similarities to all concepts
        similarities = []
        for name, hv in self._codebook.items():
            sim = self._similarity(state_hv, hv)
            if sim >= self.cfg.min_similarity:
                similarities.append((name, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Extract metrics
        if similarities:
            max_sim = similarities[0][1]
            mean_sim = np.mean([s for _, s in similarities])
        else:
            max_sim = 0.0
            mean_sim = 0.0

        # Determine flags
        is_anomaly = max_sim < self.cfg.anomaly_threshold
        is_novel = max_sim < self.cfg.novelty_threshold

        if is_anomaly:
            self._anomaly_count += 1

        return ProbeResult(
            top_concepts=similarities[:self.cfg.top_k],
            max_similarity=max_sim,
            mean_similarity=float(mean_sim),
            is_anomaly=is_anomaly,
            is_novel=is_novel,
            anomaly_score=1.0 - max_sim,
        )

    def find_similar(self, hv: np.ndarray, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """Find all concepts above similarity threshold."""
        results = []
        for name, concept_hv in self._codebook.items():
            sim = self._similarity(hv, concept_hv)
            if sim >= threshold:
                results.append((name, sim))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get probe statistics."""
        return {
            "num_concepts": self.num_concepts,
            "probe_count": self._probe_count,
            "anomaly_count": self._anomaly_count,
            "anomaly_rate": self._anomaly_count / max(1, self._probe_count),
        }

    def save_codebook(self) -> Dict[str, Any]:
        """Export codebook for persistence."""
        return {
            name: {
                "hv": hv.tolist(),
                "meta": self._concept_meta.get(name, {}),
            }
            for name, hv in self._codebook.items()
        }

    def load_codebook(self, data: Dict[str, Any]):
        """Import codebook from saved data."""
        for name, entry in data.items():
            hv = np.array(entry["hv"], dtype=np.int8)
            meta = entry.get("meta", {})
            self.add_concept(name, hv, meta)


# ============================================================================
# Pre-defined Concept Categories
# ============================================================================

def create_system_concepts(encoder) -> Dict[str, np.ndarray]:
    """
    Create pre-defined concepts for system monitoring.

    These are "known patterns" that the probe can recognize.
    """
    concepts = {}

    # Normal operation patterns
    concepts["normal_idle"] = encoder.encode_metrics({
        "cpu": 0.1, "memory": 0.3, "disk_io": 0.05, "network": 0.1
    })
    concepts["normal_load"] = encoder.encode_metrics({
        "cpu": 0.5, "memory": 0.5, "disk_io": 0.3, "network": 0.3
    })
    concepts["normal_peak"] = encoder.encode_metrics({
        "cpu": 0.8, "memory": 0.7, "disk_io": 0.5, "network": 0.5
    })

    # Anomaly patterns
    concepts["cpu_spike"] = encoder.encode_metrics({
        "cpu": 0.95, "memory": 0.3, "disk_io": 0.1, "network": 0.1
    })
    concepts["memory_pressure"] = encoder.encode_metrics({
        "cpu": 0.3, "memory": 0.95, "disk_io": 0.2, "network": 0.1
    })
    concepts["disk_thrash"] = encoder.encode_metrics({
        "cpu": 0.4, "memory": 0.6, "disk_io": 0.95, "network": 0.1
    })
    concepts["network_storm"] = encoder.encode_metrics({
        "cpu": 0.3, "memory": 0.4, "disk_io": 0.1, "network": 0.95
    })

    # User behavior patterns
    concepts["user_active"] = encoder.encode_text("keyboard mouse activity user input")
    concepts["user_idle"] = encoder.encode_text("idle inactive no input screensaver")
    concepts["user_frustration"] = encoder.encode_text("rapid clicking backspace error retry")

    return concepts


def create_probe_with_system_concepts(encoder, config: Optional[HDProbeConfig] = None) -> HDProbe:
    """Create a probe pre-loaded with system monitoring concepts."""
    probe = HDProbe(config)
    concepts = create_system_concepts(encoder)
    for name, hv in concepts.items():
        probe.add_concept(name, hv, {"source": "predefined", "category": "system"})
    return probe
