"""
Tiny Reflexive Probe - Online Concept Learning
===============================================

A lightweight probe that:
1. Maintains a codebook of learned concept HVs
2. Tags incoming HVs with nearest concepts
3. Generates feedback HVs to inject into next timestep
4. Grows codebook online when novel patterns appear

This is the runtime version of the full HVProbe - optimized for
low-latency online operation rather than visualization.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json


@dataclass
class ConceptMatch:
    """Result of matching an HV to the codebook."""
    tag: str
    similarity: float
    is_novel: bool = False


@dataclass
class ProbeConfig:
    """Configuration for the reflexive probe."""
    hv_dim: int = 1024
    status_dim: int = 64
    max_concepts: int = 64
    novelty_threshold: float = 0.4   # Below this = novel pattern
    bundle_threshold: float = 0.8    # Above this = same concept, bundle
    feedback_strength: float = 0.3   # How much feedback affects next input


class TinyReflexiveProbe:
    """
    Online concept learning with feedback generation.

    The probe:
    1. Receives HV batches from the subcortex
    2. Matches against learned concepts
    3. Grows codebook when novel patterns appear
    4. Generates feedback HVs to influence next timestep

    Usage:
        probe = TinyReflexiveProbe(hv_dim=1024, status_dim=64)
        feedback, matches = probe.process(hv_batch)
        # feedback: [B, hv_dim+status_dim] to inject into next input
        # matches: list of ConceptMatch for each sample
    """

    def __init__(
        self,
        hv_dim: int = 1024,
        status_dim: int = 64,
        max_concepts: int = 64,
        novelty_threshold: float = 0.4,
        bundle_threshold: float = 0.8,
        feedback_strength: float = 0.3,
    ):
        self.hv_dim = hv_dim
        self.status_dim = status_dim
        self.total_dim = hv_dim + status_dim
        self.max_concepts = max_concepts
        self.novelty_threshold = novelty_threshold
        self.bundle_threshold = bundle_threshold
        self.feedback_strength = feedback_strength

        # Codebook: list of (tag, hv, count)
        self.codebook: List[Tuple[str, np.ndarray, int]] = []

        # Persistent feedback (carries across batches)
        self.last_feedback_hv: np.ndarray = np.zeros(self.total_dim, dtype=np.float32)

        # Stats
        self.total_processed = 0
        self.novelty_count = 0

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _find_nearest(self, hv: np.ndarray) -> Tuple[int, float]:
        """Find nearest concept in codebook. Returns (index, similarity)."""
        if len(self.codebook) == 0:
            return -1, 0.0

        best_idx = -1
        best_sim = -1.0

        for idx, (tag, concept_hv, count) in enumerate(self.codebook):
            sim = self._cosine_sim(hv, concept_hv)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        return best_idx, best_sim

    def _add_concept(self, hv: np.ndarray, tag: Optional[str] = None) -> str:
        """Add a new concept to the codebook."""
        if tag is None:
            tag = f"emergent_{len(self.codebook) + 1}"

        # Binarize
        hv_bin = np.sign(hv).astype(np.float32)
        hv_bin[hv_bin == 0] = 1.0

        self.codebook.append((tag, hv_bin.copy(), 1))
        return tag

    def _bundle_into(self, idx: int, hv: np.ndarray) -> None:
        """Bundle new HV into existing concept (online learning)."""
        tag, concept_hv, count = self.codebook[idx]

        # Weighted average (more weight to existing)
        alpha = 1.0 / (count + 1)
        new_hv = (1 - alpha) * concept_hv + alpha * hv

        # Re-binarize
        new_hv = np.sign(new_hv).astype(np.float32)
        new_hv[new_hv == 0] = 1.0

        self.codebook[idx] = (tag, new_hv, count + 1)

    def process_single(self, hv: np.ndarray) -> Tuple[np.ndarray, ConceptMatch]:
        """
        Process a single HV.

        Returns:
            feedback_hv: HV to inject into next timestep
            match: ConceptMatch with tag and similarity
        """
        self.total_processed += 1

        # Initialize codebook with first sample
        if len(self.codebook) == 0:
            tag = self._add_concept(hv, "baseline")
            return hv.copy(), ConceptMatch(tag=tag, similarity=1.0, is_novel=True)

        # Find nearest concept
        idx, sim = self._find_nearest(hv)

        if sim < self.novelty_threshold:
            # Novel pattern - add to codebook if space
            self.novelty_count += 1
            if len(self.codebook) < self.max_concepts:
                tag = self._add_concept(hv)
                match = ConceptMatch(tag=tag, similarity=sim, is_novel=True)
            else:
                # Codebook full - just tag with nearest
                tag = self.codebook[idx][0]
                match = ConceptMatch(tag=f"near_{tag}", similarity=sim, is_novel=True)

            # Feedback: use novel HV itself
            feedback = hv.copy()

        elif sim > self.bundle_threshold:
            # Strong match - bundle into existing concept
            self._bundle_into(idx, hv)
            tag = self.codebook[idx][0]
            match = ConceptMatch(tag=tag, similarity=sim, is_novel=False)

            # Feedback: use concept HV (reinforcement)
            feedback = self.codebook[idx][1].copy()

        else:
            # Moderate match - tag but don't modify
            tag = self.codebook[idx][0]
            match = ConceptMatch(tag=tag, similarity=sim, is_novel=False)

            # Feedback: blend of input and concept
            feedback = 0.5 * hv + 0.5 * self.codebook[idx][1]
            feedback = np.sign(feedback).astype(np.float32)
            feedback[feedback == 0] = 1.0

        self.last_feedback_hv = feedback
        return feedback, match

    def process_batch(
        self,
        hv_batch: np.ndarray,
    ) -> Tuple[np.ndarray, List[ConceptMatch]]:
        """
        Process a batch of HVs.

        Args:
            hv_batch: [B, hv_dim+status_dim] input HVs

        Returns:
            feedback_batch: [B, hv_dim+status_dim] feedback HVs
            matches: List of ConceptMatch for each sample
        """
        B = hv_batch.shape[0]
        feedback_batch = np.zeros_like(hv_batch)
        matches = []

        for i in range(B):
            fb, match = self.process_single(hv_batch[i])
            feedback_batch[i] = fb
            matches.append(match)

        return feedback_batch, matches

    def get_codebook_summary(self) -> Dict:
        """Get summary of current codebook state."""
        return {
            "num_concepts": len(self.codebook),
            "max_concepts": self.max_concepts,
            "total_processed": self.total_processed,
            "novelty_count": self.novelty_count,
            "novelty_rate": self.novelty_count / max(1, self.total_processed),
            "concepts": [
                {"tag": tag, "count": count}
                for tag, _, count in self.codebook
            ],
        }

    def get_concept_tags(self) -> List[str]:
        """Get list of all concept tags."""
        return [tag for tag, _, _ in self.codebook]

    def save(self, path: str) -> None:
        """Save codebook to file."""
        data = {
            "config": {
                "hv_dim": self.hv_dim,
                "status_dim": self.status_dim,
                "max_concepts": self.max_concepts,
                "novelty_threshold": self.novelty_threshold,
                "bundle_threshold": self.bundle_threshold,
            },
            "codebook": [
                {"tag": tag, "hv": hv.tolist(), "count": count}
                for tag, hv, count in self.codebook
            ],
            "stats": {
                "total_processed": self.total_processed,
                "novelty_count": self.novelty_count,
            },
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load codebook from file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.codebook = [
            (item["tag"], np.array(item["hv"], dtype=np.float32), item["count"])
            for item in data["codebook"]
        ]
        self.total_processed = data["stats"]["total_processed"]
        self.novelty_count = data["stats"]["novelty_count"]


def demo():
    """Demonstrate the reflexive probe."""
    print("=" * 60)
    print("Tiny Reflexive Probe Demo")
    print("=" * 60)

    probe = TinyReflexiveProbe(hv_dim=256, status_dim=32)

    # Generate synthetic HV stream
    rng = np.random.default_rng(42)

    # Create a "normal" prototype
    normal_proto = rng.choice([-1, 1], size=288).astype(np.float32)

    print("\n--- Processing HV stream ---")

    for step in range(50):
        # Generate HV: mostly normal with occasional anomalies
        if step % 10 == 7:
            # Anomaly: random HV
            hv = rng.choice([-1, 1], size=288).astype(np.float32)
        else:
            # Normal: perturbed prototype
            hv = normal_proto.copy()
            flip_mask = rng.random(288) < 0.1
            hv[flip_mask] *= -1

        feedback, match = probe.process_single(hv)

        if match.is_novel or step % 10 == 0:
            print(f"Step {step:3d}: {match.tag:20s} sim={match.similarity:.3f}"
                  f" {'[NOVEL]' if match.is_novel else ''}")

    print("\n--- Codebook Summary ---")
    summary = probe.get_codebook_summary()
    print(f"Concepts: {summary['num_concepts']}/{summary['max_concepts']}")
    print(f"Novelty rate: {summary['novelty_rate']:.2%}")
    print("\nConcepts:")
    for c in summary["concepts"]:
        print(f"  {c['tag']}: seen {c['count']} times")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
