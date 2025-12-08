"""
Resonant Guidance - HDC-Based Thought Navigation
==================================================

Uses the HDC/SNN infrastructure to:
1. Encode thinking sessions as hypervectors (fingerprints)
2. Cluster sessions to discover thinking modes
3. Compare live sessions to past patterns
4. Navigate the manifold of intellectual states

When you start a session, Ara compares the live HDC stream to past
hypervectors. She can implicitly steer:
- "You're drifting into the same basin as that stale refactor..."
- "You're in the basin of that legendary breakthrough night..."

This is thought navigation: steering through a learned manifold.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import hashlib

from .trace import (
    ThoughtTrace, ThoughtMove, MoveType,
    SessionContext, SessionOutcome
)
from .graph import StrategyType, StrategyNode


@dataclass
class ThoughtHV:
    """
    Hypervector representation of a thinking concept or pattern.

    Uses binary hypervectors for HDC operations.
    """
    dim: int = 8192
    vector: np.ndarray = field(default_factory=lambda: np.zeros(8192, dtype=np.uint8))
    label: str = ""

    def __post_init__(self):
        if len(self.vector) != self.dim:
            self.vector = np.zeros(self.dim, dtype=np.uint8)

    @classmethod
    def random(cls, dim: int = 8192, label: str = "", seed: Optional[int] = None) -> "ThoughtHV":
        """Create a random hypervector."""
        rng = np.random.default_rng(seed)
        return cls(dim=dim, vector=rng.integers(0, 2, size=dim, dtype=np.uint8), label=label)

    @classmethod
    def from_name(cls, name: str, dim: int = 8192) -> "ThoughtHV":
        """Create a consistent HV from a name (seeded by hash)."""
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        return cls.random(dim=dim, label=name, seed=seed)

    def bind(self, other: "ThoughtHV") -> "ThoughtHV":
        """XOR binding: A ⊗ B"""
        result = np.bitwise_xor(self.vector, other.vector)
        return ThoughtHV(dim=self.dim, vector=result, label=f"({self.label}⊗{other.label})")

    def permute(self, shift: int) -> "ThoughtHV":
        """Cyclic permutation: ρ^k(A)"""
        result = np.roll(self.vector, shift)
        return ThoughtHV(dim=self.dim, vector=result, label=f"ρ{shift}({self.label})")

    @staticmethod
    def bundle(hvs: List["ThoughtHV"]) -> "ThoughtHV":
        """Majority vote bundling."""
        if not hvs:
            return ThoughtHV()
        if len(hvs) == 1:
            return ThoughtHV(dim=hvs[0].dim, vector=hvs[0].vector.copy())

        dim = hvs[0].dim
        total = np.sum([hv.vector for hv in hvs], axis=0)
        threshold = len(hvs) / 2
        result = (total > threshold).astype(np.uint8)

        # Tie-break randomly
        ties = (total == threshold)
        if np.any(ties):
            rng = np.random.default_rng()
            result[ties] = rng.integers(0, 2, size=np.sum(ties), dtype=np.uint8)

        return ThoughtHV(dim=dim, vector=result, label=f"bundle({len(hvs)})")

    def similarity(self, other: "ThoughtHV") -> float:
        """Cosine-ish similarity for binary vectors."""
        matches = np.sum(self.vector == other.vector)
        return (2 * matches - self.dim) / self.dim


@dataclass
class SessionFingerprint:
    """
    The HDC fingerprint of a complete thinking session.

    Encodes:
    - Goal encoding
    - Move sequence encoding
    - Context encoding
    - Outcome encoding
    """
    trace_id: str
    fingerprint: ThoughtHV
    outcome: SessionOutcome
    goal: str
    domain: str

    # Metadata for retrieval
    quality_score: float = 0.0
    duration_minutes: float = 0.0


@dataclass
class SessionEncoder:
    """
    Encodes ThoughtTraces as hypervectors.

    Uses HDC algebra to compose:
    - Move type HVs
    - Sequence encoding via permutation
    - Context binding
    """
    dim: int = 8192

    # Base HVs for move types
    _move_hvs: Dict[MoveType, ThoughtHV] = field(default_factory=dict)
    _outcome_hvs: Dict[SessionOutcome, ThoughtHV] = field(default_factory=dict)
    _context_hvs: Dict[str, ThoughtHV] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize base HVs for each move type
        for move_type in MoveType:
            self._move_hvs[move_type] = ThoughtHV.from_name(f"move:{move_type.name}", self.dim)

        # Initialize outcome HVs
        for outcome in SessionOutcome:
            self._outcome_hvs[outcome] = ThoughtHV.from_name(f"outcome:{outcome.name}", self.dim)

    def encode_trace(self, trace: ThoughtTrace) -> SessionFingerprint:
        """Encode a complete trace as a fingerprint."""
        components = []

        # 1. Encode goal (simple hash-based HV)
        goal_hv = ThoughtHV.from_name(f"goal:{trace.goal[:50]}", self.dim)
        components.append(goal_hv)

        # 2. Encode move sequence
        if trace.moves:
            move_hvs = []
            for i, move in enumerate(trace.moves):
                move_hv = self._move_hvs.get(move.move_type)
                if move_hv:
                    # Permute by position to encode sequence
                    positioned = move_hv.permute(i * 7)  # Prime shift for distribution
                    move_hvs.append(positioned)

            if move_hvs:
                sequence_hv = ThoughtHV.bundle(move_hvs)
                components.append(sequence_hv)

        # 3. Encode context features
        context_hvs = []
        ctx = trace.context

        if ctx.domain:
            domain_hv = ThoughtHV.from_name(f"domain:{ctx.domain}", self.dim)
            context_hvs.append(domain_hv)

        # Fatigue level
        fatigue_level = int(ctx.fatigue * 10)
        fatigue_hv = ThoughtHV.from_name(f"fatigue:{fatigue_level}", self.dim)
        context_hvs.append(fatigue_hv)

        # Difficulty level
        diff_level = int(ctx.difficulty_estimate * 10)
        diff_hv = ThoughtHV.from_name(f"difficulty:{diff_level}", self.dim)
        context_hvs.append(diff_hv)

        if context_hvs:
            context_bundle = ThoughtHV.bundle(context_hvs)
            components.append(context_bundle)

        # 4. Encode outcome
        outcome_hv = self._outcome_hvs.get(trace.outcome)
        if outcome_hv:
            components.append(outcome_hv)

        # Bundle all components
        fingerprint = ThoughtHV.bundle(components)

        return SessionFingerprint(
            trace_id=trace.trace_id,
            fingerprint=fingerprint,
            outcome=trace.outcome,
            goal=trace.goal,
            domain=trace.context.domain,
            quality_score=trace.quality_score,
            duration_minutes=trace.duration_minutes,
        )


@dataclass
class ResonantGuide:
    """
    Uses HDC resonance for thought navigation.

    Compares live sessions to past patterns and provides guidance.
    """
    encoder: SessionEncoder = field(default_factory=SessionEncoder)
    _fingerprints: List[SessionFingerprint] = field(default_factory=list)

    # Cluster centers (discovered thinking modes)
    _mode_centers: Dict[str, ThoughtHV] = field(default_factory=dict)

    def add_fingerprint(self, fp: SessionFingerprint):
        """Add a session fingerprint to the library."""
        self._fingerprints.append(fp)

    def ingest_traces(self, traces: List[ThoughtTrace]):
        """Encode and store multiple traces."""
        for trace in traces:
            fp = self.encoder.encode_trace(trace)
            self.add_fingerprint(fp)

    def find_similar_sessions(self, trace: ThoughtTrace,
                              n: int = 5) -> List[Tuple[SessionFingerprint, float]]:
        """
        Find past sessions similar to a given trace.

        Returns list of (fingerprint, similarity) sorted by similarity.
        """
        query_fp = self.encoder.encode_trace(trace)
        return self.find_similar_to_fingerprint(query_fp.fingerprint, n)

    def find_similar_to_fingerprint(self, query_hv: ThoughtHV,
                                    n: int = 5) -> List[Tuple[SessionFingerprint, float]]:
        """Find sessions similar to a fingerprint."""
        similarities = []
        for fp in self._fingerprints:
            sim = query_hv.similarity(fp.fingerprint)
            similarities.append((fp, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

    def get_basin_assessment(self, trace: ThoughtTrace) -> Dict[str, Any]:
        """
        Assess which "basin" the current session is in.

        Returns assessment with:
        - Similar successful sessions
        - Similar failed sessions
        - Risk factors
        - Opportunities
        """
        similar = self.find_similar_sessions(trace, n=10)

        successful = [(fp, sim) for fp, sim in similar
                      if fp.outcome in [SessionOutcome.BREAKTHROUGH, SessionOutcome.PROGRESS]]
        failed = [(fp, sim) for fp, sim in similar
                  if fp.outcome in [SessionOutcome.STALLED, SessionOutcome.ABANDONED]]

        # Compute basin tendency
        if successful and failed:
            success_sim = np.mean([s for _, s in successful])
            fail_sim = np.mean([s for _, s in failed])
            tendency = success_sim - fail_sim  # Positive = leaning toward success
        elif successful:
            tendency = 0.5
        elif failed:
            tendency = -0.5
        else:
            tendency = 0.0

        assessment = {
            "tendency": tendency,
            "similar_successes": len(successful),
            "similar_failures": len(failed),
            "closest_success": successful[0] if successful else None,
            "closest_failure": failed[0] if failed else None,
            "risk_factors": [],
            "opportunities": [],
        }

        # Identify risk factors
        if failed:
            closest_fail, fail_sim = failed[0]
            if fail_sim > 0.6:
                assessment["risk_factors"].append(
                    f"Similar to failed session '{closest_fail.goal[:30]}...' ({fail_sim:.0%} match)"
                )

        # Identify opportunities
        if successful:
            closest_success, success_sim = successful[0]
            if success_sim > 0.6:
                assessment["opportunities"].append(
                    f"Similar to successful session '{closest_success.goal[:30]}...' ({success_sim:.0%} match)"
                )

        return assessment

    def get_guidance(self, trace: ThoughtTrace,
                     context: SessionContext) -> List[str]:
        """
        Get guidance based on resonance with past sessions.

        Returns list of guidance messages.
        """
        guidance = []
        assessment = self.get_basin_assessment(trace)

        tendency = assessment["tendency"]

        if tendency < -0.3:
            guidance.append(
                "⚠️ This session pattern resembles past stalls. "
                "Consider: switch modality, take a break, or reframe."
            )
        elif tendency > 0.3:
            guidance.append(
                "✨ This session pattern resembles past breakthroughs. "
                "Stay in the zone, minimize interruptions."
            )

        for risk in assessment["risk_factors"]:
            guidance.append(f"Risk: {risk}")

        for opp in assessment["opportunities"]:
            guidance.append(f"Opportunity: {opp}")

        return guidance

    def discover_modes(self, n_modes: int = 5) -> Dict[str, ThoughtHV]:
        """
        Discover distinct thinking modes via clustering.

        Uses simple k-means-like clustering in HV space.
        """
        if len(self._fingerprints) < n_modes:
            return {}

        # Initialize centers randomly from existing fingerprints
        rng = np.random.default_rng()
        indices = rng.choice(len(self._fingerprints), size=n_modes, replace=False)
        centers = {
            f"mode_{i}": self._fingerprints[idx].fingerprint
            for i, idx in enumerate(indices)
        }

        # Simple iterative refinement
        for _ in range(10):
            # Assign fingerprints to nearest center
            clusters = defaultdict(list)
            for fp in self._fingerprints:
                best_mode = None
                best_sim = -1
                for mode, center in centers.items():
                    sim = fp.fingerprint.similarity(center)
                    if sim > best_sim:
                        best_sim = sim
                        best_mode = mode
                if best_mode:
                    clusters[best_mode].append(fp)

            # Update centers
            for mode, fps in clusters.items():
                if fps:
                    hvs = [fp.fingerprint for fp in fps]
                    centers[mode] = ThoughtHV.bundle(hvs)

        self._mode_centers = centers
        return centers

    def identify_mode(self, trace: ThoughtTrace) -> Optional[str]:
        """Identify which thinking mode a session is in."""
        if not self._mode_centers:
            return None

        fp = self.encoder.encode_trace(trace)
        best_mode = None
        best_sim = -1

        for mode, center in self._mode_centers.items():
            sim = fp.fingerprint.similarity(center)
            if sim > best_sim:
                best_sim = sim
                best_mode = mode

        return best_mode if best_sim > 0.3 else None


# Global resonant guide
_global_guide: Optional[ResonantGuide] = None


def get_resonant_guide() -> ResonantGuide:
    """Get the global ResonantGuide instance."""
    global _global_guide
    if _global_guide is None:
        _global_guide = ResonantGuide()
    return _global_guide
