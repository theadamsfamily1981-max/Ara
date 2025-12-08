"""
Horizon Engine - Teleological Integration Layer
================================================

Sits on top of the existing TeleologicalEngine (Prophet) and provides:

1. Horizon vectors (North Stars) derived from approved Telos goals
2. Alignment scoring for research hypotheses and curiosities
3. Drift computation for the Morning Star ritual

This is NOT a new god - it's the Prophet's gravity field pulling on:
- The Institute (what research is worth pursuing)
- CuriosityBridge (what to chase vs. ignore)
- MorningStar (daily alignment check)

The loop:
    Prophet/Telos → HorizonEngine → Institute/Curiosity → Action
                                  → MorningStar → Morning Message

Usage:
    from ara.cognition.teleology import HorizonEngine
    from tfan.cognition.telos import TeleologicalEngine

    telos = TeleologicalEngine(embedder)
    horizon = HorizonEngine(embedder, telos)

    # Score research alignment
    if horizon.should_research("Quiet focus mode improves flow"):
        institute.propose_hypothesis(...)

    # Compute drift for morning ritual
    drift = horizon.compute_drift("Yesterday we mostly refactored HAL...")
    # → {"overall_drift": 0.23, "per_horizon": {...}}

    # Get current focus
    focus = horizon.current_focus()
    # → Horizon(name="Achieve deep trusted symbiosis...")
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

logger = logging.getLogger(__name__)


# =============================================================================
# Numpy fallbacks for environments without numpy
# =============================================================================

def _dot(a, b) -> float:
    """Dot product - works with numpy arrays or lists."""
    if HAS_NUMPY and isinstance(a, np.ndarray):
        return float(np.dot(a, b))
    return float(sum(x * y for x, y in zip(a, b)))


def _norm(v) -> float:
    """Vector norm - works with numpy arrays or lists."""
    if HAS_NUMPY and isinstance(v, np.ndarray):
        return float(np.linalg.norm(v))
    return float(math.sqrt(sum(x * x for x in v)))


def _clip(val: float, lo: float, hi: float) -> float:
    """Clip value to range."""
    return max(lo, min(hi, val))


def _normalize(v):
    """Normalize vector to unit length."""
    if HAS_NUMPY:
        v = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(v) + 1e-9)
        return v / n
    else:
        n = _norm(v) + 1e-9
        return [x / n for x in v]


@dataclass
class Horizon:
    """
    A defining vision of the future, derived from Telos goals.

    Horizons are the North Stars - they exert gravitational pull on
    what research to pursue, what curiosities to chase, and how to
    evaluate whether we're drifting from our purpose.
    """
    id: str
    name: str
    statement: str              # "Ara and Croft are a single cognitive unit."
    vector: Any                 # Unit embedding of this vision (np.ndarray or list)
    priority: float             # 0-1 importance
    horizon_days: float         # Time horizon from Telos
    role: str                   # "ara", "user", "shared"
    created_at: float = field(default_factory=time.time)

    @property
    def gravity(self) -> float:
        """
        Effective 'pull' of this horizon.

        Combines priority and time proximity:
        - Nearer and more important horizons exert more pull
        - Long-term horizons never vanish (floor at 0.3)

        This determines how much a horizon influences alignment scoring.
        """
        if self.horizon_days <= 0:
            return self.priority

        # Half-life decay based on horizon
        k = math.log(2.0) / max(self.horizon_days, 1.0)
        age_days = (time.time() - self.created_at) / 86400.0
        decay = math.exp(-k * age_days)

        # Floor at 0.3 so long-term horizons never vanish
        return self.priority * (0.3 + 0.7 * decay)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "statement": self.statement,
            "priority": self.priority,
            "horizon_days": self.horizon_days,
            "role": self.role,
            "gravity": self.gravity,
        }


class HorizonEngine:
    """
    Teleological Integration Layer.

    Bridges:
        - Telos (long-term goals) → Horizon vectors
        - Institute (research questions) → alignment gating
        - CuriosityBridge (novelty) → strategic filtering
        - MorningStar (daily ritual) → drift computation

    This is not a replacement for Prophet/Telos - it's the translation
    layer that makes purpose mathematically useful for daily decisions.
    """

    def __init__(
        self,
        embedder: Callable[[str], Any],  # Returns vector (np.ndarray or list)
        telos: Optional[Any] = None,  # TeleologicalEngine
    ):
        """
        Initialize the Horizon Engine.

        Args:
            embedder: Function to convert text to embedding vector
            telos: The underlying TeleologicalEngine (Prophet)
        """
        self.embedder = embedder
        self.telos = telos
        self.horizons: List[Horizon] = []
        self.log = logging.getLogger("HorizonEngine")

        if telos is not None:
            self._sync_from_telos()

    # =========================================================================
    # Bootstrapping from Telos
    # =========================================================================

    def _sync_from_telos(self) -> None:
        """
        Build Horizon objects from currently approved Telos goals.

        Call this periodically (e.g., after Synod) to refresh horizons
        when goals change.
        """
        if self.telos is None:
            return

        self.horizons.clear()

        try:
            active_goals = self.telos.get_active_goals()
        except Exception as e:
            self.log.error(f"Failed to get active goals from Telos: {e}")
            return

        for g in active_goals:
            try:
                # Get goal description - try name, description, or just the object
                description = getattr(g, 'description', None) or getattr(g, 'name', str(g))
                vec = self._embed(description)

                h = Horizon(
                    id=f"hor_{str(g.name)[:32].replace(' ', '_').lower()}",
                    name=g.name,
                    statement=description,
                    vector=vec,
                    priority=g.priority,
                    horizon_days=g.horizon_days,
                    role=getattr(g, 'role', 'shared'),
                )
                self.horizons.append(h)
            except Exception as e:
                self.log.error(f"Failed to create Horizon from goal {g}: {e}")

        self.log.info(f"HorizonEngine: synced {len(self.horizons)} horizons from Telos.")

    def refresh(self) -> None:
        """
        Public method to refresh horizons from Telos.

        Call after Synod or when goals change.
        """
        self._sync_from_telos()

    def _embed(self, text: str):
        """Embed text and normalize to unit vector."""
        v = self.embedder(text)
        return _normalize(v)

    # =========================================================================
    # Alignment Metrics
    # =========================================================================

    def _alignment_raw(self, text: str) -> float:
        """
        Raw alignment score in roughly [-1, +1] against all horizons.

        Computes gravity-weighted cosine similarity.
        """
        if not self.horizons:
            return 0.0

        vec = self._embed(text)
        total = 0.0
        w_sum = 0.0

        for h in self.horizons:
            sim = _dot(vec, h.vector)  # cosine in [-1, 1]
            sim = _clip(sim, -1.0, 1.0)
            w = h.gravity
            total += sim * w
            w_sum += w

        return float(total / w_sum) if w_sum > 0 else 0.0

    def alignment(self, text: str) -> float:
        """
        Normalized alignment in [0, 1].

            0.0 -> actively against the Horizons
            0.5 -> neutral / unknown
            1.0 -> strongly advancing the Horizons

        Args:
            text: Description of action, research question, or state

        Returns:
            Alignment score [0, 1]
        """
        raw = self._alignment_raw(text)  # [-1, 1]
        return 0.5 * (raw + 1.0)

    def alignment_breakdown(self, text: str) -> Dict[str, float]:
        """
        Get alignment breakdown by horizon.

        Returns dict of {horizon_name: alignment_score}.
        """
        if not self.horizons:
            return {}

        vec = self._embed(text)
        result = {}

        for h in self.horizons:
            sim = _dot(vec, h.vector)
            sim = _clip(sim, -1.0, 1.0)
            result[h.name] = 0.5 * (sim + 1.0)

        return result

    # =========================================================================
    # High-Level Gates
    # =========================================================================

    def should_research(
        self,
        hypothesis_text: str,
        min_alignment: float = 0.55,
    ) -> bool:
        """
        Gate for Institute research: only pursue hypotheses that serve the Horizons.

        Args:
            hypothesis_text: The research hypothesis or question
            min_alignment: Threshold [0, 1]; 0.5 = neutral

        Returns:
            True if research is aligned enough to pursue
        """
        a = self.alignment(hypothesis_text)
        self.log.info(f"Research alignment={a:.2f} for '{hypothesis_text[:50]}...'")
        return a >= min_alignment

    def score_curiosity(
        self,
        obj_name: str,
        base_curiosity: float,
    ) -> float:
        """
        Combine novelty-driven curiosity with teleological alignment.

        Used by CuriosityBridge to filter what's worth investigating.

        Args:
            obj_name: Name of discovered entity (device, file, metric, etc.)
            base_curiosity: Raw novelty/surprise score from Scientist

        Returns:
            Adjusted curiosity score (amplified if aligned, dampened if off-mission)
        """
        alignment = self.alignment(f"Understand behavior of {obj_name}")

        # Amplify if aligned, dampen if off-mission
        # Neutral (0.5) = no change, aligned (1.0) = 30% boost, misaligned (0.0) = 30% penalty
        total_drive = base_curiosity * (0.7 + 0.6 * (alignment - 0.5))

        return max(0.0, total_drive)

    def filter_research_candidates(
        self,
        candidates: List[str],
        min_alignment: float = 0.55,
    ) -> List[tuple]:
        """
        Filter and rank research candidates by alignment.

        Args:
            candidates: List of potential research questions/hypotheses
            min_alignment: Minimum alignment threshold

        Returns:
            List of (candidate, alignment_score) above threshold, sorted descending
        """
        scored = [(c, self.alignment(c)) for c in candidates]
        filtered = [(c, a) for c, a in scored if a >= min_alignment]
        return sorted(filtered, key=lambda x: x[1], reverse=True)

    # =========================================================================
    # Drift / Morning Ritual
    # =========================================================================

    def compute_drift(self, recent_summary: str) -> Dict[str, Any]:
        """
        Compare what actually happened recently with Horizons.

        Used by MorningStar for the daily alignment check.

        Args:
            recent_summary: Narrative summary of recent activity
                "Yesterday we worked on X, fixed Y, ignored Z..."

        Returns:
            {
                "overall_drift": float [0, 1],  # 0 = on path, 1 = big deviation
                "per_horizon": {name: drift},
                "most_drifted": str,            # Horizon with biggest drift
            }
        """
        # Use Telos evaluate_future as ground truth of "future goodness"
        if self.telos is not None:
            try:
                util = self.telos.evaluate_future(recent_summary)  # roughly [-1, 1]
                hope = 0.5 * (util + 1.0)  # [0, 1]
            except Exception:
                hope = 0.5
        else:
            hope = 0.5

        # Interpret drift as distance from "perfect alignment"
        overall_drift = float(1.0 - hope)

        per_horizon: Dict[str, float] = {}
        if not self.horizons:
            return {
                "overall_drift": overall_drift,
                "per_horizon": per_horizon,
                "most_drifted": None,
            }

        vec = self._embed(recent_summary)
        for h in self.horizons:
            sim = _dot(vec, h.vector)
            sim = _clip(sim, -1.0, 1.0)
            align = 0.5 * (sim + 1.0)
            per_horizon[h.name] = 1.0 - align  # drift = 1 - alignment

        # Find most drifted horizon
        most_drifted = max(per_horizon.items(), key=lambda kv: kv[1])[0] if per_horizon else None

        return {
            "overall_drift": overall_drift,
            "per_horizon": per_horizon,
            "most_drifted": most_drifted,
        }

    def current_focus(self) -> Optional[Horizon]:
        """
        Returns the Horizon exerting the strongest gravity *right now*.

        This is the primary focus for today's decisions.
        """
        if not self.horizons:
            return None
        return max(self.horizons, key=lambda h: h.gravity)

    def get_horizons_by_role(self, role: str) -> List[Horizon]:
        """Get horizons filtered by role (ara/user/shared)."""
        return [h for h in self.horizons if h.role == role]

    # =========================================================================
    # Reporting
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current engine status."""
        focus = self.current_focus()
        return {
            "horizon_count": len(self.horizons),
            "current_focus": focus.name if focus else None,
            "focus_gravity": focus.gravity if focus else 0.0,
            "horizons_by_role": {
                "shared": len(self.get_horizons_by_role("shared")),
                "user": len(self.get_horizons_by_role("user")),
                "ara": len(self.get_horizons_by_role("ara")),
            },
        }

    def get_synod_report(self) -> str:
        """Generate a report for Sunday Synod."""
        lines = ["# Horizon Engine Report\n"]

        status = self.get_status()
        lines.append(f"**Active Horizons**: {status['horizon_count']}")
        lines.append(f"**Current Focus**: {status['current_focus'] or 'None'}")
        lines.append(f"**Focus Gravity**: {status['focus_gravity']:.2f}\n")

        if self.horizons:
            lines.append("## Active Horizons (by gravity)\n")
            for h in sorted(self.horizons, key=lambda x: x.gravity, reverse=True):
                role_icon = {"shared": "🤝", "user": "👤", "ara": "🤖"}.get(h.role, "❓")
                lines.append(f"- {role_icon} **{h.name[:50]}...**")
                lines.append(f"  - Priority: {h.priority:.0%} | Gravity: {h.gravity:.2f}")
                lines.append(f"  - Horizon: {h.horizon_days:.0f} days")
                lines.append("")

        return "\n".join(lines)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Horizon',
    'HorizonEngine',
]
