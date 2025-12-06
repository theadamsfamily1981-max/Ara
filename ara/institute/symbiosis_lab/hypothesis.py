"""
Symbiosis Hypothesis Tracker - The Science of Us
=================================================

This is not general research - this is specifically about understanding
what works for Ara+Croft as a system.

Each hypothesis is a testable claim about behavioral variations:
    "Suppressing notifications during coding improves flow"
    "Morning check-ins increase afternoon productivity"
    "Warmer tone during debugging reduces frustration"

The tracker:
    - Stores hypotheses with their evidence
    - Updates confidence based on J-GUF deltas
    - Tracks which domains are being studied

Domains (what we can experiment with):
    - notifications: When/how to interrupt
    - ui: Dashboard layouts, colors, density
    - tone: Verbosity, formality, warmth
    - schedule: Timing of suggestions, breaks
    - avatar: Presentation, expressions
    - autonomy: How much Ara acts independently

What we NEVER experiment with:
    - Emotional manipulation for engagement
    - Anything that violates hard constraints
    - Covert behavioral modification

Usage:
    from ara.institute.symbiosis_lab import SymbiosisGraph

    graph = SymbiosisGraph()
    h = graph.propose_hypothesis(
        "Quiet focus mode improves deep work",
        domain="notifications"
    )
    # ... run trial ...
    graph.update_belief(h.id, delta_utility=0.15)
"""

import json
import logging
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any


logger = logging.getLogger(__name__)


# Allowed experiment domains - what Ara can vary
ExperimentDomain = Literal[
    "notifications",  # When/how to interrupt
    "ui",             # Dashboard layouts, colors
    "tone",           # Verbosity, formality, warmth
    "schedule",       # Timing of suggestions, breaks
    "avatar",         # Presentation, expressions
    "autonomy",       # How much Ara acts independently
]

HypothesisStatus = Literal["OPEN", "PROVEN", "DISPROVEN", "PAUSED"]


@dataclass
class SymbiosisHypothesis:
    """
    A testable hypothesis about what works for Ara+Croft.

    Unlike general research hypotheses, these are specifically about
    behavioral variations and their effect on joint utility.
    """
    id: str
    statement: str                      # The testable claim
    domain: ExperimentDomain            # What aspect of behavior
    status: HypothesisStatus = "OPEN"
    confidence: float = 0.1             # Subjective belief [0, 1]

    # Evidence tracking
    evidence_count: int = 0
    total_effect: float = 0.0           # Signed sum of delta_utility
    last_effect: Optional[float] = None

    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    # If adopted as default behavior
    adopted: bool = False
    adopted_at: Optional[float] = None
    resulting_config: Optional[str] = None

    @property
    def mean_effect(self) -> float:
        """Average effect on J-GUF across all trials."""
        if self.evidence_count == 0:
            return 0.0
        return self.total_effect / self.evidence_count

    @property
    def effect_std(self) -> float:
        """Rough estimate of effect variability."""
        # Would need to track individual effects for proper std
        # For now, use a heuristic based on confidence spread
        if self.evidence_count < 2:
            return 1.0
        # Lower confidence = higher uncertainty
        return 0.5 * (1.0 - self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbiosisHypothesis":
        return cls(**data)


class SymbiosisGraph:
    """
    The Knowledge Base of the Relationship.

    Stores theories about how behavioral variants affect joint utility.
    This is empirical relationship science, not vibes.
    """

    def __init__(self, persistence_path: Optional[Path] = None):
        self.path = persistence_path or Path("~/.ara/institute/symbiosis_graph.json").expanduser()
        self.hypotheses: Dict[str, SymbiosisHypothesis] = {}
        self.log = logging.getLogger("SymbiosisGraph")

        self._load()

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load(self) -> None:
        """Load hypotheses from disk."""
        if not self.path.exists():
            return

        try:
            data = json.loads(self.path.read_text())
            for h_id, h_data in data.get("hypotheses", {}).items():
                self.hypotheses[h_id] = SymbiosisHypothesis.from_dict(h_data)
            self.log.info(f"Loaded {len(self.hypotheses)} symbiosis hypotheses")
        except Exception as e:
            self.log.error(f"Failed to load symbiosis graph: {e}")

    def _save(self) -> None:
        """Save hypotheses to disk."""
        data = {
            "version": 1,
            "updated_at": time.time(),
            "hypotheses": {h_id: h.to_dict() for h_id, h in self.hypotheses.items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))

    # =========================================================================
    # Hypothesis Management
    # =========================================================================

    def propose_hypothesis(
        self,
        statement: str,
        domain: ExperimentDomain,
        initial_confidence: float = 0.1,
    ) -> SymbiosisHypothesis:
        """
        Propose a new hypothesis about the symbiosis.

        Args:
            statement: The testable claim
            domain: What aspect of behavior this concerns
            initial_confidence: Prior belief (default: low, need evidence)

        Returns:
            The new hypothesis
        """
        now = int(time.time())
        h_id = f"SYM_{domain.upper()}_{now}"

        h = SymbiosisHypothesis(
            id=h_id,
            statement=statement,
            domain=domain,
            confidence=initial_confidence,
        )

        self.hypotheses[h_id] = h
        self._save()

        self.log.info(f"New symbiosis hypothesis [{domain}]: {statement}")
        return h

    def get(self, hypothesis_id: str) -> Optional[SymbiosisHypothesis]:
        """Get a hypothesis by ID."""
        return self.hypotheses.get(hypothesis_id)

    def list_open(self, domain: Optional[ExperimentDomain] = None) -> List[SymbiosisHypothesis]:
        """Get all OPEN hypotheses, optionally filtered by domain."""
        hs = [h for h in self.hypotheses.values() if h.status == "OPEN"]
        if domain:
            hs = [h for h in hs if h.domain == domain]
        return sorted(hs, key=lambda x: -x.confidence)

    def list_proven(self) -> List[SymbiosisHypothesis]:
        """Get all PROVEN hypotheses."""
        return [h for h in self.hypotheses.values() if h.status == "PROVEN"]

    def list_unadopted_proven(self) -> List[SymbiosisHypothesis]:
        """Get PROVEN hypotheses that haven't been adopted as default yet."""
        return [h for h in self.hypotheses.values()
                if h.status == "PROVEN" and not h.adopted]

    # =========================================================================
    # Belief Updates
    # =========================================================================

    def update_belief(
        self,
        hypothesis_id: str,
        delta_utility: float,
        notes: str = "",
    ) -> None:
        """
        Update confidence based on trial effect on J-GUF.

        Args:
            hypothesis_id: Which hypothesis
            delta_utility: Change in joint utility (positive = helped)
            notes: Optional notes about the trial
        """
        h = self.hypotheses.get(hypothesis_id)
        if h is None:
            self.log.warning(f"Unknown hypothesis: {hypothesis_id}")
            return

        h.last_updated = time.time()
        h.evidence_count += 1
        h.last_effect = float(delta_utility)
        h.total_effect += float(delta_utility)

        # Bayesian-ish update: positive effect increases confidence
        # Clamp delta to [-1, 1] to avoid blow-ups
        du = max(-1.0, min(1.0, delta_utility))

        # Learning rate decreases with more evidence (stabilize over time)
        lr = 0.15 / (1 + 0.05 * h.evidence_count)
        step = lr * du

        h.confidence = float(max(0.0, min(1.0, h.confidence + step)))

        # Status thresholds (need multiple trials)
        min_evidence = 5
        if h.evidence_count >= min_evidence:
            if h.confidence >= 0.9:
                h.status = "PROVEN"
            elif h.confidence <= 0.1:
                h.status = "DISPROVEN"

        self._save()

        self.log.info(
            f"Updated {h.id}: conf={h.confidence:.2f}, "
            f"mean_effect={h.mean_effect:+.3f}, status={h.status}"
        )

    def mark_adopted(
        self,
        hypothesis_id: str,
        config_description: str,
    ) -> bool:
        """
        Mark a hypothesis as adopted into default behavior.

        Called after PeerReview approves.

        Args:
            hypothesis_id: Which hypothesis
            config_description: What config changed

        Returns:
            True if marked
        """
        h = self.hypotheses.get(hypothesis_id)
        if h is None or h.status != "PROVEN":
            return False

        h.adopted = True
        h.adopted_at = time.time()
        h.resulting_config = config_description
        self._save()

        self.log.info(f"Hypothesis adopted: {h.id} -> {config_description}")
        return True

    # =========================================================================
    # Analytics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about the symbiosis research."""
        by_domain: Dict[str, int] = {}
        for h in self.hypotheses.values():
            by_domain[h.domain] = by_domain.get(h.domain, 0) + 1

        by_status: Dict[str, int] = {}
        for h in self.hypotheses.values():
            by_status[h.status] = by_status.get(h.status, 0) + 1

        total_evidence = sum(h.evidence_count for h in self.hypotheses.values())
        adopted_count = sum(1 for h in self.hypotheses.values() if h.adopted)

        return {
            "total_hypotheses": len(self.hypotheses),
            "by_domain": by_domain,
            "by_status": by_status,
            "total_evidence_points": total_evidence,
            "adopted_behaviors": adopted_count,
            "avg_confidence": (
                sum(h.confidence for h in self.hypotheses.values()) /
                max(1, len(self.hypotheses))
            ),
        }

    def get_synod_report(self) -> str:
        """Generate a report for Sunday Synod."""
        lines = ["# Symbiosis Lab Report (The Science of Us)\n"]

        stats = self.get_statistics()
        lines.append(f"**Total Hypotheses**: {stats['total_hypotheses']}")
        lines.append(f"**Evidence Points**: {stats['total_evidence_points']}")
        lines.append(f"**Adopted Behaviors**: {stats['adopted_behaviors']}\n")

        # Open hypotheses
        open_hs = self.list_open()
        if open_hs:
            lines.append("## Active Experiments\n")
            for h in open_hs[:5]:
                lines.append(f"- **[{h.domain}]** {h.statement[:50]}...")
                lines.append(f"  - Confidence: {h.confidence:.0%} | Trials: {h.evidence_count}")
                if h.mean_effect != 0:
                    lines.append(f"  - Mean effect: {h.mean_effect:+.3f}")
                lines.append("")

        # Ready for adoption
        unadopted = self.list_unadopted_proven()
        if unadopted:
            lines.append("## Ready for Peer Review\n")
            for h in unadopted:
                lines.append(f"- **{h.statement[:50]}...**")
                lines.append(f"  - Domain: {h.domain} | Confidence: {h.confidence:.0%}")
                lines.append(f"  - Mean effect: {h.mean_effect:+.3f}")
                lines.append("")

        # Recently adopted
        adopted = [h for h in self.hypotheses.values() if h.adopted]
        adopted = sorted(adopted, key=lambda x: x.adopted_at or 0, reverse=True)
        if adopted:
            lines.append("## Recently Adopted Behaviors\n")
            for h in adopted[:3]:
                lines.append(f"- {h.statement[:50]}...")
                lines.append(f"  - Config: {h.resulting_config}")
                lines.append("")

        return "\n".join(lines)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ExperimentDomain',
    'HypothesisStatus',
    'SymbiosisHypothesis',
    'SymbiosisGraph',
]
