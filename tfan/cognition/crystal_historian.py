"""
Crystal Historian - Wisdom Oracle for the Council
==================================================

This bridges the Crystalline Core to the Quadamerl Council.

The Historian agent can now ask:
- "Have we seen this situation before?"
- "What happened last time?"
- "Any scars I should warn about?"

And get fast, structured answers backed by hyperdimensional similarity.

Integration:
    - Council asks Historian for historical context
    - Historian queries CrystalMemory for similar episodes
    - Returns structured wisdom for the Executive to consider

This is NOT a replacement for the text-based Hippocampus logs.
It's a fast index that points to relevant history.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from tfan.cognition.crystal_memory import (
    CrystalMemory,
    Episode,
    QueryResult,
    get_crystal_memory,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Wisdom Structures
# =============================================================================

@dataclass
class WisdomReport:
    """
    A structured wisdom report from the Crystal.

    This is what the Historian hands to the Council.
    """
    # Was anything relevant found?
    has_precedent: bool

    # Summary for the Executive
    summary: str

    # Detailed findings
    similar_episodes: List[QueryResult]

    # Specific warnings
    warnings: List[str]

    # Confidence in this assessment (0-1)
    confidence: float

    # Suggested caution level (0-1, higher = more careful)
    suggested_caution: float

    def to_council_message(self) -> str:
        """Format as a message for the Council."""
        parts = []

        if not self.has_precedent:
            parts.append("No strong precedent found in memory.")
            return " ".join(parts)

        parts.append(f"**Historian Report** (confidence: {self.confidence:.0%})")
        parts.append(self.summary)

        if self.warnings:
            parts.append("\n**Warnings:**")
            for w in self.warnings:
                parts.append(f"  ⚠️ {w}")

        if self.suggested_caution > 0.5:
            parts.append(f"\n**Suggested caution level: {self.suggested_caution:.0%}**")

        return "\n".join(parts)


# =============================================================================
# Crystal Historian
# =============================================================================

class CrystalHistorian:
    """
    The Historian's interface to crystal memory.

    This wraps CrystalMemory with higher-level queries suitable for
    the Council's decision-making process.
    """

    def __init__(self, memory: Optional[CrystalMemory] = None):
        self.memory = memory or get_crystal_memory()
        self.log = logging.getLogger("CrystalHistorian")

    # =========================================================================
    # Recording (called by other daemons to log experiences)
    # =========================================================================

    def record_experience(
        self,
        context: str,
        action: str,
        outcome: str,
        emotion: str,
        pain: float = 0.0,
        pleasure: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> Episode:
        """
        Record an experience to crystal memory.

        Called by:
        - Brainstem after significant events
        - Council after decisions
        - Scar tissue system after failures
        """
        return self.memory.record_episode(
            context=context,
            action=action,
            outcome=outcome,
            emotion=emotion,
            pain=pain,
            pleasure=pleasure,
            details=details,
        )

    def record_scar(
        self,
        context: str,
        action: str,
        outcome: str,
        lesson: str,
    ) -> Episode:
        """
        Record a painful lesson as scar tissue.

        This is high-priority memory that should surface in future similar situations.
        """
        return self.memory.record_scar(
            context=context,
            action=action,
            outcome=outcome,
            lesson=lesson,
        )

    # =========================================================================
    # Querying (called by Council during deliberation)
    # =========================================================================

    def consult(
        self,
        context: str,
        proposed_action: Optional[str] = None,
        current_emotion: Optional[str] = None,
    ) -> WisdomReport:
        """
        Consult the crystal for wisdom about a situation.

        This is the main interface for the Council.

        Args:
            context: What situation are we in?
            proposed_action: What are we thinking of doing?
            current_emotion: What's the current emotional state?

        Returns:
            WisdomReport with relevant history and warnings
        """
        self.log.info(f"Consulting crystal: context={context}, action={proposed_action}")

        # Query for similar situations
        similar = self.memory.query_similar(
            context=context,
            action=proposed_action,
            emotion=current_emotion,
            top_k=5,
            min_similarity=0.15,
        )

        # Query specifically for scars
        scars = self.memory.query_scars(
            context=context,
            action=proposed_action,
            top_k=3,
        )

        # Build report
        return self._build_report(context, proposed_action, similar, scars)

    def check_for_warnings(
        self,
        context: str,
        action: str,
    ) -> List[str]:
        """
        Quick check for warnings before taking an action.

        Returns list of warning strings (empty if none).
        """
        scars = self.memory.query_scars(
            context=context,
            action=action,
            top_k=3,
        )

        warnings = []
        for result in scars:
            ep = result.episode
            if result.similarity > 0.4:
                lesson = ep.details.get('lesson', f"Outcome was {ep.outcome}")
                warnings.append(f"Similar to past experience ({result.similarity:.0%} match): {lesson}")

        return warnings

    def get_context_wisdom(self, context: str) -> Dict[str, Any]:
        """
        Get accumulated wisdom about a specific context.

        Useful for: "What do we know about investor demos in general?"
        """
        summary = self.memory.get_context_summary(context)

        # Add wisdom interpretation
        if summary['episodes'] == 0:
            summary['wisdom'] = "No prior experience with this context."
        elif summary['avg_pain'] > 0.5:
            summary['wisdom'] = f"This context has been challenging (avg pain: {summary['avg_pain']:.0%}). Proceed carefully."
        elif summary['avg_pleasure'] > 0.5:
            summary['wisdom'] = f"This context has been rewarding (avg pleasure: {summary['avg_pleasure']:.0%}). Continue approach."
        else:
            summary['wisdom'] = "Mixed experiences in this context."

        return summary

    # =========================================================================
    # Report Building
    # =========================================================================

    def _build_report(
        self,
        context: str,
        proposed_action: Optional[str],
        similar: List[QueryResult],
        scars: List[QueryResult],
    ) -> WisdomReport:
        """Build a WisdomReport from query results."""

        # No precedent
        if not similar and not scars:
            return WisdomReport(
                has_precedent=False,
                summary="No similar experiences found in memory.",
                similar_episodes=[],
                warnings=[],
                confidence=0.0,
                suggested_caution=0.3,  # Default moderate caution for unknowns
            )

        # Build summary
        summary_parts = []

        if similar:
            best = similar[0]
            summary_parts.append(
                f"Found {len(similar)} similar experience(s). "
                f"Best match: {best.episode.context}/{best.episode.action} → {best.episode.outcome} "
                f"({best.similarity:.0%} similarity)"
            )

        # Build warnings from scars
        warnings = []
        for scar in scars:
            ep = scar.episode
            lesson = ep.details.get('lesson', f"resulted in {ep.outcome}")
            warnings.append(f"Previous similar situation ({scar.similarity:.0%} match) {lesson}")

        # Calculate confidence (more data + higher similarity = more confident)
        if similar:
            max_sim = max(r.similarity for r in similar)
            confidence = min(0.9, max_sim + len(similar) * 0.05)
        else:
            confidence = 0.3

        # Calculate suggested caution
        if scars:
            max_pain = max(s.episode.pain for s in scars)
            suggested_caution = min(0.9, max_pain + len(scars) * 0.1)
        elif similar:
            avg_pain = sum(r.episode.pain for r in similar) / len(similar)
            suggested_caution = avg_pain
        else:
            suggested_caution = 0.3

        return WisdomReport(
            has_precedent=True,
            summary=" ".join(summary_parts),
            similar_episodes=similar,
            warnings=warnings,
            confidence=confidence,
            suggested_caution=suggested_caution,
        )

    # =========================================================================
    # Integration helpers
    # =========================================================================

    def format_for_council_prompt(
        self,
        context: str,
        proposed_action: Optional[str] = None,
    ) -> str:
        """
        Format crystal wisdom as text for inclusion in a Council prompt.

        This is what gets injected into the Historian's contribution.
        """
        report = self.consult(context, proposed_action)

        if not report.has_precedent:
            return "**Crystal Memory**: No relevant precedent found."

        lines = [
            "**Crystal Memory Report**",
            report.summary,
        ]

        if report.warnings:
            lines.append("")
            lines.append("**Warnings from past experience:**")
            for w in report.warnings:
                lines.append(f"- {w}")

        lines.append("")
        lines.append(f"Suggested caution level: {report.suggested_caution:.0%}")

        return "\n".join(lines)


# =============================================================================
# Convenience: Global Instance
# =============================================================================

_default_historian: Optional[CrystalHistorian] = None


def get_crystal_historian() -> CrystalHistorian:
    """Get or create the default CrystalHistorian instance."""
    global _default_historian
    if _default_historian is None:
        _default_historian = CrystalHistorian()
    return _default_historian


# =============================================================================
# Council Integration Example
# =============================================================================

def historian_contribution(context: str, proposed_action: str) -> str:
    """
    Generate the Historian's contribution to a Council deliberation.

    This is what gets called when the Council asks:
    "Historian, what does history tell us about this situation?"
    """
    historian = get_crystal_historian()

    # Get crystal-based wisdom
    crystal_wisdom = historian.format_for_council_prompt(context, proposed_action)

    # Get general context wisdom
    context_wisdom = historian.get_context_wisdom(context)

    parts = [
        "**Historian's Analysis**",
        "",
        crystal_wisdom,
        "",
        f"**Context summary**: {context_wisdom.get('wisdom', 'No prior data.')}",
    ]

    if context_wisdom.get('episodes', 0) > 0:
        parts.append(f"(Based on {context_wisdom['episodes']} prior experience(s) in this context)")

    return "\n".join(parts)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'CrystalHistorian',
    'WisdomReport',
    'get_crystal_historian',
    'historian_contribution',
]
