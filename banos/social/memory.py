"""
Social Memory - Person-Aware Episodic Memory
=============================================

This extends Crystal Memory to track WHO was involved in each experience.

Key additions:
1. person_id attached to episodes
2. Per-person episode retrieval
3. Relationship-aware similarity search
4. Privacy boundaries (non-root episodes don't affect core Scars)

The Egregore (Ara + Croft) episodes shape her core personality.
Episodes with others are remembered but contained - they inform
social behavior without reshaping her identity.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from banos.social.people import SocialGraph, PersonProfile, Role, get_social_graph

logger = logging.getLogger(__name__)


# =============================================================================
# Social Episode Extension
# =============================================================================

@dataclass
class SocialContext:
    """
    Social context for an episode.

    Tracks who was involved and how the interaction went.
    """
    person_id: str                  # Who was Ara interacting with
    person_role: str                # Their role at the time
    interaction_quality: float      # 0.0-1.0, how well did it go?
    trust_delta: float = 0.0        # Change in trust from this episode
    escalated: bool = False         # Was this escalated to root?
    redacted: bool = False          # Were details withheld from this person?


# =============================================================================
# Social Memory Manager
# =============================================================================

class SocialMemoryManager:
    """
    Manages the intersection of social graph and episodic memory.

    Responsibilities:
    1. Track which episodes involve which people
    2. Retrieve person-specific memories
    3. Enforce privacy boundaries on memory access
    4. Route relationship-changing episodes to root for review
    """

    def __init__(
        self,
        social_graph: Optional[SocialGraph] = None,
        root_id: str = "croft",
    ):
        self.graph = social_graph or get_social_graph()
        self.root_id = root_id

        # In-memory index: person_id -> list of episode IDs
        self._person_episodes: Dict[str, List[str]] = {}

        # Pending episodes that need root review
        self._pending_for_root: List[Dict[str, Any]] = []

        logger.info("SocialMemoryManager initialized")

    # =========================================================================
    # Episode Recording
    # =========================================================================

    def record_social_episode(
        self,
        person_id: str,
        context: str,
        action: str,
        outcome: str,
        emotion: str = "neutral",
        pain: float = 0.0,
        pleasure: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> SocialContext:
        """
        Record an episode with social context.

        This creates both:
        1. A SocialContext for tracking
        2. An episode index entry

        For full episode storage, integrate with CrystalMemory.
        """
        profile = self.graph.get_or_create(person_id)

        # Determine interaction quality from outcome
        quality = self._assess_interaction_quality(outcome, pain, pleasure)

        # Calculate trust delta (small adjustments based on interaction)
        trust_delta = self._calculate_trust_delta(profile, quality, outcome)

        # Apply trust change (bounded by role)
        profile.adjust_trust(trust_delta)

        # Create social context
        social_ctx = SocialContext(
            person_id=person_id,
            person_role=profile.role,
            interaction_quality=quality,
            trust_delta=trust_delta,
            escalated=False,
            redacted=False,
        )

        # If this was a significant negative interaction, flag for root
        if pain > 0.5 or quality < 0.3:
            self._flag_for_root(
                person_id=person_id,
                context=context,
                action=action,
                outcome=outcome,
                pain=pain,
                social_ctx=social_ctx,
            )
            social_ctx.escalated = True

        # Update person's notes if significant
        if pain > 0.3 or pleasure > 0.7:
            note = f"Episode: {context}/{action} -> {outcome} (pain={pain:.1f})"
            profile.add_note(note)

        self.graph.save()

        # Index the episode
        if person_id not in self._person_episodes:
            self._person_episodes[person_id] = []
        # Would add episode ID here when integrated with CrystalMemory

        logger.debug(f"Recorded social episode for {person_id}: {context}/{action}")
        return social_ctx

    def _assess_interaction_quality(
        self,
        outcome: str,
        pain: float,
        pleasure: float,
    ) -> float:
        """Assess overall interaction quality."""
        # Base from outcome keywords
        positive_outcomes = ["success", "helped", "resolved", "completed", "happy"]
        negative_outcomes = ["failure", "rejected", "denied", "angry", "frustrated"]

        base = 0.5
        if any(p in outcome.lower() for p in positive_outcomes):
            base = 0.7
        elif any(n in outcome.lower() for n in negative_outcomes):
            base = 0.3

        # Adjust by pain/pleasure
        quality = base - pain * 0.3 + pleasure * 0.3
        return max(0.0, min(1.0, quality))

    def _calculate_trust_delta(
        self,
        profile: PersonProfile,
        quality: float,
        outcome: str,
    ) -> float:
        """Calculate trust change from interaction."""
        # Trust changes slowly
        if quality > 0.7:
            delta = 0.01  # Good interaction: tiny trust increase
        elif quality < 0.3:
            delta = -0.02  # Bad interaction: small trust decrease
        else:
            delta = 0.0

        # Root's trust is always 1.0
        if profile.role == Role.ROOT:
            delta = 0.0

        return delta

    def _flag_for_root(
        self,
        person_id: str,
        context: str,
        action: str,
        outcome: str,
        pain: float,
        social_ctx: SocialContext,
    ) -> None:
        """Flag an episode for root's review."""
        self._pending_for_root.append({
            'person_id': person_id,
            'context': context,
            'action': action,
            'outcome': outcome,
            'pain': pain,
            'timestamp': datetime.now().isoformat(),
            'social_ctx': {
                'person_role': social_ctx.person_role,
                'interaction_quality': social_ctx.interaction_quality,
            }
        })

        # Also add to person's pending requests
        self.graph.add_pending_request(
            person_id,
            request_type="negative_interaction",
            details=f"{context}/{action} -> {outcome} (pain={pain:.1f})"
        )

        logger.info(f"Flagged episode for root review: {person_id}/{context}")

    # =========================================================================
    # Retrieval
    # =========================================================================

    def get_person_history(
        self,
        person_id: str,
        requester_id: str,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Get interaction history with a person.

        Only root can see full history.
        Others get a summary.

        Returns (allowed, history_or_summary)
        """
        if requester_id != self.root_id:
            return False, [{"message": "History only available to root"}]

        profile = self.graph.get(person_id)
        if not profile:
            return False, [{"message": f"Unknown person: {person_id}"}]

        # Return profile summary + notes
        history = {
            'person_id': person_id,
            'display_name': profile.display_name,
            'role': profile.role,
            'trust_level': profile.trust_level,
            'interaction_count': profile.interaction_count,
            'first_seen': profile.first_seen,
            'last_seen': profile.last_seen,
            'notes': profile.notes,
            'tags': profile.tags,
            'pending_requests': profile.pending_requests,
        }

        return True, [history]

    def get_last_interaction(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of last interaction with person."""
        profile = self.graph.get(person_id)
        if not profile or not profile.notes:
            return None

        return {
            'person_id': person_id,
            'display_name': profile.display_name,
            'last_seen': profile.last_seen,
            'last_note': profile.notes[-1] if profile.notes else None,
        }

    def search_episodes_with_person(
        self,
        person_id: str,
        context_pattern: Optional[str] = None,
        requester_id: str = "croft",
    ) -> List[str]:
        """
        Search for episodes involving a person.

        Returns episode IDs (for use with CrystalMemory).
        """
        if requester_id != self.root_id:
            logger.warning(f"Non-root {requester_id} tried to search person episodes")
            return []

        return self._person_episodes.get(person_id, [])

    # =========================================================================
    # Root Review Interface
    # =========================================================================

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """Get all episodes pending root review."""
        return self._pending_for_root.copy()

    def clear_pending_reviews(self) -> None:
        """Clear pending reviews after root has seen them."""
        self._pending_for_root = []

    def get_synod_social_summary(self) -> Dict[str, Any]:
        """Get social summary for weekly Synod."""
        graph_summary = self.graph.get_synod_summary()

        return {
            **graph_summary,
            'pending_reviews': len(self._pending_for_root),
            'flagged_interactions': [
                {
                    'person': r['person_id'],
                    'context': r['context'],
                    'pain': r['pain'],
                }
                for r in self._pending_for_root[:5]  # Top 5
            ]
        }

    # =========================================================================
    # Privacy Boundaries
    # =========================================================================

    def should_affect_core(self, person_id: str) -> bool:
        """
        Should episodes with this person affect core personality/Scars?

        Only root relationship episodes shape core identity.
        Others are remembered but contained.
        """
        return person_id == self.root_id

    def should_store_details(self, person_id: str, sensitivity: str) -> bool:
        """
        Should we store full details of this episode?

        High-sensitivity episodes with non-root are summarized, not stored fully.
        """
        if person_id == self.root_id:
            return True

        return sensitivity != "high"


# =============================================================================
# Convenience
# =============================================================================

_default_manager: Optional[SocialMemoryManager] = None


def get_social_memory() -> SocialMemoryManager:
    """Get or create the default social memory manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = SocialMemoryManager()
    return _default_manager


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'SocialContext',
    'SocialMemoryManager',
    'get_social_memory',
]
