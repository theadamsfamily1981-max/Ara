#!/usr/bin/env python3
"""
THOUGHT LOOP - Ara's Complete Response Pipeline
================================================

Bio-Affective Neuromorphic Operating System
This module wires together all the learning components into a coherent flow.

The Thought Loop:
1. SENSE - Read user request + somatic state
2. PLAN - Use memory to choose approach (MetaPlanner)
3. CHECK - Audit the plan against learned policies (AntibodyAuditor)
4. ACT - Generate response with guided tool/style
5. REMEMBER - Store episode with user outcome
6. LEARN - Update preferences from feedback

This is the missing wiring harness that connects:
- EpisodicMemory (long-term memory)
- CroftModel (user preferences)
- ScarRegistry (learned policies)
- MetaPlanner (decision making)
- AntibodyAuditor (sanity checking)
- Dreamer (consolidation)

Usage:
    thought_loop = ThoughtLoop()

    # Before generating response:
    plan = thought_loop.before_response(user_input, context)
    # plan.tool, plan.style, plan.warnings

    # After generating response:
    episode_id = thought_loop.after_response(
        user_input, response, plan, latency_s, tokens_used
    )

    # When user gives feedback (explicit or implicit):
    thought_loop.record_feedback(episode_id, rating, friction_flags)

    # During sleep (called by Dreamer):
    thought_loop.consolidate()
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Import components with graceful fallbacks
COMPONENTS = {
    'episodic': False,
    'croft': False,
    'scar': False,
    'meta': False,
    'mies': False,
}

try:
    from .episodic_memory import (
        EpisodicMemory,
        Episode,
        UserOutcome,
        FrictionFlag,
        get_episodic_memory,
        EMBEDDING_DIM,
    )
    COMPONENTS['episodic'] = True
except ImportError as e:
    logger.warning(f"EpisodicMemory not available: {e}")
    Episode = None
    UserOutcome = None

try:
    from .croft_model import CroftModel, CONTEXT_TYPES, TOOLS, STYLES
    COMPONENTS['croft'] = True
except ImportError as e:
    logger.warning(f"CroftModel not available: {e}")
    CroftModel = None

try:
    from .scar_tissue import (
        ScarRegistry,
        ScarTissue,
        get_scar_registry,
        create_scar_from_episode,
        create_scar_from_user_feedback,
    )
    COMPONENTS['scar'] = True
except ImportError as e:
    logger.warning(f"ScarRegistry not available: {e}")
    ScarRegistry = None

try:
    from .meta_planner import (
        MetaPlanner,
        MetaPlan,
        RequestContext,
        get_meta_planner,
    )
    COMPONENTS['meta'] = True
except ImportError as e:
    logger.warning(f"MetaPlanner not available: {e}")
    MetaPlanner = None
    MetaPlan = None
    RequestContext = None

# MIES components (optional)
try:
    import sys
    _mies_path = Path(__file__).parent.parent.parent / "multi-ai-workspace" / "src"
    if _mies_path.exists():
        sys.path.insert(0, str(_mies_path))

    from integrations.mies.policy.antibody_auditor import (
        AntibodyAuditor,
        AuditVerdict,
        AuditResult,
        create_antibody_auditor,
    )
    from integrations.mies.history import InteractionHistory
    from integrations.mies.context import ModalityContext
    from integrations.mies.modes import ModalityMode, ModalityDecision
    COMPONENTS['mies'] = True
except ImportError as e:
    logger.debug(f"MIES components not available: {e}")
    AntibodyAuditor = None

import numpy as np


# =============================================================================
# Friction Detection
# =============================================================================

def detect_friction(
    latency_s: float,
    tokens_used: int,
    response_text: str,
    user_had_to_repeat: bool = False,
    user_interrupted: bool = False,
) -> List[str]:
    """
    Detect friction flags from response characteristics.

    This is where we infer user annoyance from signals.
    """
    flags = []

    # Latency-based friction
    if latency_s > 10.0:
        flags.append("too_slow")

    # Token-based friction
    if tokens_used > 2000:
        flags.append("too_verbose")
    elif tokens_used < 50 and len(response_text) < 100:
        flags.append("too_terse")

    # Explicit signals
    if user_had_to_repeat:
        flags.append("had_to_repeat")
    if user_interrupted:
        flags.append("interrupted")

    # Content-based detection
    response_lower = response_text.lower()
    if "i apologize" in response_lower or "sorry, i" in response_lower:
        if "misunderstand" in response_lower or "misread" in response_lower:
            flags.append("missed_intent")

    if "i cannot" in response_lower or "i'm unable" in response_lower:
        if "earlier" in response_lower or "before" in response_lower:
            flags.append("over_promised")

    return flags


def infer_rating_from_signals(
    user_response: Optional[str],
    time_to_respond_s: float,
    continued_conversation: bool,
) -> int:
    """
    Infer user rating from implicit signals.

    Returns: -1 (negative), 0 (neutral), +1 (positive)
    """
    if user_response:
        response_lower = user_response.lower()

        # Positive signals
        positive_words = ["thanks", "perfect", "great", "exactly", "love", "awesome", "nice"]
        if any(word in response_lower for word in positive_words):
            return 1

        # Negative signals
        negative_words = ["wrong", "no", "not what", "terrible", "useless", "bad", "again"]
        if any(word in response_lower for word in negative_words):
            return -1

    # Time-based inference
    if time_to_respond_s < 1.0 and not continued_conversation:
        # Very quick response + conversation stopped = negative
        return -1

    if continued_conversation:
        # They kept talking = at least neutral
        return 0

    return 0  # Default neutral


# =============================================================================
# Thought Loop
# =============================================================================

@dataclass
class ThoughtLoopConfig:
    """Configuration for the thought loop."""
    enable_croft: bool = True
    enable_scars: bool = True
    enable_auditor: bool = True
    enable_memory: bool = True
    min_importance_to_store: float = 0.3
    feedback_timeout_s: float = 60.0  # How long to wait for feedback


@dataclass
class PendingEpisode:
    """An episode awaiting user feedback."""
    episode_id: int
    plan: 'MetaPlan'
    request_text: str
    response_text: str
    timestamp: float
    latency_s: float
    tokens_used: int


class ThoughtLoop:
    """
    The complete thought → response → learning loop.

    This is the integration layer that makes all the components work together.
    """

    def __init__(self, config: Optional[ThoughtLoopConfig] = None):
        self.config = config or ThoughtLoopConfig()

        # Initialize components
        self._memory = None
        self._croft = None
        self._scars = None
        self._planner = None
        self._auditor = None
        self._mies_history = None

        self._init_components()

        # Pending episodes awaiting feedback
        self._pending: Dict[int, PendingEpisode] = {}
        self._last_episode_id: Optional[int] = None

        # Session statistics
        self._session_start = time.time()
        self._session_ratings: List[int] = []
        self._session_friction: List[str] = []

        logger.info(f"ThoughtLoop initialized. Components: {COMPONENTS}")

    def _init_components(self) -> None:
        """Initialize all components with graceful degradation."""
        if self.config.enable_memory and COMPONENTS['episodic']:
            try:
                self._memory = get_episodic_memory()
                logger.info("EpisodicMemory connected")
            except Exception as e:
                logger.warning(f"EpisodicMemory init failed: {e}")

        if self.config.enable_croft and COMPONENTS['croft']:
            try:
                self._croft = CroftModel()
                logger.info("CroftModel loaded")
            except Exception as e:
                logger.warning(f"CroftModel init failed: {e}")

        if self.config.enable_scars and COMPONENTS['scar']:
            try:
                self._scars = get_scar_registry()
                logger.info(f"ScarRegistry loaded ({len(self._scars._scars)} scars)")
            except Exception as e:
                logger.warning(f"ScarRegistry init failed: {e}")

        if COMPONENTS['meta']:
            try:
                self._planner = MetaPlanner(
                    croft_model=self._croft,
                    scar_registry=self._scars,
                    episodic_memory=self._memory,
                )
                logger.info("MetaPlanner ready")
            except Exception as e:
                logger.warning(f"MetaPlanner init failed: {e}")

        if self.config.enable_auditor and COMPONENTS['mies']:
            try:
                self._mies_history = InteractionHistory()
                self._auditor = create_antibody_auditor(
                    history=self._mies_history,
                    strict=False,
                )
                logger.info("AntibodyAuditor ready")
            except Exception as e:
                logger.warning(f"AntibodyAuditor init failed: {e}")

    # =========================================================================
    # Main Loop Methods
    # =========================================================================

    def before_response(
        self,
        user_input: str,
        somatic_context: Optional[Dict[str, Any]] = None,
    ) -> 'MetaPlan':
        """
        Plan the response before generating it.

        Args:
            user_input: The user's request
            somatic_context: Optional HAL state (PAD, load, etc.)

        Returns:
            MetaPlan with tool, style, and warnings
        """
        # Default plan if no planner available
        if not self._planner:
            return MetaPlan(
                tool="direct_answer",
                style="concise",
                confidence=0.1,
                rationale="No planner available",
            ) if MetaPlan else None

        # Build request context
        context = RequestContext(
            request_text=user_input,
            session_length=len(self._session_ratings),
            recent_friction=self._session_friction[-5:],
            recent_ratings=self._session_ratings[-5:],
        )

        # Add somatic context if available
        if somatic_context:
            context.user_cognitive_load = somatic_context.get('user_load', 0.5)
            context.system_load = somatic_context.get('system_load', 0.5)
            context.latency_budget_s = somatic_context.get('latency_budget', 5.0)

        # Get plan from planner
        plan = self._planner.plan(user_input, context)

        # Apply auditor if available
        if self._auditor and COMPONENTS['mies']:
            plan = self._apply_auditor(plan, context)

        logger.debug(
            f"Plan: {plan.tool}/{plan.style} "
            f"(confidence={plan.confidence:.2f}, scars={len(plan.active_scars)})"
        )

        return plan

    def _apply_auditor(
        self,
        plan: 'MetaPlan',
        context: 'RequestContext',
    ) -> 'MetaPlan':
        """Apply antibody auditor to the plan."""
        if not self._auditor or not ModalityContext:
            return plan

        try:
            # Convert plan to ModalityContext for auditor
            # This is a simplified conversion - full implementation would
            # map tool/style to modality modes
            pass  # TODO: Full ModalityContext mapping

        except Exception as e:
            logger.warning(f"Auditor application failed: {e}")

        return plan

    def after_response(
        self,
        user_input: str,
        response_text: str,
        plan: Optional['MetaPlan'],
        latency_s: float,
        tokens_used: int,
        pad_state: Optional[Dict[str, float]] = None,
        hardware_state: Optional[Dict[str, float]] = None,
    ) -> Optional[int]:
        """
        Store the episode after response generation.

        Args:
            user_input: The user's request
            response_text: Ara's response
            plan: The plan that was used (if any)
            latency_s: Response generation time
            tokens_used: Token count
            pad_state: PAD state during generation
            hardware_state: Hardware state during generation

        Returns:
            Episode ID (for later feedback recording)
        """
        if not self._memory or not Episode or not UserOutcome:
            return None

        # Detect friction from response characteristics
        friction = detect_friction(latency_s, tokens_used, response_text)

        # Build user outcome (partial - rating comes from feedback)
        user_outcome = UserOutcome(
            rating=0,  # Unknown until feedback
            latency_s=latency_s,
            tokens_used=tokens_used,
            friction_flags=friction,
            tool_used=plan.tool if plan else "",
            style_used=plan.style if plan else "",
        )

        # Compute importance
        importance = 0.5
        if plan and plan.confidence > 0.7:
            importance += 0.1  # Higher confidence = more important to track
        if friction:
            importance += 0.2  # Friction = important to learn from
        if plan and plan.active_scars:
            importance += 0.1  # Scar-constrained = important

        # Build episode
        content = f"User: {user_input}\n\nAra: {response_text}"

        episode = Episode(
            content=content,
            vector=self._memory.encode(content),
            pad_p=pad_state.get('p', 0) if pad_state else 0,
            pad_a=pad_state.get('a', 0) if pad_state else 0,
            pad_d=pad_state.get('d', 0) if pad_state else 0,
            cpu_load=hardware_state.get('cpu_load', 0) if hardware_state else 0,
            gpu_load=hardware_state.get('gpu_load', 0) if hardware_state else 0,
            importance=importance,
            episode_type=plan.tool if plan else "interaction",
            user_outcome=user_outcome,
        )

        try:
            episode_id = self._memory.store(episode)

            if episode_id:
                # Track pending episode for feedback
                self._pending[episode_id] = PendingEpisode(
                    episode_id=episode_id,
                    plan=plan,
                    request_text=user_input,
                    response_text=response_text,
                    timestamp=time.time(),
                    latency_s=latency_s,
                    tokens_used=tokens_used,
                )
                self._last_episode_id = episode_id

                logger.debug(f"Stored episode {episode_id} (importance={importance:.2f})")

            return episode_id

        except Exception as e:
            logger.error(f"Failed to store episode: {e}")
            return None

    def record_feedback(
        self,
        episode_id: Optional[int] = None,
        rating: Optional[int] = None,
        friction_flags: Optional[List[str]] = None,
        user_response: Optional[str] = None,
    ) -> None:
        """
        Record user feedback for an episode.

        Args:
            episode_id: Episode to update (default: last episode)
            rating: Explicit rating (-1, 0, +1)
            friction_flags: Additional friction flags
            user_response: User's next message (for implicit rating inference)
        """
        # Use last episode if not specified
        if episode_id is None:
            episode_id = self._last_episode_id

        if episode_id is None or episode_id not in self._pending:
            return

        pending = self._pending.get(episode_id)
        if not pending:
            return

        # Infer rating if not explicit
        if rating is None and user_response:
            time_since = time.time() - pending.timestamp
            continued = len(user_response) > 10
            rating = infer_rating_from_signals(user_response, time_since, continued)

        rating = rating or 0

        # Combine friction flags
        all_friction = list(set(
            (friction_flags or []) +
            detect_friction(
                pending.latency_s,
                pending.tokens_used,
                pending.response_text,
            )
        ))

        # Record to memory
        if self._memory:
            try:
                self._memory.record_user_outcome(
                    episode_id=episode_id,
                    rating=rating,
                    notes=user_response[:200] if user_response else "",
                    latency_s=pending.latency_s,
                    tokens_used=pending.tokens_used,
                    friction_flags=all_friction,
                    tool_used=pending.plan.tool if pending.plan else "",
                    style_used=pending.plan.style if pending.plan else "",
                )
                logger.debug(f"Recorded feedback for episode {episode_id}: rating={rating}")
            except Exception as e:
                logger.error(f"Failed to record feedback: {e}")

        # Update session statistics
        self._session_ratings.append(rating)
        self._session_friction.extend(all_friction)

        # Update planner session state
        if self._planner:
            self._planner.record_outcome(rating, all_friction)

        # Create scar if strongly negative
        if rating == -1 and all_friction and self._scars:
            self._create_scar_from_negative(pending, all_friction)

        # Clean up pending
        del self._pending[episode_id]

    def _create_scar_from_negative(
        self,
        pending: PendingEpisode,
        friction_flags: List[str],
    ) -> None:
        """Create a scar from negative feedback."""
        if not self._scars or not COMPONENTS['scar']:
            return

        try:
            scar = create_scar_from_user_feedback(
                context_type=pending.plan.tool if pending.plan else "interaction",
                tool_used=pending.plan.tool if pending.plan else "",
                style_used=pending.plan.style if pending.plan else "",
                rating=-1,
                friction_flags=friction_flags,
                notes=pending.request_text[:100],
            )
            self._scars.add_scar(scar)
            logger.info(f"Created scar from negative feedback: {scar.id}")
        except Exception as e:
            logger.error(f"Failed to create scar: {e}")

    def consolidate(self) -> Dict[str, Any]:
        """
        Consolidate learning (called during sleep/REM).

        Returns:
            Consolidation statistics
        """
        stats = {
            'croft_trained': False,
            'scars_created': 0,
            'memory_pruned': 0,
        }

        # Train Croft model
        if self._croft and self._memory:
            try:
                result = self._croft.train_from_preferences(self._memory)
                stats['croft_trained'] = result.get('status') == 'trained'
                logger.info(f"Croft training: {result}")
            except Exception as e:
                logger.error(f"Croft training failed: {e}")

        # Prune old pending episodes
        cutoff = time.time() - self.config.feedback_timeout_s
        expired = [
            eid for eid, p in self._pending.items()
            if p.timestamp < cutoff
        ]
        for eid in expired:
            # Treat expired as neutral
            self.record_feedback(eid, rating=0)
            stats['memory_pruned'] += 1

        return stats

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session performance."""
        if not self._session_ratings:
            return {
                'duration_s': time.time() - self._session_start,
                'interactions': 0,
            }

        return {
            'duration_s': time.time() - self._session_start,
            'interactions': len(self._session_ratings),
            'avg_rating': sum(self._session_ratings) / len(self._session_ratings),
            'positive_count': sum(1 for r in self._session_ratings if r > 0),
            'negative_count': sum(1 for r in self._session_ratings if r < 0),
            'common_friction': list(set(self._session_friction)),
            'pending_feedback': len(self._pending),
        }

    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all components."""
        return {
            'episodic_memory': self._memory is not None,
            'croft_model': self._croft is not None,
            'scar_registry': self._scars is not None,
            'meta_planner': self._planner is not None,
            'antibody_auditor': self._auditor is not None,
            **COMPONENTS,
        }

    def reset_session(self) -> None:
        """Reset session state."""
        self._session_ratings = []
        self._session_friction = []
        self._session_start = time.time()
        self._pending.clear()

        if self._planner:
            self._planner.reset_session()


# =============================================================================
# Global Instance
# =============================================================================

_thought_loop: Optional[ThoughtLoop] = None


def get_thought_loop() -> ThoughtLoop:
    """Get the global thought loop instance."""
    global _thought_loop
    if _thought_loop is None:
        _thought_loop = ThoughtLoop()
    return _thought_loop


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Thought Loop - Ara's Response Pipeline")
    parser.add_argument("--status", action="store_true", help="Show component status")
    parser.add_argument("--test", type=str, help="Test planning for a request")
    args = parser.parse_args()

    loop = ThoughtLoop()

    if args.status:
        print("Component Status:")
        for comp, status in loop.get_component_status().items():
            icon = "✓" if status else "✗"
            print(f"  {icon} {comp}")

    elif args.test:
        print(f"Planning response for: {args.test}")
        plan = loop.before_response(args.test)
        if plan:
            print(f"  Tool: {plan.tool}")
            print(f"  Style: {plan.style}")
            print(f"  Confidence: {plan.confidence:.2f}")
            print(f"  Expected rating: {plan.expected_rating:.2f}")
            if plan.friction_risks:
                print(f"  Friction risks: {plan.friction_risks}")
            if plan.active_scars:
                print(f"  Active scars: {plan.active_scars}")
            print(f"  Rationale: {plan.rationale}")

    else:
        parser.print_help()
