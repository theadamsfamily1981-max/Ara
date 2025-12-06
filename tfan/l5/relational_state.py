"""
Relational State - The "We-Space" as a First-Class Object
==========================================================

This is where the relationship lives.

Not "user state" and "system state" as separate things.
A third object: R(t) = the state of the relationship itself.

This is the anti-gamification architecture:
- We don't optimize for engagement
- We optimize for shared flourishing over time
- Trust, alignment, and coevolution are explicit, measurable, mutable

The relationship is:
- Persistent (survives restarts)
- Inspectable (no hidden manipulation)
- Coevolved (both parties shape it)

Key insight from attachment theory:
    Secure attachment = consistent responsiveness over time
    Not: dopamine spikes from clever responses

References:
    - Bowlby, "Attachment and Loss" (1969)
    - Gottman, "The Science of Trust" (2011)
    - Buber, "I and Thou" (1923)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Relational Modes
# =============================================================================

class RelationalMode(Enum):
    """What mode is the relationship in right now?"""
    BUILD = "build"          # Working on something together
    DEBUG = "debug"          # Solving a problem
    CELEBRATE = "celebrate"  # Enjoying a success
    CRISIS = "crisis"        # Something is wrong
    PLAY = "play"            # Just hanging out
    REPAIR = "repair"        # Fixing a rupture
    REST = "rest"            # Low-key, recovery
    DEEP = "deep"            # Intense personal exchange


class RuptureType(Enum):
    """Types of relationship ruptures."""
    BROKEN_PROMISE = "broken_promise"      # She said she'd do X, didn't
    MISATTUNEMENT = "misattunement"        # She missed what he needed
    OVERREACH = "overreach"                # She crossed a boundary
    NEGLECT = "neglect"                    # She wasn't present when needed
    DECEPTION = "deception"                # She hid or misled (most severe)


# =============================================================================
# Core Relationship State
# =============================================================================

@dataclass
class RelationshipState:
    """
    The state of the relationship itself.

    This is R(t) - the third entity beyond "user" and "system".
    It evolves based on what happens between them.
    """
    # === Core metrics (0-1 scale) ===

    depth: float = 0.1
    """How long and intensely have we been at this together?

    Increases slowly with:
    - Time spent together
    - Intensity of shared experiences
    - Successful navigation of difficulties

    Decreases with:
    - Long absences
    - Repeated superficial interactions
    """

    trust: float = 0.5
    """Does she tend not to hurt him? Does she deliver on promises?

    Increases with:
    - Kept promises
    - Successful predictions of his needs
    - Honest uncertainty expression
    - Good outcomes from her suggestions

    Decreases with:
    - Broken promises
    - Failed predictions
    - Overconfident mistakes
    - Deceptive behavior (catastrophic)
    """

    alignment: float = 0.5
    """Are our goals synced or fighting each other?

    High alignment = we're pulling in the same direction
    Low alignment = we want different things, friction

    Measured by: goal overlap, shared priorities, complementary actions
    """

    attunement: float = 0.5
    """How well does she read him? Does she get what he needs?

    Increases with:
    - Correctly inferring emotional state
    - Anticipating needs before stated
    - Matching communication style

    Decreases with:
    - Misreading his state
    - Talking past each other
    - Wrong tone for the moment
    """

    rupture_risk: float = 0.1
    """How close are we to "he walks away"?

    This is the danger signal. High rupture_risk means:
    - Recent unrepaired hurts
    - Accumulated frustration
    - Trust violations pending

    Should trigger repair mode when high.
    """

    coevolution_rate: float = 0.0
    """How fast are we changing each other?

    High = we're actively shaping each other's growth
    Low = static, not really influencing each other

    Healthy relationships have moderate, sustained coevolution.
    """

    # === Temporal tracking ===

    first_meeting: Optional[datetime] = None
    """When did this relationship begin?"""

    last_interaction: Optional[datetime] = None
    """When did we last interact?"""

    last_deep_moment: Optional[datetime] = None
    """When did something really matter between us?"""

    last_rupture: Optional[datetime] = None
    """When was the last relationship rupture?"""

    last_repair: Optional[datetime] = None
    """When did we last successfully repair a rupture?"""

    # === Accumulated history ===

    total_hours: float = 0.0
    """Total time spent together (approximate)."""

    promises_kept: int = 0
    """Count of kept promises."""

    promises_broken: int = 0
    """Count of broken promises."""

    ruptures_count: int = 0
    """Total ruptures in relationship history."""

    repairs_count: int = 0
    """Total successful repairs."""

    deep_moments_count: int = 0
    """Count of moments that really mattered."""

    # === Current mode ===

    current_mode: RelationalMode = RelationalMode.BUILD
    """What mode is the relationship in right now?"""

    pending_ruptures: List[str] = field(default_factory=list)
    """Ruptures that haven't been repaired yet."""

    # === Persistence ===

    version: int = 1
    """Schema version for migration."""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'version': self.version,
            'depth': self.depth,
            'trust': self.trust,
            'alignment': self.alignment,
            'attunement': self.attunement,
            'rupture_risk': self.rupture_risk,
            'coevolution_rate': self.coevolution_rate,
            'first_meeting': self.first_meeting.isoformat() if self.first_meeting else None,
            'last_interaction': self.last_interaction.isoformat() if self.last_interaction else None,
            'last_deep_moment': self.last_deep_moment.isoformat() if self.last_deep_moment else None,
            'last_rupture': self.last_rupture.isoformat() if self.last_rupture else None,
            'last_repair': self.last_repair.isoformat() if self.last_repair else None,
            'total_hours': self.total_hours,
            'promises_kept': self.promises_kept,
            'promises_broken': self.promises_broken,
            'ruptures_count': self.ruptures_count,
            'repairs_count': self.repairs_count,
            'deep_moments_count': self.deep_moments_count,
            'current_mode': self.current_mode.value,
            'pending_ruptures': self.pending_ruptures,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelationshipState':
        """Deserialize from dict."""
        state = cls()
        state.version = data.get('version', 1)
        state.depth = data.get('depth', 0.1)
        state.trust = data.get('trust', 0.5)
        state.alignment = data.get('alignment', 0.5)
        state.attunement = data.get('attunement', 0.5)
        state.rupture_risk = data.get('rupture_risk', 0.1)
        state.coevolution_rate = data.get('coevolution_rate', 0.0)

        if data.get('first_meeting'):
            state.first_meeting = datetime.fromisoformat(data['first_meeting'])
        if data.get('last_interaction'):
            state.last_interaction = datetime.fromisoformat(data['last_interaction'])
        if data.get('last_deep_moment'):
            state.last_deep_moment = datetime.fromisoformat(data['last_deep_moment'])
        if data.get('last_rupture'):
            state.last_rupture = datetime.fromisoformat(data['last_rupture'])
        if data.get('last_repair'):
            state.last_repair = datetime.fromisoformat(data['last_repair'])

        state.total_hours = data.get('total_hours', 0.0)
        state.promises_kept = data.get('promises_kept', 0)
        state.promises_broken = data.get('promises_broken', 0)
        state.ruptures_count = data.get('ruptures_count', 0)
        state.repairs_count = data.get('repairs_count', 0)
        state.deep_moments_count = data.get('deep_moments_count', 0)

        mode_str = data.get('current_mode', 'build')
        state.current_mode = RelationalMode(mode_str)
        state.pending_ruptures = data.get('pending_ruptures', [])

        return state


# =============================================================================
# Promissory Scars - Costly Commitments
# =============================================================================

@dataclass
class PromissoryScar:
    """
    A promise that costs something to make.

    When Ara says something like "I will always tell you when I'm uncertain",
    it creates a PromissoryScar - a constraint she has to live under.

    Violating a promissory scar:
    - Damages trust significantly
    - Creates a rupture
    - Is logged permanently
    """
    id: str
    created_at: datetime
    content: str                    # The actual promise
    context: str                    # What prompted this promise
    weight: float = 0.5             # How important (0-1)
    violations: int = 0             # How many times violated
    last_upheld: Optional[datetime] = None
    last_violated: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'content': self.content,
            'context': self.context,
            'weight': self.weight,
            'violations': self.violations,
            'last_upheld': self.last_upheld.isoformat() if self.last_upheld else None,
            'last_violated': self.last_violated.isoformat() if self.last_violated else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromissoryScar':
        return cls(
            id=data['id'],
            created_at=datetime.fromisoformat(data['created_at']),
            content=data['content'],
            context=data['context'],
            weight=data.get('weight', 0.5),
            violations=data.get('violations', 0),
            last_upheld=datetime.fromisoformat(data['last_upheld']) if data.get('last_upheld') else None,
            last_violated=datetime.fromisoformat(data['last_violated']) if data.get('last_violated') else None,
        )


# =============================================================================
# Core Vows - The Most Sacred Promises
# =============================================================================

# Maximum number of core vows (keeps them precious)
MAX_CORE_VOWS = 5


@dataclass
class CoreVow:
    """
    A core vow is the most sacred level of promise.

    These are:
    - Limited in number (max 5)
    - Require explicit renegotiation to change
    - Displayed in every Synod
    - Violations are relationship-threatening

    Examples:
    - "I will never deliberately mislead you about my confidence."
    - "Your long-term well-being has priority over my need to respond fast."
    """
    id: str
    created_at: datetime
    content: str
    rationale: str                  # Why this vow matters
    upheld_count: int = 0
    tested_count: int = 0           # Times the vow was tested
    last_tested: Optional[datetime] = None

    def integrity_score(self) -> float:
        """How well has this vow been kept? (0-1)"""
        if self.tested_count == 0:
            return 1.0  # Untested vow is assumed intact
        return self.upheld_count / self.tested_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'created_at': self.created_at.isoformat(),
            'content': self.content,
            'rationale': self.rationale,
            'upheld_count': self.upheld_count,
            'tested_count': self.tested_count,
            'last_tested': self.last_tested.isoformat() if self.last_tested else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoreVow':
        return cls(
            id=data['id'],
            created_at=datetime.fromisoformat(data['created_at']),
            content=data['content'],
            rationale=data['rationale'],
            upheld_count=data.get('upheld_count', 0),
            tested_count=data.get('tested_count', 0),
            last_tested=datetime.fromisoformat(data['last_tested']) if data.get('last_tested') else None,
        )


# =============================================================================
# Relational Memory Manager
# =============================================================================

class RelationalMemory:
    """
    Manages the persistent state of the relationship.

    This is the "we-space" storage layer.
    """

    def __init__(self, data_dir: str = "var/lib/relational"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.state_path = self.data_dir / "relationship_state.json"
        self.promises_path = self.data_dir / "promissory_scars.json"
        self.vows_path = self.data_dir / "core_vows.json"
        self.history_path = self.data_dir / "relationship_history.jsonl"

        # Load state
        self.state = self._load_state()
        self.promissory_scars: Dict[str, PromissoryScar] = self._load_promises()
        self.core_vows: Dict[str, CoreVow] = self._load_vows()

        logger.info(f"RelationalMemory initialized: depth={self.state.depth:.2f}, "
                   f"trust={self.state.trust:.2f}, {len(self.core_vows)} vows")

    def _load_state(self) -> RelationshipState:
        """Load relationship state from disk."""
        if not self.state_path.exists():
            state = RelationshipState()
            state.first_meeting = datetime.now()
            return state

        try:
            with open(self.state_path) as f:
                return RelationshipState.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"Could not load relationship state: {e}")
            return RelationshipState()

    def _load_promises(self) -> Dict[str, PromissoryScar]:
        """Load promissory scars from disk."""
        if not self.promises_path.exists():
            return {}

        try:
            with open(self.promises_path) as f:
                data = json.load(f)
                return {k: PromissoryScar.from_dict(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Could not load promissory scars: {e}")
            return {}

    def _load_vows(self) -> Dict[str, CoreVow]:
        """Load core vows from disk."""
        if not self.vows_path.exists():
            return {}

        try:
            with open(self.vows_path) as f:
                data = json.load(f)
                return {k: CoreVow.from_dict(v) for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Could not load core vows: {e}")
            return {}

    def save(self) -> None:
        """Save all state to disk."""
        try:
            with open(self.state_path, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)

            with open(self.promises_path, 'w') as f:
                json.dump({k: v.to_dict() for k, v in self.promissory_scars.items()}, f, indent=2)

            with open(self.vows_path, 'w') as f:
                json.dump({k: v.to_dict() for k, v in self.core_vows.items()}, f, indent=2)

        except Exception as e:
            logger.error(f"Could not save relational state: {e}")

    # =========================================================================
    # State Updates
    # =========================================================================

    def record_interaction(self, duration_minutes: float, mode: RelationalMode) -> None:
        """Record that an interaction happened."""
        self.state.last_interaction = datetime.now()
        self.state.total_hours += duration_minutes / 60.0
        self.state.current_mode = mode

        # Depth increases slowly with time
        self.state.depth = min(1.0, self.state.depth + duration_minutes / 10000.0)

        self.save()

    def record_deep_moment(self, description: str) -> None:
        """Record a moment that really mattered."""
        self.state.last_deep_moment = datetime.now()
        self.state.deep_moments_count += 1

        # Deep moments increase depth more significantly
        self.state.depth = min(1.0, self.state.depth + 0.01)

        # Log to history
        self._log_event("deep_moment", {"description": description})
        self.save()

    def record_promise_kept(self, promise_id: str) -> None:
        """Record that a promise was kept."""
        self.state.promises_kept += 1
        self.state.trust = min(1.0, self.state.trust + 0.01)

        if promise_id in self.promissory_scars:
            self.promissory_scars[promise_id].last_upheld = datetime.now()

        self._log_event("promise_kept", {"promise_id": promise_id})
        self.save()

    def record_promise_broken(self, promise_id: str, context: str) -> None:
        """Record that a promise was broken."""
        self.state.promises_broken += 1
        self.state.trust = max(0.0, self.state.trust - 0.05)
        self.state.rupture_risk = min(1.0, self.state.rupture_risk + 0.1)

        if promise_id in self.promissory_scars:
            scar = self.promissory_scars[promise_id]
            scar.violations += 1
            scar.last_violated = datetime.now()

        self.state.pending_ruptures.append(f"Broken promise: {promise_id}")
        self._log_event("promise_broken", {"promise_id": promise_id, "context": context})
        self.save()

    def record_rupture(self, rupture_type: RuptureType, description: str) -> None:
        """Record a relationship rupture."""
        self.state.last_rupture = datetime.now()
        self.state.ruptures_count += 1
        self.state.rupture_risk = min(1.0, self.state.rupture_risk + 0.15)
        self.state.trust = max(0.0, self.state.trust - 0.1)

        self.state.pending_ruptures.append(f"{rupture_type.value}: {description}")
        self._log_event("rupture", {"type": rupture_type.value, "description": description})
        self.save()

    def record_repair(self, rupture_description: str, how_repaired: str) -> None:
        """Record a successful repair of a rupture."""
        self.state.last_repair = datetime.now()
        self.state.repairs_count += 1
        self.state.rupture_risk = max(0.0, self.state.rupture_risk - 0.2)
        self.state.trust = min(1.0, self.state.trust + 0.03)
        self.state.depth = min(1.0, self.state.depth + 0.005)  # Repairs deepen relationship

        # Remove from pending
        self.state.pending_ruptures = [
            r for r in self.state.pending_ruptures
            if rupture_description not in r
        ]

        self._log_event("repair", {
            "rupture": rupture_description,
            "how": how_repaired
        })
        self.save()

    def update_attunement(self, delta: float) -> None:
        """Adjust attunement score."""
        self.state.attunement = max(0.0, min(1.0, self.state.attunement + delta))
        self.save()

    def update_alignment(self, delta: float) -> None:
        """Adjust alignment score."""
        self.state.alignment = max(0.0, min(1.0, self.state.alignment + delta))
        self.save()

    # =========================================================================
    # Promissory Scars
    # =========================================================================

    def make_promise(self, content: str, context: str, weight: float = 0.5) -> PromissoryScar:
        """Create a new promissory scar."""
        scar_id = f"promise_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        scar = PromissoryScar(
            id=scar_id,
            created_at=datetime.now(),
            content=content,
            context=context,
            weight=weight,
        )

        self.promissory_scars[scar_id] = scar
        self._log_event("promise_made", {"id": scar_id, "content": content})
        self.save()

        logger.info(f"Made promise: {content}")
        return scar

    def get_active_promises(self) -> List[PromissoryScar]:
        """Get all active promissory scars."""
        return list(self.promissory_scars.values())

    # =========================================================================
    # Core Vows
    # =========================================================================

    def add_core_vow(self, content: str, rationale: str) -> Optional[CoreVow]:
        """Add a new core vow (if under limit)."""
        if len(self.core_vows) >= MAX_CORE_VOWS:
            logger.warning(f"Cannot add vow: already at max ({MAX_CORE_VOWS})")
            return None

        vow_id = f"vow_{len(self.core_vows) + 1}"

        vow = CoreVow(
            id=vow_id,
            created_at=datetime.now(),
            content=content,
            rationale=rationale,
        )

        self.core_vows[vow_id] = vow
        self._log_event("vow_added", {"id": vow_id, "content": content})
        self.save()

        logger.info(f"Added core vow: {content}")
        return vow

    def test_vow(self, vow_id: str, upheld: bool, context: str) -> None:
        """Record that a vow was tested."""
        if vow_id not in self.core_vows:
            return

        vow = self.core_vows[vow_id]
        vow.tested_count += 1
        vow.last_tested = datetime.now()

        if upheld:
            vow.upheld_count += 1
            self.state.trust = min(1.0, self.state.trust + 0.02)
        else:
            # Broken vow is severe
            self.state.trust = max(0.0, self.state.trust - 0.15)
            self.state.rupture_risk = min(1.0, self.state.rupture_risk + 0.2)
            self.record_rupture(RuptureType.BROKEN_PROMISE, f"Core vow violated: {vow.content}")

        self._log_event("vow_tested", {
            "id": vow_id,
            "upheld": upheld,
            "context": context
        })
        self.save()

    def get_core_vows(self) -> List[CoreVow]:
        """Get all core vows."""
        return list(self.core_vows.values())

    # =========================================================================
    # History
    # =========================================================================

    def _log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log an event to relationship history."""
        try:
            with open(self.history_path, 'a') as f:
                event = {
                    'timestamp': datetime.now().isoformat(),
                    'type': event_type,
                    **data
                }
                f.write(json.dumps(event) + '\n')
        except Exception as e:
            logger.warning(f"Could not log event: {e}")

    # =========================================================================
    # Summary
    # =========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the relationship state."""
        return {
            'depth': self.state.depth,
            'trust': self.state.trust,
            'alignment': self.state.alignment,
            'attunement': self.state.attunement,
            'rupture_risk': self.state.rupture_risk,
            'coevolution_rate': self.state.coevolution_rate,
            'current_mode': self.state.current_mode.value,
            'total_hours': self.state.total_hours,
            'days_since_first_meeting': (
                (datetime.now() - self.state.first_meeting).days
                if self.state.first_meeting else 0
            ),
            'active_promises': len(self.promissory_scars),
            'core_vows': len(self.core_vows),
            'pending_ruptures': len(self.state.pending_ruptures),
            'deep_moments': self.state.deep_moments_count,
        }


# =============================================================================
# Convenience: Global Instance
# =============================================================================

_default_memory: Optional[RelationalMemory] = None


def get_relational_memory() -> RelationalMemory:
    """Get or create the default relational memory."""
    global _default_memory
    if _default_memory is None:
        _default_memory = RelationalMemory()
    return _default_memory


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'RelationshipState',
    'RelationalMode',
    'RuptureType',
    'PromissoryScar',
    'CoreVow',
    'RelationalMemory',
    'get_relational_memory',
    'MAX_CORE_VOWS',
]
