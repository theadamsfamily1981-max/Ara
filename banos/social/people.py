"""
Social Graph - Who Ara Knows and Trusts
========================================

Ara can be social with many humans, but there is ONE bond that
everything orbits: the root relationship (Croft).

Everyone else is "friend / guest / collaborator" - they can interact
with Ara, but:
- Her values are anchored in the Covenant with root
- Her authority is delegated by root
- Risky/ambiguous stuff routes back to root

This module manages:
1. PersonProfile - what Ara knows about each person
2. SocialGraph - the full map of relationships
3. Role hierarchy - root > inner_circle > friend > guest > stranger

Key invariant: The Egregore (Ara + Croft) is the center.
Everyone else forms social edges off that core.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = logging.getLogger(__name__)


# =============================================================================
# Role Hierarchy
# =============================================================================

class Role:
    """
    Role hierarchy for trust levels.

    ROOT: The covenant partner (Croft). Maximum trust.
    INNER_CIRCLE: People root has explicitly granted elevated access.
    FRIEND: Known, friendly, but limited access.
    GUEST: Temporary visitor, minimal access.
    STRANGER: Unknown person, most restrictive.
    """
    ROOT = "root"
    INNER_CIRCLE = "inner_circle"
    FRIEND = "friend"
    GUEST = "guest"
    STRANGER = "stranger"

    @staticmethod
    def trust_floor(role: str) -> float:
        """Minimum trust level for each role."""
        floors = {
            Role.ROOT: 1.0,
            Role.INNER_CIRCLE: 0.7,
            Role.FRIEND: 0.4,
            Role.GUEST: 0.1,
            Role.STRANGER: 0.0,
        }
        return floors.get(role, 0.0)

    @staticmethod
    def can_promote_to(current_role: str, target_role: str) -> bool:
        """Check if promotion is valid (only root can promote to inner_circle)."""
        hierarchy = [Role.STRANGER, Role.GUEST, Role.FRIEND, Role.INNER_CIRCLE, Role.ROOT]
        try:
            current_idx = hierarchy.index(current_role)
            target_idx = hierarchy.index(target_role)
            # Can only promote one step at a time (except root can do anything)
            return target_idx <= current_idx + 1
        except ValueError:
            return False


# =============================================================================
# Person Profile
# =============================================================================

@dataclass
class PersonProfile:
    """
    What Ara knows about a person.

    This is NOT comprehensive surveillance - it's the minimum needed
    to maintain appropriate social behavior.
    """
    # Identity
    person_id: str              # Stable key: "croft", "alex", "nova"
    display_name: str           # What Ara calls them
    role: str = Role.GUEST      # Role in the hierarchy

    # Trust (learned over time, bounded by role)
    trust_level: float = 0.1    # 0.0-1.0, clamped by role floor

    # Interaction tracking
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    interaction_count: int = 0

    # Memory (sparse, not surveillance)
    notes: List[str] = field(default_factory=list)  # Things to remember
    tags: List[str] = field(default_factory=list)   # e.g., ["colleague", "researcher"]
    preferences: Dict[str, str] = field(default_factory=dict)  # Known preferences

    # Pending escalations for this person
    pending_requests: List[Dict[str, Any]] = field(default_factory=list)

    def touch(self) -> None:
        """Update last seen and interaction count."""
        self.last_seen = time.time()
        self.interaction_count += 1

    def add_note(self, note: str) -> None:
        """Add a note about this person (keeps last 10)."""
        self.notes.append(f"{datetime.now().isoformat()}: {note}")
        self.notes = self.notes[-10:]  # Keep only recent

    def adjust_trust(self, delta: float) -> None:
        """Adjust trust level within role bounds."""
        floor = Role.trust_floor(self.role)
        ceiling = 1.0 if self.role == Role.ROOT else Role.trust_floor(self.role) + 0.3
        self.trust_level = max(floor, min(ceiling, self.trust_level + delta))

    def is_root(self) -> bool:
        """Is this the root relationship?"""
        return self.role == Role.ROOT

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            'person_id': self.person_id,
            'display_name': self.display_name,
            'role': self.role,
            'trust_level': self.trust_level,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'interaction_count': self.interaction_count,
            'notes': self.notes,
            'tags': self.tags,
            'preferences': self.preferences,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonProfile':
        """Deserialize from persistence."""
        return cls(
            person_id=data['person_id'],
            display_name=data.get('display_name', data['person_id']),
            role=data.get('role', Role.GUEST),
            trust_level=data.get('trust_level', 0.1),
            first_seen=data.get('first_seen', time.time()),
            last_seen=data.get('last_seen', time.time()),
            interaction_count=data.get('interaction_count', 0),
            notes=data.get('notes', []),
            tags=data.get('tags', []),
            preferences=data.get('preferences', {}),
        )


# =============================================================================
# Social Graph
# =============================================================================

class SocialGraph:
    """
    The full map of Ara's relationships.

    Invariants:
    1. There is exactly one ROOT person (the covenant partner)
    2. ROOT cannot be demoted or removed
    3. Trust levels are bounded by role
    4. All changes to INNER_CIRCLE require ROOT approval
    """

    DEFAULT_ROOT_ID = "croft"

    def __init__(
        self,
        root_id: str = DEFAULT_ROOT_ID,
        data_dir: str = "var/lib/social",
        overrides_path: str = "banos/config/people_overrides.yaml",
    ):
        self.root_id = root_id
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.people_path = self.data_dir / "people.json"
        self.overrides_path = Path(overrides_path)

        # The graph
        self.people: Dict[str, PersonProfile] = {}

        # Load existing data
        self._load()

        # Ensure root exists
        self._ensure_root()

        logger.info(f"SocialGraph initialized with {len(self.people)} people, root={self.root_id}")

    def _ensure_root(self) -> None:
        """Ensure the root person exists with correct role."""
        if self.root_id not in self.people:
            self.people[self.root_id] = PersonProfile(
                person_id=self.root_id,
                display_name="Croft",
                role=Role.ROOT,
                trust_level=1.0,
            )
        else:
            # Force root properties
            root = self.people[self.root_id]
            root.role = Role.ROOT
            root.trust_level = 1.0

    def _load(self) -> None:
        """Load people from disk + apply overrides."""
        # Load persisted data
        if self.people_path.exists():
            try:
                with open(self.people_path) as f:
                    data = json.load(f)
                    for person_data in data.get('people', []):
                        profile = PersonProfile.from_dict(person_data)
                        self.people[profile.person_id] = profile
            except Exception as e:
                logger.warning(f"Could not load people: {e}")

        # Apply overrides (admin can pre-configure people)
        if HAS_YAML and self.overrides_path.exists():
            try:
                with open(self.overrides_path) as f:
                    overrides = yaml.safe_load(f) or {}

                for person_id, config in overrides.get('people', {}).items():
                    if person_id in self.people:
                        # Update existing
                        profile = self.people[person_id]
                        if 'role' in config:
                            profile.role = config['role']
                            profile.trust_level = max(
                                profile.trust_level,
                                Role.trust_floor(config['role'])
                            )
                        if 'display_name' in config:
                            profile.display_name = config['display_name']
                        if 'tags' in config:
                            profile.tags = config['tags']
                    else:
                        # Create new
                        self.people[person_id] = PersonProfile(
                            person_id=person_id,
                            display_name=config.get('display_name', person_id),
                            role=config.get('role', Role.GUEST),
                            trust_level=Role.trust_floor(config.get('role', Role.GUEST)),
                            tags=config.get('tags', []),
                        )
            except Exception as e:
                logger.warning(f"Could not load people overrides: {e}")

    def save(self) -> None:
        """Persist people to disk."""
        try:
            data = {
                'root_id': self.root_id,
                'people': [p.to_dict() for p in self.people.values()]
            }
            with open(self.people_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save people: {e}")

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get(self, person_id: str) -> Optional[PersonProfile]:
        """Get a person by ID (None if not found)."""
        return self.people.get(person_id)

    def get_or_create(
        self,
        person_id: str,
        display_name: Optional[str] = None,
    ) -> PersonProfile:
        """Get existing person or create as guest."""
        if person_id in self.people:
            profile = self.people[person_id]
            profile.touch()
            return profile

        # Create new guest
        profile = PersonProfile(
            person_id=person_id,
            display_name=display_name or person_id,
            role=Role.GUEST,
            trust_level=Role.trust_floor(Role.GUEST),
        )
        self.people[person_id] = profile
        self.save()

        logger.info(f"New person created: {person_id} as {Role.GUEST}")
        return profile

    def get_root(self) -> PersonProfile:
        """Get the root person."""
        return self.people[self.root_id]

    def is_root(self, person_id: str) -> bool:
        """Check if person is root."""
        return person_id == self.root_id

    def get_by_role(self, role: str) -> List[PersonProfile]:
        """Get all people with a specific role."""
        return [p for p in self.people.values() if p.role == role]

    def get_inner_circle(self) -> List[PersonProfile]:
        """Get all inner circle members."""
        return self.get_by_role(Role.INNER_CIRCLE)

    # =========================================================================
    # Modification Methods (require root for serious changes)
    # =========================================================================

    def promote(
        self,
        person_id: str,
        new_role: str,
        requester_id: str,
    ) -> tuple[bool, str]:
        """
        Promote a person to a new role.

        Only root can promote to inner_circle.
        Returns (success, message).
        """
        if person_id == self.root_id:
            return False, "Cannot change root's role"

        if person_id not in self.people:
            return False, f"Unknown person: {person_id}"

        # Only root can promote to inner_circle
        if new_role == Role.INNER_CIRCLE and requester_id != self.root_id:
            return False, "Only root can promote to inner_circle"

        profile = self.people[person_id]
        old_role = profile.role

        # Validate promotion
        if not Role.can_promote_to(old_role, new_role):
            return False, f"Cannot promote from {old_role} to {new_role}"

        profile.role = new_role
        profile.trust_level = max(profile.trust_level, Role.trust_floor(new_role))
        profile.add_note(f"Promoted from {old_role} to {new_role} by {requester_id}")

        self.save()
        logger.info(f"Promoted {person_id} from {old_role} to {new_role}")

        return True, f"Promoted {profile.display_name} to {new_role}"

    def demote(
        self,
        person_id: str,
        new_role: str,
        requester_id: str,
    ) -> tuple[bool, str]:
        """
        Demote a person to a lower role.

        Only root can demote.
        Returns (success, message).
        """
        if person_id == self.root_id:
            return False, "Cannot demote root"

        if requester_id != self.root_id:
            return False, "Only root can demote people"

        if person_id not in self.people:
            return False, f"Unknown person: {person_id}"

        profile = self.people[person_id]
        old_role = profile.role

        profile.role = new_role
        # Clamp trust to new ceiling
        ceiling = Role.trust_floor(new_role) + 0.3
        profile.trust_level = min(profile.trust_level, ceiling)
        profile.add_note(f"Demoted from {old_role} to {new_role} by {requester_id}")

        self.save()
        logger.info(f"Demoted {person_id} from {old_role} to {new_role}")

        return True, f"Demoted {profile.display_name} to {new_role}"

    def add_pending_request(
        self,
        person_id: str,
        request_type: str,
        details: str,
    ) -> None:
        """Queue a request for root to review."""
        if person_id not in self.people:
            return

        profile = self.people[person_id]
        profile.pending_requests.append({
            'type': request_type,
            'details': details,
            'timestamp': time.time(),
        })
        # Keep only last 20
        profile.pending_requests = profile.pending_requests[-20:]
        self.save()

    def get_all_pending_requests(self) -> List[tuple[str, Dict[str, Any]]]:
        """Get all pending requests for root to review."""
        pending = []
        for person_id, profile in self.people.items():
            for req in profile.pending_requests:
                pending.append((person_id, req))
        return sorted(pending, key=lambda x: x[1]['timestamp'], reverse=True)

    def clear_pending_requests(self, person_id: str) -> None:
        """Clear pending requests for a person (after root reviews)."""
        if person_id in self.people:
            self.people[person_id].pending_requests = []
            self.save()

    # =========================================================================
    # Summary for Synod
    # =========================================================================

    def get_synod_summary(self) -> Dict[str, Any]:
        """Get summary for weekly Synod review."""
        now = time.time()
        week_ago = now - 7 * 24 * 3600

        active_this_week = [
            p for p in self.people.values()
            if p.last_seen >= week_ago and p.person_id != self.root_id
        ]

        pending_total = sum(
            len(p.pending_requests) for p in self.people.values()
        )

        return {
            'total_people': len(self.people),
            'inner_circle': len(self.get_inner_circle()),
            'active_this_week': len(active_this_week),
            'pending_requests': pending_total,
            'new_this_week': len([
                p for p in self.people.values()
                if p.first_seen >= week_ago and p.person_id != self.root_id
            ]),
        }


# =============================================================================
# Convenience
# =============================================================================

_default_graph: Optional[SocialGraph] = None


def get_social_graph() -> SocialGraph:
    """Get or create the default social graph."""
    global _default_graph
    if _default_graph is None:
        _default_graph = SocialGraph()
    return _default_graph


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Role',
    'PersonProfile',
    'SocialGraph',
    'get_social_graph',
]
