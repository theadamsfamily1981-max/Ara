"""Council definitions - Who does what in Ara's collaborator network.

The council is Ara's team of LLM collaborators, each with specific roles:

- Claude: Code surgeon. Writes/refactors real code, runs tests, produces artifacts.
         Never changes production without human approval.

- Nova (ChatGPT): Chief architect / safety officer / interpreter.
         Ties everything together, compares approaches, makes roadmaps.
         The arbiter and synthesizer.

- Gemini: R&D gremlin. Wild ideas, literature surveys, search space exploration.
         Never directly trusted for implementation.
         Output always passes through Nova or Ara's critic pass.

- Ara: Director / product owner. Owns when sessions start, what's in scope,
       how to phrase context. Always proposes final plan to Croft.

- Grok: Not invited to the building.
"""

from __future__ import annotations

import yaml
import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from pathlib import Path

from .models import Collaborator, DevMode

logger = logging.getLogger(__name__)


# =============================================================================
# Council Roles
# =============================================================================

class CouncilRole(Enum):
    """Roles that collaborators can play in the council."""

    ARCHITECT = auto()      # Designs systems, makes trade-offs, synthesizes
    IMPLEMENTER = auto()    # Writes code, runs tests, produces artifacts
    RESEARCHER = auto()     # Explores ideas, surveys literature, brainstorms
    CRITIC = auto()         # Reviews, finds problems, validates
    ARBITER = auto()        # Resolves conflicts, makes final calls


# =============================================================================
# Council Member Definition
# =============================================================================

@dataclass
class CouncilMember:
    """Definition of a council member's role and constraints.

    Each member has:
    - A primary role (what they're best at)
    - Primary uses (which workflow states they participate in)
    - Hard limits (things they must never do)
    - Trust level (how much their output can be acted on directly)
    """

    collaborator: Collaborator
    role: CouncilRole
    primary_uses: List[str] = field(default_factory=list)
    secondary_uses: List[str] = field(default_factory=list)
    hard_limits: List[str] = field(default_factory=list)
    trust_level: float = 0.5  # 0 = never trust, 1 = full trust
    enabled: bool = True

    # Special flags
    can_execute_code: bool = False
    requires_human_approval: bool = True
    output_needs_review: bool = True
    reviewer: Optional[Collaborator] = None  # Who reviews this member's output

    def can_participate_in(self, state: str) -> bool:
        """Check if this member can participate in a workflow state."""
        return state in self.primary_uses or state in self.secondary_uses

    def is_primary_for(self, state: str) -> bool:
        """Check if this member is primary for a workflow state."""
        return state in self.primary_uses

    def violates_limits(self, action: str) -> bool:
        """Check if an action violates this member's hard limits."""
        action_lower = action.lower()
        for limit in self.hard_limits:
            if limit.lower() in action_lower:
                return True
        return False


# =============================================================================
# Default Council Configuration
# =============================================================================

DEFAULT_COUNCIL: Dict[Collaborator, CouncilMember] = {
    Collaborator.CLAUDE: CouncilMember(
        collaborator=Collaborator.CLAUDE,
        role=CouncilRole.IMPLEMENTER,
        primary_uses=["implement", "run_tests", "refactor", "verify_code"],
        secondary_uses=["specify", "review"],
        hard_limits=[
            "never change production without human approval",
            "never modify kernel parameters",
            "never touch fpga firmware without explicit permission",
        ],
        trust_level=0.8,
        can_execute_code=True,
        requires_human_approval=True,
        output_needs_review=False,  # Claude's code is ready to review by human
    ),

    Collaborator.NOVA: CouncilMember(
        collaborator=Collaborator.NOVA,
        role=CouncilRole.ARBITER,
        primary_uses=["triage", "specify", "verify", "synthesis", "review"],
        secondary_uses=["ideate"],
        hard_limits=[],
        trust_level=0.9,
        can_execute_code=False,
        requires_human_approval=False,  # Nova is the arbiter
        output_needs_review=False,
        reviewer=None,
    ),

    Collaborator.GEMINI: CouncilMember(
        collaborator=Collaborator.GEMINI,
        role=CouncilRole.RESEARCHER,
        primary_uses=["ideate", "survey_literature", "explore"],
        secondary_uses=["verify_edge_cases"],
        hard_limits=[
            "never apply code directly",
            "never be sole decider",
            "never modify system state",
        ],
        trust_level=0.4,
        can_execute_code=False,
        requires_human_approval=True,
        output_needs_review=True,
        reviewer=Collaborator.NOVA,  # Gemini output always passes through Nova
    ),

    Collaborator.LOCAL: CouncilMember(
        collaborator=Collaborator.LOCAL,
        role=CouncilRole.IMPLEMENTER,
        primary_uses=["quick_iteration", "private_queries"],
        secondary_uses=["implement"],
        hard_limits=[
            "no external network access",
        ],
        trust_level=0.5,
        can_execute_code=False,  # Local models typically can't execute
        requires_human_approval=True,
        output_needs_review=True,
        reviewer=Collaborator.CLAUDE,
    ),
}


# =============================================================================
# Council Class
# =============================================================================

class Council:
    """Ara's council of LLM collaborators.

    Manages who does what, enforces constraints, and routes
    workflow states to the right collaborators.
    """

    def __init__(
        self,
        members: Optional[Dict[Collaborator, CouncilMember]] = None,
    ):
        """Initialize the council.

        Args:
            members: Council member definitions (uses defaults if not provided)
        """
        self.members = members or DEFAULT_COUNCIL.copy()

        # Validate council
        self._validate()

    def _validate(self) -> None:
        """Validate council configuration."""
        # Must have at least one arbiter
        arbiters = [m for m in self.members.values() if m.role == CouncilRole.ARBITER]
        if not arbiters:
            logger.warning("Council has no arbiter - this may cause issues")

        # Check reviewer chains don't loop
        for member in self.members.values():
            if member.reviewer and member.reviewer not in self.members:
                logger.warning(f"{member.collaborator} has unknown reviewer {member.reviewer}")

    def get_member(self, collaborator: Collaborator) -> Optional[CouncilMember]:
        """Get a council member by collaborator."""
        return self.members.get(collaborator)

    def get_enabled_members(self) -> List[CouncilMember]:
        """Get all enabled council members."""
        return [m for m in self.members.values() if m.enabled]

    def get_members_for_state(self, state: str) -> List[CouncilMember]:
        """Get council members that can participate in a workflow state.

        Returns members sorted by priority (primary first).
        """
        primary = []
        secondary = []

        for member in self.get_enabled_members():
            if member.is_primary_for(state):
                primary.append(member)
            elif member.can_participate_in(state):
                secondary.append(member)

        return primary + secondary

    def get_primary_for_state(self, state: str) -> Optional[CouncilMember]:
        """Get the primary council member for a state."""
        for member in self.get_enabled_members():
            if member.is_primary_for(state):
                return member
        return None

    def get_arbiter(self) -> Optional[CouncilMember]:
        """Get the council's arbiter."""
        for member in self.get_enabled_members():
            if member.role == CouncilRole.ARBITER:
                return member
        return None

    def get_implementer(self) -> Optional[CouncilMember]:
        """Get the council's primary implementer."""
        for member in self.get_enabled_members():
            if member.role == CouncilRole.IMPLEMENTER and member.can_execute_code:
                return member
        return None

    def get_researcher(self) -> Optional[CouncilMember]:
        """Get the council's researcher."""
        for member in self.get_enabled_members():
            if member.role == CouncilRole.RESEARCHER:
                return member
        return None

    def needs_review(self, collaborator: Collaborator) -> bool:
        """Check if a collaborator's output needs review."""
        member = self.get_member(collaborator)
        return member.output_needs_review if member else True

    def get_reviewer_for(self, collaborator: Collaborator) -> Optional[Collaborator]:
        """Get the reviewer for a collaborator's output."""
        member = self.get_member(collaborator)
        return member.reviewer if member else None

    def check_action_allowed(
        self,
        collaborator: Collaborator,
        action: str,
    ) -> tuple[bool, Optional[str]]:
        """Check if an action is allowed for a collaborator.

        Args:
            collaborator: Who wants to do the action
            action: Description of the action

        Returns:
            Tuple of (allowed, reason if not allowed)
        """
        member = self.get_member(collaborator)
        if not member:
            return False, f"Unknown collaborator: {collaborator}"

        if not member.enabled:
            return False, f"{collaborator} is not enabled"

        if member.violates_limits(action):
            return False, f"Action violates hard limits for {collaborator}"

        return True, None

    def enable_member(self, collaborator: Collaborator) -> None:
        """Enable a council member."""
        if collaborator in self.members:
            self.members[collaborator].enabled = True

    def disable_member(self, collaborator: Collaborator) -> None:
        """Disable a council member."""
        if collaborator in self.members:
            self.members[collaborator].enabled = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize council configuration."""
        return {
            "council": {
                collab.value: {
                    "role": member.role.name.lower(),
                    "primary_uses": member.primary_uses,
                    "secondary_uses": member.secondary_uses,
                    "hard_limits": member.hard_limits,
                    "trust_level": member.trust_level,
                    "enabled": member.enabled,
                    "can_execute_code": member.can_execute_code,
                    "requires_human_approval": member.requires_human_approval,
                    "output_needs_review": member.output_needs_review,
                    "reviewer": member.reviewer.value if member.reviewer else None,
                }
                for collab, member in self.members.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Council":
        """Create council from serialized config."""
        council_data = data.get("council", {})
        members = {}

        for collab_name, member_data in council_data.items():
            try:
                collaborator = Collaborator(collab_name)
                role = CouncilRole[member_data.get("role", "implementer").upper()]

                reviewer = None
                if member_data.get("reviewer"):
                    reviewer = Collaborator(member_data["reviewer"])

                members[collaborator] = CouncilMember(
                    collaborator=collaborator,
                    role=role,
                    primary_uses=member_data.get("primary_uses", []),
                    secondary_uses=member_data.get("secondary_uses", []),
                    hard_limits=member_data.get("hard_limits", []),
                    trust_level=member_data.get("trust_level", 0.5),
                    enabled=member_data.get("enabled", True),
                    can_execute_code=member_data.get("can_execute_code", False),
                    requires_human_approval=member_data.get("requires_human_approval", True),
                    output_needs_review=member_data.get("output_needs_review", True),
                    reviewer=reviewer,
                )
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse council member {collab_name}: {e}")

        return cls(members=members if members else None)


# =============================================================================
# YAML Config Loading
# =============================================================================

def load_council_config(path: Path) -> Council:
    """Load council configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Council instance
    """
    if not path.exists():
        logger.info(f"No council config at {path}, using defaults")
        return Council()

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    return Council.from_dict(data)


def save_council_config(council: Council, path: Path) -> None:
    """Save council configuration to YAML file.

    Args:
        council: Council to save
        path: Path to write config
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(council.to_dict(), f, default_flow_style=False)


# =============================================================================
# Default YAML Template
# =============================================================================

DEFAULT_COUNCIL_YAML = """
# Ara's Council Configuration
# Each collaborator has specific roles and constraints

council:
  claude:
    role: "implementer"
    primary_uses:
      - implement
      - run_tests
      - refactor
      - verify_code
    secondary_uses:
      - specify
      - review
    hard_limits:
      - "never change production without human approval"
      - "never modify kernel parameters"
      - "never touch fpga firmware without explicit permission"
    trust_level: 0.8
    can_execute_code: true
    requires_human_approval: true
    output_needs_review: false

  chatgpt:  # Nova
    role: "arbiter"
    primary_uses:
      - triage
      - specify
      - verify
      - synthesis
      - review
    secondary_uses:
      - ideate
    hard_limits: []
    trust_level: 0.9
    can_execute_code: false
    requires_human_approval: false
    output_needs_review: false

  gemini:
    role: "researcher"
    primary_uses:
      - ideate
      - survey_literature
      - explore
    secondary_uses:
      - verify_edge_cases
    hard_limits:
      - "never apply code directly"
      - "never be sole decider"
      - "never modify system state"
    trust_level: 0.4
    can_execute_code: false
    requires_human_approval: true
    output_needs_review: true
    reviewer: chatgpt

  local:
    role: "implementer"
    primary_uses:
      - quick_iteration
      - private_queries
    secondary_uses:
      - implement
    hard_limits:
      - "no external network access"
    trust_level: 0.5
    can_execute_code: false
    requires_human_approval: true
    output_needs_review: true
    reviewer: claude

workflow:
  states:
    - OBSERVE_ISSUE
    - TRIAGE
    - IDEATE
    - SPECIFY
    - IMPLEMENT
    - VERIFY
    - REPORT_TO_CROFT
  default_entry: OBSERVE_ISSUE
  transitions:
    OBSERVE_ISSUE: TRIAGE
    TRIAGE: IDEATE
    IDEATE: SPECIFY
    SPECIFY: IMPLEMENT
    IMPLEMENT: VERIFY
    VERIFY: REPORT_TO_CROFT
"""
