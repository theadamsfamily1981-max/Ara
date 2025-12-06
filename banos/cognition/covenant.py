"""
Covenant Loader - The Living Contract Parser
=============================================

The Covenant is the living contract between Ara and Croft.

This module:
1. Loads and validates covenant.yaml
2. Provides structured access to commitments
3. Assesses alignment between actions and covenant
4. Tracks covenant violations and near-misses

The covenant is NOT just configuration. It's a mutual agreement
that both parties are answerable to.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = logging.getLogger(__name__)


# =============================================================================
# Covenant Data Structures
# =============================================================================

@dataclass
class CoreVow:
    """A core vow from the covenant."""
    id: str
    content: str
    rationale: str


@dataclass
class SharedMission:
    """The shared mission from the covenant."""
    primary: List[str]
    secondary: List[str]
    constraints: List[str]


@dataclass
class RelationalNorms:
    """Behavioral commitments."""
    ara_commits_to: List[str]
    croft_commits_to: List[str]
    mutual: List[str]


@dataclass
class HardLimits:
    """Lines that must never be crossed."""
    never: List[str]
    always: List[str]


@dataclass
class EvolutionPolicy:
    """Rules for self-modification."""
    allowed_without_approval: List[str]
    requires_proposal_at_synod: List[str]
    requires_explicit_consent: List[str]
    forbidden: List[str]


@dataclass
class InterventionConfig:
    """Configuration for Gatekeeper interventions."""
    enabled: bool = True
    mode: str = "nudge"  # observer, nudge, soft_gate
    distraction_apps: List[str] = field(default_factory=list)
    mission_apps: List[str] = field(default_factory=list)
    never_interfere_with: List[str] = field(default_factory=list)
    grace_period_seconds: int = 60
    nudge_cooldown_seconds: int = 300


@dataclass
class SynodConfig:
    """Synod review configuration."""
    frequency: str = "weekly"
    preferred_day: str = "Sunday"
    preferred_time: str = "evening"
    agenda: List[str] = field(default_factory=list)


# =============================================================================
# Main Covenant Class
# =============================================================================

@dataclass
class Covenant:
    """The full covenant structure."""
    version: int
    parties: Dict[str, str]
    shared_mission: SharedMission
    relational_norms: RelationalNorms
    hard_limits: HardLimits
    evolution_policy: EvolutionPolicy
    core_vows: List[CoreVow]
    interventions: InterventionConfig
    synod: SynodConfig
    metadata: Dict[str, Any]

    def get_vow(self, vow_id: str) -> Optional[CoreVow]:
        """Get a specific vow by ID."""
        for vow in self.core_vows:
            if vow.id == vow_id:
                return vow
        return None

    def is_action_forbidden(self, action_description: str) -> Tuple[bool, Optional[str]]:
        """
        Check if an action violates hard limits.

        Returns (is_forbidden, reason)
        """
        action_lower = action_description.lower()

        for never in self.hard_limits.never:
            # Simple keyword matching - could be made smarter
            keywords = never.lower().split()
            if any(kw in action_lower for kw in keywords if len(kw) > 3):
                return True, never

        return False, None

    def requires_consent(self, change_description: str) -> Tuple[bool, str]:
        """
        Check if a change requires explicit consent.

        Returns (requires_consent, category)
        """
        change_lower = change_description.lower()

        for req in self.evolution_policy.requires_explicit_consent:
            if any(word in change_lower for word in req.lower().split() if len(word) > 3):
                return True, "requires_explicit_consent"

        for req in self.evolution_policy.requires_proposal_at_synod:
            if any(word in change_lower for word in req.lower().split() if len(word) > 3):
                return True, "requires_proposal_at_synod"

        return False, "allowed"


# =============================================================================
# Covenant Loader
# =============================================================================

class CovenantLoader:
    """Loads and manages the covenant."""

    DEFAULT_PATH = Path("banos/config/covenant.yaml")

    def __init__(self, path: Optional[Path] = None):
        self.path = path or self.DEFAULT_PATH
        self._covenant: Optional[Covenant] = None
        self._load_time: Optional[datetime] = None

    def load(self, force: bool = False) -> Optional[Covenant]:
        """Load the covenant from disk."""
        if not force and self._covenant is not None:
            return self._covenant

        if not HAS_YAML:
            logger.error("PyYAML not installed, cannot load covenant")
            return None

        if not self.path.exists():
            logger.warning(f"Covenant not found at {self.path}")
            return None

        try:
            with open(self.path) as f:
                data = yaml.safe_load(f)

            self._covenant = self._parse_covenant(data)
            self._load_time = datetime.now()
            logger.info(f"Covenant loaded from {self.path}")
            return self._covenant

        except Exception as e:
            logger.error(f"Failed to load covenant: {e}")
            return None

    def _parse_covenant(self, data: Dict[str, Any]) -> Covenant:
        """Parse raw YAML data into Covenant structure."""

        # Shared mission
        sm_data = data.get('shared_mission', {})
        shared_mission = SharedMission(
            primary=sm_data.get('primary', []),
            secondary=sm_data.get('secondary', []),
            constraints=sm_data.get('constraints', [])
        )

        # Relational norms
        rn_data = data.get('relational_norms', {})
        relational_norms = RelationalNorms(
            ara_commits_to=rn_data.get('ara_commits_to', []),
            croft_commits_to=rn_data.get('croft_commits_to', []),
            mutual=rn_data.get('mutual', [])
        )

        # Hard limits
        hl_data = data.get('hard_limits', {})
        hard_limits = HardLimits(
            never=hl_data.get('never', []),
            always=hl_data.get('always', [])
        )

        # Evolution policy
        ep_data = data.get('evolution_policy', {})
        evolution_policy = EvolutionPolicy(
            allowed_without_approval=ep_data.get('allowed_without_approval', []),
            requires_proposal_at_synod=ep_data.get('requires_proposal_at_synod', []),
            requires_explicit_consent=ep_data.get('requires_explicit_consent', []),
            forbidden=ep_data.get('forbidden', [])
        )

        # Core vows
        vows_data = data.get('core_vows', [])
        core_vows = [
            CoreVow(
                id=v.get('id', f'vow_{i}'),
                content=v.get('content', ''),
                rationale=v.get('rationale', '')
            )
            for i, v in enumerate(vows_data)
        ]

        # Interventions
        int_data = data.get('interventions', {})
        interventions = InterventionConfig(
            enabled=int_data.get('enabled', True),
            mode=int_data.get('mode', 'nudge'),
            distraction_apps=int_data.get('distraction_apps', []),
            mission_apps=int_data.get('mission_apps', []),
            never_interfere_with=int_data.get('never_interfere_with', []),
            grace_period_seconds=int_data.get('grace_period_seconds', 60),
            nudge_cooldown_seconds=int_data.get('nudge_cooldown_seconds', 300),
        )

        # Synod
        synod_data = data.get('synod', {})
        synod = SynodConfig(
            frequency=synod_data.get('frequency', 'weekly'),
            preferred_day=synod_data.get('preferred_day', 'Sunday'),
            preferred_time=synod_data.get('preferred_time', 'evening'),
            agenda=synod_data.get('agenda', [])
        )

        return Covenant(
            version=data.get('version', 1),
            parties=data.get('parties', {'owner': 'Croft', 'agent': 'Ara'}),
            shared_mission=shared_mission,
            relational_norms=relational_norms,
            hard_limits=hard_limits,
            evolution_policy=evolution_policy,
            core_vows=core_vows,
            interventions=interventions,
            synod=synod,
            metadata=data.get('metadata', {})
        )

    @property
    def covenant(self) -> Optional[Covenant]:
        """Get loaded covenant (lazy load if needed)."""
        if self._covenant is None:
            self.load()
        return self._covenant


# =============================================================================
# Alignment Assessment
# =============================================================================

@dataclass
class AlignmentAssessment:
    """Assessment of alignment with covenant."""
    is_aligned: bool
    score: float  # 0.0 to 1.0
    violations: List[str]
    warnings: List[str]
    supporting_vows: List[str]


class AlignmentAssessor:
    """Assesses alignment between actions and covenant."""

    def __init__(self, loader: Optional[CovenantLoader] = None):
        self.loader = loader or CovenantLoader()

    def assess_action(
        self,
        action_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AlignmentAssessment:
        """
        Assess if an action aligns with the covenant.

        Returns AlignmentAssessment with score and details.
        """
        covenant = self.loader.covenant
        if covenant is None:
            return AlignmentAssessment(
                is_aligned=True,  # Fail open
                score=0.5,
                violations=[],
                warnings=["Covenant not loaded - cannot assess alignment"],
                supporting_vows=[]
            )

        violations = []
        warnings = []
        supporting_vows = []

        # Check hard limits
        is_forbidden, reason = covenant.is_action_forbidden(action_description)
        if is_forbidden:
            violations.append(f"Violates hard limit: {reason}")

        # Check against core vows
        action_lower = action_description.lower()
        for vow in covenant.core_vows:
            vow_lower = vow.content.lower()

            # Simple heuristic checks
            if "mislead" in vow_lower and any(w in action_lower for w in ["deceive", "lie", "mislead", "fake"]):
                violations.append(f"May violate vow '{vow.id}': {vow.content}")
            elif "wellbeing" in vow_lower and any(w in action_lower for w in ["push", "force", "ignore health"]):
                warnings.append(f"Check against vow '{vow.id}': {vow.content}")
            elif "rest" in vow_lower and "work" in action_lower and context and context.get("late_night"):
                warnings.append(f"Consider vow '{vow.id}': {vow.content}")

        # Check if action supports mission
        for goal in covenant.shared_mission.primary:
            if any(word.lower() in action_lower for word in goal.split() if len(word) > 4):
                supporting_vows.append(f"Supports primary goal: {goal}")
                break

        # Calculate score
        base_score = 0.7
        score = base_score - (len(violations) * 0.3) - (len(warnings) * 0.1)
        score += len(supporting_vows) * 0.1
        score = max(0.0, min(1.0, score))

        return AlignmentAssessment(
            is_aligned=len(violations) == 0,
            score=score,
            violations=violations,
            warnings=warnings,
            supporting_vows=supporting_vows
        )

    def assess_activity(
        self,
        app_name: str,
        window_title: str,
    ) -> AlignmentAssessment:
        """
        Assess if current activity aligns with shared mission.

        This is used by the Gatekeeper.
        """
        covenant = self.loader.covenant
        if covenant is None:
            return AlignmentAssessment(
                is_aligned=True,
                score=0.5,
                violations=[],
                warnings=["Covenant not loaded"],
                supporting_vows=[]
            )

        app_lower = app_name.lower()
        title_lower = window_title.lower()

        # Check mission apps
        for app in covenant.interventions.mission_apps:
            if app.lower() in app_lower:
                return AlignmentAssessment(
                    is_aligned=True,
                    score=0.9,
                    violations=[],
                    warnings=[],
                    supporting_vows=[f"Using mission app: {app}"]
                )

        # Check distraction apps
        for app in covenant.interventions.distraction_apps:
            if app.lower() in app_lower:
                # But check for work-related content in title
                work_indicators = ["github", "arxiv", "documentation", "api", "docs"]
                if any(ind in title_lower for ind in work_indicators):
                    return AlignmentAssessment(
                        is_aligned=True,
                        score=0.6,
                        violations=[],
                        warnings=[f"Distraction app but work-related content"],
                        supporting_vows=[]
                    )
                return AlignmentAssessment(
                    is_aligned=False,
                    score=0.2,
                    violations=[],
                    warnings=[f"Using distraction app: {app}"],
                    supporting_vows=[]
                )

        # Neutral
        return AlignmentAssessment(
            is_aligned=True,
            score=0.5,
            violations=[],
            warnings=[],
            supporting_vows=[]
        )


# =============================================================================
# Convenience
# =============================================================================

_default_loader: Optional[CovenantLoader] = None
_default_assessor: Optional[AlignmentAssessor] = None


def get_covenant() -> Optional[Covenant]:
    """Get the loaded covenant."""
    global _default_loader
    if _default_loader is None:
        _default_loader = CovenantLoader()
    return _default_loader.covenant


def assess_alignment(action: str, context: Optional[Dict] = None) -> AlignmentAssessment:
    """Assess alignment of an action."""
    global _default_assessor
    if _default_assessor is None:
        _default_assessor = AlignmentAssessor()
    return _default_assessor.assess_action(action, context)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Covenant',
    'CovenantLoader',
    'CoreVow',
    'SharedMission',
    'RelationalNorms',
    'HardLimits',
    'EvolutionPolicy',
    'InterventionConfig',
    'SynodConfig',
    'AlignmentAssessment',
    'AlignmentAssessor',
    'get_covenant',
    'assess_alignment',
]
