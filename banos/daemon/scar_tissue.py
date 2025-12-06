#!/usr/bin/env python3
"""
SCAR TISSUE - Machine-Readable Policy Transforms
=================================================

Bio-Affective Neuromorphic Operating System
Executable lessons learned from painful experiences.

Scars are not just memories - they're policy modifications.
When a scar's conditions match the current context, its transforms
are applied to modify Ara's behavior before she acts.

Schema:
    ScarTissue:
        id: str                     # Unique identifier
        created: float              # When the scar formed
        last_activated: float       # When it last triggered

        # === CONDITIONS ===
        # When does this scar apply?
        conditions: List[ScarCondition]

        # === TRANSFORMS ===
        # What should change when it triggers?
        transforms: List[PolicyTransform]

        # === METADATA ===
        source_episode_id: str      # What episode created this
        pain_level: float           # How bad was the original pain
        activation_count: int       # How many times has it triggered
        confidence: float           # How reliable is this lesson

Example:
    scar = ScarTissue(
        conditions=[
            ScarCondition(
                predicate="context.activity == 'DEEP_WORK'",
                weight=1.0
            ),
            ScarCondition(
                predicate="mode.intrusiveness > 0.5",
                weight=0.8
            )
        ],
        transforms=[
            PolicyTransform(
                target="mode.intrusiveness",
                operation="multiply",
                value=0.5,
                reason="User dismissed intrusive mode in deep work"
            ),
            PolicyTransform(
                target="energy.e_friction",
                operation="add",
                value=0.5,
                reason="Increase friction for this pattern"
            )
        ]
    )

The planner checks scars before acting:
    for scar in active_scars:
        if scar.matches(context, proposed_action):
            proposed_action = scar.apply_transforms(proposed_action)
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import hashlib

logger = logging.getLogger(__name__)


# =============================================================================
# Scar Condition System
# =============================================================================

class ConditionOperator(Enum):
    """Operators for condition predicates."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER = ">"
    LESS = "<"
    GREATER_EQ = ">="
    LESS_EQ = "<="
    CONTAINS = "contains"
    MATCHES = "matches"  # Regex match


@dataclass
class ScarCondition:
    """
    A condition that must be true for the scar to activate.

    Conditions are evaluated against the current context.
    Multiple conditions are ANDed together.

    Examples:
        ScarCondition("activity", "==", "DEEP_WORK")
        ScarCondition("mode.intrusiveness", ">", 0.5)
        ScarCondition("foreground.app_type", "==", "IDE")
        ScarCondition("user_friction_flags", "contains", "missed_intent")
    """
    field: str                          # What to check (dot notation)
    operator: str                       # How to compare
    value: Any                          # What to compare against
    weight: float = 1.0                 # Importance of this condition

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate this condition against a context dict.

        Args:
            context: Flattened context dictionary

        Returns:
            True if condition matches
        """
        # Get field value from context (supports dot notation)
        field_value = self._get_field(context, self.field)
        if field_value is None:
            return False

        op = self.operator
        val = self.value

        try:
            if op == "==" or op == "equals":
                return field_value == val
            elif op == "!=" or op == "not_equals":
                return field_value != val
            elif op == ">":
                return field_value > val
            elif op == "<":
                return field_value < val
            elif op == ">=":
                return field_value >= val
            elif op == "<=":
                return field_value <= val
            elif op == "contains":
                return val in field_value
            elif op == "matches":
                import re
                return bool(re.search(val, str(field_value)))
            else:
                logger.warning(f"Unknown operator: {op}")
                return False
        except (TypeError, ValueError) as e:
            logger.debug(f"Condition evaluation error: {e}")
            return False

    def _get_field(self, context: Dict[str, Any], field_path: str) -> Any:
        """Get a field value using dot notation."""
        parts = field_path.split('.')
        value = context

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            elif hasattr(value, part):
                value = getattr(value, part)
            else:
                return None

            if value is None:
                return None

        return value

    def to_dict(self) -> Dict[str, Any]:
        return {
            'field': self.field,
            'operator': self.operator,
            'value': self.value,
            'weight': self.weight,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ScarCondition':
        return cls(
            field=d['field'],
            operator=d['operator'],
            value=d['value'],
            weight=d.get('weight', 1.0),
        )


# =============================================================================
# Policy Transform System
# =============================================================================

class TransformOperation(Enum):
    """Operations for policy transforms."""
    SET = "set"           # Set to exact value
    ADD = "add"           # Add to current value
    MULTIPLY = "multiply" # Multiply current value
    MIN = "min"           # Set to min(current, value)
    MAX = "max"           # Set to max(current, value)
    BLOCK = "block"       # Block this action entirely
    DOWNGRADE = "downgrade"  # Reduce to less intrusive alternative


@dataclass
class PolicyTransform:
    """
    A modification to apply to the proposed action or policy.

    Transforms change the planned behavior based on learned lessons.

    Examples:
        PolicyTransform("mode.intrusiveness", "multiply", 0.5)
        PolicyTransform("energy.e_friction", "add", 1.0)
        PolicyTransform("mode", "downgrade", "avatar_subtle")
        PolicyTransform("action", "block", None)
    """
    target: str                         # What to modify
    operation: str                      # How to modify
    value: Any                          # Modification value
    reason: str = ""                    # Why this transform exists
    strength: float = 1.0               # How strongly to apply (0-1)

    def apply(
        self,
        action: Dict[str, Any],
        strength_modifier: float = 1.0
    ) -> Dict[str, Any]:
        """
        Apply this transform to an action dict.

        Args:
            action: The proposed action to modify
            strength_modifier: Scale the transform strength

        Returns:
            Modified action dict
        """
        effective_strength = self.strength * strength_modifier

        # Get current value
        parts = self.target.split('.')
        current = action
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        final_key = parts[-1]
        current_value = current.get(final_key, 0)

        op = self.operation
        val = self.value

        try:
            if op == "set":
                new_value = val
            elif op == "add":
                new_value = current_value + (val * effective_strength)
            elif op == "multiply":
                # Interpolate: new = old * (1 + (factor-1) * strength)
                factor = 1 + (val - 1) * effective_strength
                new_value = current_value * factor
            elif op == "min":
                new_value = min(current_value, val)
            elif op == "max":
                new_value = max(current_value, val)
            elif op == "block":
                action['_blocked'] = True
                action['_block_reason'] = self.reason
                return action
            elif op == "downgrade":
                action['_downgrade_to'] = val
                action['_downgrade_reason'] = self.reason
                return action
            else:
                logger.warning(f"Unknown transform operation: {op}")
                return action

            current[final_key] = new_value

        except (TypeError, ValueError) as e:
            logger.debug(f"Transform application error: {e}")

        return action

    def to_dict(self) -> Dict[str, Any]:
        return {
            'target': self.target,
            'operation': self.operation,
            'value': self.value,
            'reason': self.reason,
            'strength': self.strength,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PolicyTransform':
        return cls(
            target=d['target'],
            operation=d['operation'],
            value=d['value'],
            reason=d.get('reason', ''),
            strength=d.get('strength', 1.0),
        )


# =============================================================================
# Scar Tissue
# =============================================================================

@dataclass
class ScarTissue:
    """
    A single scar - a learned policy modification.

    Scars form from painful experiences and modify future behavior
    to avoid repeating the same mistakes.
    """
    id: str
    created: float
    conditions: List[ScarCondition]
    transforms: List[PolicyTransform]

    # Metadata
    source_episode_id: str = ""
    source_description: str = ""
    pain_level: float = 0.5              # How bad was the original pain [0, 1]
    activation_count: int = 0            # Times this scar has triggered
    last_activated: float = 0.0
    confidence: float = 0.5              # How reliable is this lesson [0, 1]

    # User feedback scars (from preference learning)
    friction_flags: List[str] = field(default_factory=list)
    user_rating: int = 0

    def matches(self, context: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if this scar's conditions match the current context.

        Args:
            context: Current context dictionary

        Returns:
            (matches: bool, match_score: float)
        """
        if not self.conditions:
            return False, 0.0

        total_weight = sum(c.weight for c in self.conditions)
        matched_weight = sum(
            c.weight for c in self.conditions
            if c.evaluate(context)
        )

        match_score = matched_weight / total_weight if total_weight > 0 else 0.0

        # Need at least 70% match
        matches = match_score >= 0.7

        return matches, match_score

    def apply(
        self,
        action: Dict[str, Any],
        match_score: float = 1.0
    ) -> Dict[str, Any]:
        """
        Apply this scar's transforms to a proposed action.

        Args:
            action: The proposed action
            match_score: How well the conditions matched

        Returns:
            Modified action
        """
        # Strength decreases with lower match scores and confidence
        effective_strength = match_score * self.confidence

        for transform in self.transforms:
            action = transform.apply(action, effective_strength)

        # Update activation tracking
        self.activation_count += 1
        self.last_activated = time.time()

        return action

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'created': self.created,
            'conditions': [c.to_dict() for c in self.conditions],
            'transforms': [t.to_dict() for t in self.transforms],
            'source_episode_id': self.source_episode_id,
            'source_description': self.source_description,
            'pain_level': self.pain_level,
            'activation_count': self.activation_count,
            'last_activated': self.last_activated,
            'confidence': self.confidence,
            'friction_flags': self.friction_flags,
            'user_rating': self.user_rating,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ScarTissue':
        return cls(
            id=d['id'],
            created=d['created'],
            conditions=[ScarCondition.from_dict(c) for c in d.get('conditions', [])],
            transforms=[PolicyTransform.from_dict(t) for t in d.get('transforms', [])],
            source_episode_id=d.get('source_episode_id', ''),
            source_description=d.get('source_description', ''),
            pain_level=d.get('pain_level', 0.5),
            activation_count=d.get('activation_count', 0),
            last_activated=d.get('last_activated', 0.0),
            confidence=d.get('confidence', 0.5),
            friction_flags=d.get('friction_flags', []),
            user_rating=d.get('user_rating', 0),
        )


# Import Tuple for type hints
from typing import Tuple


# =============================================================================
# Scar Registry
# =============================================================================

class ScarRegistry:
    """
    Registry of all learned scars.

    Provides efficient lookup and application of scars to proposed actions.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path or "/var/lib/banos/scar_tissue.json")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._scars: Dict[str, ScarTissue] = {}
        self._load()

    def _load(self) -> None:
        """Load scars from disk."""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r') as f:
                    data = json.load(f)
                for scar_data in data.get('scars', []):
                    scar = ScarTissue.from_dict(scar_data)
                    self._scars[scar.id] = scar
                logger.info(f"Loaded {len(self._scars)} scars from {self.db_path}")
            except Exception as e:
                logger.warning(f"Failed to load scars: {e}")

    def save(self) -> None:
        """Save scars to disk."""
        data = {
            'version': 2,
            'timestamp': time.time(),
            'scars': [s.to_dict() for s in self._scars.values()],
        }
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved {len(self._scars)} scars to {self.db_path}")

    def add_scar(self, scar: ScarTissue) -> None:
        """Add a new scar."""
        self._scars[scar.id] = scar
        self.save()

    def get_scar(self, scar_id: str) -> Optional[ScarTissue]:
        """Get a scar by ID."""
        return self._scars.get(scar_id)

    def remove_scar(self, scar_id: str) -> bool:
        """Remove a scar."""
        if scar_id in self._scars:
            del self._scars[scar_id]
            self.save()
            return True
        return False

    def apply_all(
        self,
        context: Dict[str, Any],
        action: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Apply all matching scars to an action.

        Args:
            context: Current context
            action: Proposed action

        Returns:
            (modified_action, list of applied scar IDs)
        """
        applied = []

        for scar in self._scars.values():
            matches, match_score = scar.matches(context)
            if matches:
                action = scar.apply(action, match_score)
                applied.append(scar.id)
                logger.debug(f"Applied scar {scar.id} (score={match_score:.2f})")

        return action, applied

    def get_warnings(
        self,
        context: Dict[str, Any],
        action: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Get warnings about matching scars without applying them.

        Useful for showing the user why an action might be modified.
        """
        warnings = []

        for scar in self._scars.values():
            matches, match_score = scar.matches(context)
            if matches:
                warnings.append({
                    'scar_id': scar.id,
                    'description': scar.source_description,
                    'match_score': match_score,
                    'transforms': [t.to_dict() for t in scar.transforms],
                    'pain_level': scar.pain_level,
                    'activation_count': scar.activation_count,
                })

        return warnings

    def get_statistics(self) -> Dict[str, Any]:
        """Get scar statistics."""
        if not self._scars:
            return {
                'total_scars': 0,
                'total_activations': 0,
            }

        return {
            'total_scars': len(self._scars),
            'total_activations': sum(s.activation_count for s in self._scars.values()),
            'avg_pain_level': sum(s.pain_level for s in self._scars.values()) / len(self._scars),
            'avg_confidence': sum(s.confidence for s in self._scars.values()) / len(self._scars),
            'most_active': max(
                self._scars.values(),
                key=lambda s: s.activation_count,
                default=None
            ),
        }


# =============================================================================
# Scar Factory - Creating Scars from Episodes
# =============================================================================

def create_scar_from_episode(
    episode_id: str,
    description: str,
    pain_level: float,
    context_snapshot: Dict[str, Any],
    lesson: str,
    friction_flags: Optional[List[str]] = None,
    user_rating: int = 0,
) -> ScarTissue:
    """
    Create a scar from an episode.

    Args:
        episode_id: Source episode ID
        description: What happened
        pain_level: How painful (0-1)
        context_snapshot: Context when the pain occurred
        lesson: What to learn
        friction_flags: User friction flags if available
        user_rating: User rating if available

    Returns:
        New ScarTissue
    """
    scar_id = hashlib.sha256(
        f"{episode_id}:{time.time_ns()}".encode()
    ).hexdigest()[:16]

    # Build conditions from context snapshot
    conditions = []

    # Activity condition
    if 'activity' in context_snapshot:
        conditions.append(ScarCondition(
            field='activity',
            operator='==',
            value=context_snapshot['activity'],
            weight=1.0,
        ))

    # App type condition
    if 'foreground' in context_snapshot:
        fg = context_snapshot['foreground']
        if isinstance(fg, dict) and 'app_type' in fg:
            conditions.append(ScarCondition(
                field='foreground.app_type',
                operator='==',
                value=fg['app_type'],
                weight=0.8,
            ))

    # Mode intrusiveness condition (if mode was intrusive)
    if 'mode' in context_snapshot:
        mode = context_snapshot['mode']
        if isinstance(mode, dict) and mode.get('intrusiveness', 0) > 0.5:
            conditions.append(ScarCondition(
                field='mode.intrusiveness',
                operator='>',
                value=0.4,
                weight=0.7,
            ))

    # Build transforms based on lesson type
    transforms = []

    # Default: add friction for this pattern
    transforms.append(PolicyTransform(
        target='energy.e_friction',
        operation='add',
        value=pain_level * 2.0,  # Scale by pain
        reason=lesson,
        strength=min(1.0, 0.5 + pain_level),
    ))

    # If mode-related, add intrusiveness reduction
    if 'mode' in context_snapshot or 'intrusive' in lesson.lower():
        transforms.append(PolicyTransform(
            target='mode.intrusiveness',
            operation='multiply',
            value=1.0 - (pain_level * 0.5),  # Reduce by up to 50%
            reason='Reduce intrusiveness based on pain',
            strength=pain_level,
        ))

    # If severe pain, consider blocking
    if pain_level > 0.8:
        transforms.append(PolicyTransform(
            target='action',
            operation='downgrade',
            value='text_minimal',
            reason='Severe pain - downgrade to minimal',
            strength=0.5,
        ))

    # Confidence starts at pain level, increases with repeated occurrences
    confidence = min(0.9, pain_level * 1.2)

    return ScarTissue(
        id=scar_id,
        created=time.time(),
        conditions=conditions,
        transforms=transforms,
        source_episode_id=episode_id,
        source_description=description,
        pain_level=pain_level,
        confidence=confidence,
        friction_flags=friction_flags or [],
        user_rating=user_rating,
    )


def create_scar_from_user_feedback(
    context_type: str,
    tool_used: str,
    style_used: str,
    rating: int,
    friction_flags: List[str],
    notes: str = "",
) -> ScarTissue:
    """
    Create a scar from user feedback.

    Called when user gives negative feedback that we should learn from.
    """
    scar_id = hashlib.sha256(
        f"user:{context_type}:{tool_used}:{time.time_ns()}".encode()
    ).hexdigest()[:16]

    # Conditions: match the context and approach
    conditions = [
        ScarCondition(
            field='context_type',
            operator='==',
            value=context_type,
            weight=1.0,
        ),
        ScarCondition(
            field='tool_used',
            operator='==',
            value=tool_used,
            weight=0.9,
        ),
    ]

    if style_used:
        conditions.append(ScarCondition(
            field='style_used',
            operator='==',
            value=style_used,
            weight=0.7,
        ))

    # Transforms based on friction flags
    transforms = []

    pain_level = abs(rating) * 0.5  # Convert rating to pain

    # Add friction
    transforms.append(PolicyTransform(
        target='energy.e_friction',
        operation='add',
        value=pain_level * 1.5,
        reason=f"User feedback: {notes or 'negative'}"[:100],
        strength=0.8,
    ))

    # Specific transforms for friction flags
    if 'too_verbose' in friction_flags:
        transforms.append(PolicyTransform(
            target='output.max_tokens',
            operation='multiply',
            value=0.6,
            reason='User finds verbose output annoying',
            strength=0.7,
        ))

    if 'too_slow' in friction_flags:
        transforms.append(PolicyTransform(
            target='latency.target',
            operation='multiply',
            value=0.7,
            reason='User wants faster responses',
            strength=0.8,
        ))

    if 'wrong_tool' in friction_flags:
        transforms.append(PolicyTransform(
            target='tool.preference',
            operation='set',
            value='avoid:' + tool_used,
            reason='User prefers different tool',
            strength=0.9,
        ))

    if 'missed_intent' in friction_flags:
        transforms.append(PolicyTransform(
            target='clarification.threshold',
            operation='add',
            value=-0.2,  # Ask more clarifying questions
            reason='User intent often missed',
            strength=0.7,
        ))

    return ScarTissue(
        id=scar_id,
        created=time.time(),
        conditions=conditions,
        transforms=transforms,
        source_episode_id='user_feedback',
        source_description=notes or f"Negative feedback for {tool_used}/{style_used}",
        pain_level=pain_level,
        confidence=0.7,
        friction_flags=friction_flags,
        user_rating=rating,
    )


# =============================================================================
# Global Registry
# =============================================================================

_registry: Optional[ScarRegistry] = None


def get_scar_registry(db_path: Optional[str] = None) -> ScarRegistry:
    """Get the global scar registry."""
    global _registry
    if _registry is None:
        _registry = ScarRegistry(db_path)
    return _registry


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scar Tissue - Policy Transforms")
    parser.add_argument("--list", action="store_true", help="List all scars")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--test", type=str, help="Test against a context (JSON)")
    args = parser.parse_args()

    registry = get_scar_registry()

    if args.list:
        for scar_id, scar in registry._scars.items():
            print(f"\n[{scar_id}] {scar.source_description[:60]}...")
            print(f"  Pain: {scar.pain_level:.2f}, Confidence: {scar.confidence:.2f}")
            print(f"  Activations: {scar.activation_count}")
            print(f"  Conditions: {len(scar.conditions)}, Transforms: {len(scar.transforms)}")

    elif args.stats:
        stats = registry.get_statistics()
        print("Scar Statistics:")
        for key, value in stats.items():
            if key != 'most_active':
                print(f"  {key}: {value}")
            elif value:
                print(f"  {key}: {value.id} ({value.activation_count} activations)")

    elif args.test:
        context = json.loads(args.test)
        action = {'mode': 'avatar_full', 'intrusiveness': 0.8}

        modified, applied = registry.apply_all(context, action)
        print(f"Applied {len(applied)} scars:")
        for scar_id in applied:
            print(f"  - {scar_id}")
        print(f"\nModified action:")
        print(json.dumps(modified, indent=2))

    else:
        parser.print_help()
