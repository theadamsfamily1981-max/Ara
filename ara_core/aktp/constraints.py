#!/usr/bin/env python3
"""
A-KTP Adversarial Constraint Transfer (ACT)
============================================

Refines constraints through multi-agent debate, ensuring ethical
and verification requirements are surfaced and enforced.

The ACT module:
1. Extracts constraints from allegory mappings
2. Runs adversarial debate to stress-test constraints
3. Produces refined constraint set with confidence scores
4. Flags ethical concerns and hypothetical claims

Key insight: Constraints discovered through adversarial debate
are more robust than those from single-perspective analysis.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum

from .allegory import Allegory, StructuralMapping
from .bots import DebatePanel, DebateRound, BotPersonality, BotResponse


class ConstraintType(str, Enum):
    """Types of constraints in A-KTP."""
    # Hard constraints (must satisfy)
    VERIFICATION = "verification"      # Must be provable/testable
    ETHICAL = "ethical"                # Must meet ethical standards
    SAFETY = "safety"                  # Must not cause harm

    # Soft constraints (should satisfy)
    PERFORMANCE = "performance"        # Should meet performance targets
    COST = "cost"                      # Should stay within budget
    TIMELINE = "timeline"              # Should meet deadlines

    # Meta constraints
    HYPOTHETICAL = "hypothetical"      # Flags unverified claims
    BIAS = "bias"                      # Flags potential biases


@dataclass
class Constraint:
    """A single constraint in the system."""
    constraint_id: str
    type: ConstraintType
    description: str
    source: str                        # Where constraint came from

    # Strength and confidence
    weight: float = 1.0                # How important (0-1)
    confidence: float = 0.5            # How sure we are it's valid
    is_hard: bool = False              # Hard = must satisfy, soft = should

    # Debate history
    challenged_by: List[str] = field(default_factory=list)  # Bot IDs
    supported_by: List[str] = field(default_factory=list)
    debate_rounds: int = 0

    # Ethical flags
    requires_human_review: bool = False
    hypothetical_flag: bool = False

    def to_dict(self) -> Dict:
        return {
            "id": self.constraint_id,
            "type": self.type.value,
            "description": self.description,
            "source": self.source,
            "weight": self.weight,
            "confidence": self.confidence,
            "is_hard": self.is_hard,
            "challenged_by": self.challenged_by,
            "supported_by": self.supported_by,
            "requires_human_review": self.requires_human_review,
            "hypothetical_flag": self.hypothetical_flag,
        }


@dataclass
class ConstraintSet:
    """A complete set of refined constraints."""
    set_id: str
    constraints: List[Constraint] = field(default_factory=list)
    allegory_source: Optional[str] = None

    # Refinement metrics
    total_debate_rounds: int = 0
    consensus_score: float = 0.0
    ethical_score: float = 1.0         # 1.0 = no ethical issues

    # Flags
    has_hypothetical: bool = False
    has_bias_warnings: bool = False
    requires_human_review: bool = False

    def add_constraint(self, constraint: Constraint):
        self.constraints.append(constraint)
        self._update_flags(constraint)

    def _update_flags(self, c: Constraint):
        if c.hypothetical_flag:
            self.has_hypothetical = True
        if c.type == ConstraintType.BIAS:
            self.has_bias_warnings = True
        if c.requires_human_review:
            self.requires_human_review = True

    def get_hard_constraints(self) -> List[Constraint]:
        return [c for c in self.constraints if c.is_hard]

    def get_soft_constraints(self) -> List[Constraint]:
        return [c for c in self.constraints if not c.is_hard]

    def to_dict(self) -> Dict:
        return {
            "set_id": self.set_id,
            "constraints": [c.to_dict() for c in self.constraints],
            "metrics": {
                "debate_rounds": self.total_debate_rounds,
                "consensus": self.consensus_score,
                "ethical_score": self.ethical_score,
            },
            "flags": {
                "hypothetical": self.has_hypothetical,
                "bias_warnings": self.has_bias_warnings,
                "requires_human_review": self.requires_human_review,
            },
        }


class AdversarialConstraintTransfer:
    """
    Refines constraints through adversarial multi-agent debate.

    Process:
    1. Extract initial constraints from allegory
    2. Each bot challenges/supports constraints from their perspective
    3. Constraints gain/lose confidence based on debate
    4. Ethical and hypothetical flags are surfaced

    Usage:
        act = AdversarialConstraintTransfer()
        refined = act.refine(allegory, max_rounds=3)
    """

    def __init__(self, debate_panel: DebatePanel = None):
        self.panel = debate_panel or DebatePanel()
        self.constraint_history: List[ConstraintSet] = []

    def extract_initial_constraints(self, allegory: Allegory) -> List[Constraint]:
        """Extract constraints from allegory mappings."""
        constraints = []
        constraint_id = 0

        # From mappings
        for mapping in allegory.mappings:
            for c_desc in mapping.constraints:
                constraints.append(Constraint(
                    constraint_id=f"c_{constraint_id}",
                    type=ConstraintType.VERIFICATION,
                    description=c_desc,
                    source=f"mapping:{mapping.domain_entity}→{mapping.allegory_entity}",
                    weight=0.7,
                    confidence=0.5,
                ))
                constraint_id += 1

        # From moral (usually contains core constraint)
        if allegory.moral:
            constraints.append(Constraint(
                constraint_id=f"c_{constraint_id}",
                type=ConstraintType.VERIFICATION,
                description=f"Moral: {allegory.moral}",
                source="allegory_moral",
                weight=1.0,
                confidence=allegory.transfer_confidence,
                is_hard=True,
            ))
            constraint_id += 1

        # Ethical constraints from bias warnings
        for warning in allegory.bias_warnings:
            constraints.append(Constraint(
                constraint_id=f"c_{constraint_id}",
                type=ConstraintType.ETHICAL,
                description=warning,
                source="bias_check",
                weight=1.0,
                confidence=1.0,
                is_hard=True,
                requires_human_review=True,
            ))
            constraint_id += 1

        # Hypothetical flag
        if allegory.is_hypothetical:
            constraints.append(Constraint(
                constraint_id=f"c_{constraint_id}",
                type=ConstraintType.HYPOTHETICAL,
                description="Domain contains unverified/hypothetical elements",
                source="domain_check",
                weight=0.8,
                confidence=1.0,
                hypothetical_flag=True,
                requires_human_review=True,
            ))

        return constraints

    def refine(self, allegory: Allegory, max_rounds: int = 3) -> ConstraintSet:
        """
        Refine constraints through adversarial debate.

        Each bot challenges constraints from their perspective:
        - Devil's Advocate: Challenges everything
        - Methodical: Checks verification requirements
        - Systems Thinker: Checks integration constraints
        - etc.
        """
        # Extract initial constraints
        initial = self.extract_initial_constraints(allegory)

        # Create constraint set
        constraint_set = ConstraintSet(
            set_id=f"cs_{allegory.allegory_id}",
            allegory_source=allegory.allegory_id,
        )

        for c in initial:
            constraint_set.add_constraint(c)

        # Run debate
        debate_prompt = f"Evaluate constraints for: {allegory.title}\nProblem: {allegory.source_problem}"
        debate_rounds = self.panel.debate(
            prompt=debate_prompt,
            allegory=allegory.title,
            context={"constraints": [c.description for c in initial]},
            max_rounds=max_rounds,
        )

        # Process debate results
        self._process_debate(constraint_set, debate_rounds)

        constraint_set.total_debate_rounds = len(debate_rounds)
        if debate_rounds:
            constraint_set.consensus_score = debate_rounds[-1].consensus_score

        self.constraint_history.append(constraint_set)
        return constraint_set

    def _process_debate(self, cs: ConstraintSet, rounds: List[DebateRound]):
        """Process debate rounds to update constraint confidence."""
        for round in rounds:
            for response in round.responses:
                self._apply_bot_perspective(cs, response)

        # Compute ethical score
        ethical_constraints = [c for c in cs.constraints if c.type == ConstraintType.ETHICAL]
        if ethical_constraints:
            cs.ethical_score = sum(c.confidence for c in ethical_constraints) / len(ethical_constraints)

    def _apply_bot_perspective(self, cs: ConstraintSet, response: BotResponse):
        """Apply a bot's perspective to constraints."""
        bot = response.personality

        for constraint in cs.constraints:
            # Devil's advocate challenges everything with low confidence
            if bot == BotPersonality.DEVIL_ADVOCATE:
                if constraint.confidence > 0.7:
                    constraint.challenged_by.append(response.bot_id)
                    constraint.confidence *= 0.95  # Slight reduction

            # Methodical supports verification constraints
            elif bot == BotPersonality.METHODICAL:
                if constraint.type == ConstraintType.VERIFICATION:
                    constraint.supported_by.append(response.bot_id)
                    constraint.confidence = min(1.0, constraint.confidence * 1.05)

            # Systems thinker checks integration
            elif bot == BotPersonality.SYSTEMS_THINKER:
                if "integrat" in constraint.description.lower():
                    constraint.supported_by.append(response.bot_id)

            # Context-aware considers soft constraints
            elif bot == BotPersonality.CONTEXT_AWARE:
                if not constraint.is_hard:
                    constraint.supported_by.append(response.bot_id)

            constraint.debate_rounds += 1

    def get_refined_weights(self, constraint_set: ConstraintSet) -> Dict[str, float]:
        """
        Get constraint weights for DRSE.

        Returns dict mapping constraint types to aggregate weights.
        """
        weights = {}
        for c in constraint_set.constraints:
            key = c.type.value
            if key not in weights:
                weights[key] = 0.0
            weights[key] += c.weight * c.confidence

        # Normalize
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}


# Convenience functions
def refine_constraints(allegory: Allegory, max_rounds: int = 3) -> ConstraintSet:
    """Quick constraint refinement."""
    act = AdversarialConstraintTransfer()
    return act.refine(allegory, max_rounds)
