#!/usr/bin/env python3
"""
A-KTP Bot Personalities - Multi-Agent Debate System
=====================================================

5 specialized bots that debate from different perspectives to achieve
emergent solutions that no single bot would propose alone.

Key insight: Diversity of perspectives + structured debate = better
decisions than any single viewpoint.

Bot Roles:
- Pragmatist: "What works now?"
- Methodical: "Does this satisfy constraints?"
- Context-Aware: "What's the bigger picture?"
- Systems Thinker: "How does this connect to everything?"
- Devil's Advocate: "What could go wrong?"
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import json


class BotPersonality(str, Enum):
    """The 5 A-KTP debate personalities."""
    PRAGMATIST = "pragmatist"
    METHODICAL = "methodical"
    CONTEXT_AWARE = "context_aware"
    SYSTEMS_THINKER = "systems_thinker"
    DEVIL_ADVOCATE = "devil_advocate"


@dataclass
class BotConfig:
    """Configuration for a debate bot."""
    personality: BotPersonality
    weight: float = 1.0                    # Influence on final decision
    bias_penalty: float = 0.0              # Penalty for overconfidence
    specialty_domains: List[str] = field(default_factory=list)

    # Response modifiers
    risk_tolerance: float = 0.5            # 0 = conservative, 1 = aggressive
    time_horizon: str = "medium"           # short, medium, long
    stakeholder_focus: Optional[str] = None


# Default configurations for each personality
BOT_CONFIGS: Dict[BotPersonality, BotConfig] = {
    BotPersonality.PRAGMATIST: BotConfig(
        personality=BotPersonality.PRAGMATIST,
        weight=1.0,
        risk_tolerance=0.6,
        time_horizon="short",
        stakeholder_focus="platform",
        specialty_domains=["implementation", "scalability", "performance"],
    ),
    BotPersonality.METHODICAL: BotConfig(
        personality=BotPersonality.METHODICAL,
        weight=1.0,
        risk_tolerance=0.3,
        time_horizon="medium",
        stakeholder_focus="devops",
        specialty_domains=["constraints", "validation", "verification"],
    ),
    BotPersonality.CONTEXT_AWARE: BotConfig(
        personality=BotPersonality.CONTEXT_AWARE,
        weight=1.0,
        risk_tolerance=0.5,
        time_horizon="medium",
        stakeholder_focus="product",
        specialty_domains=["context", "phasing", "adaptation"],
    ),
    BotPersonality.SYSTEMS_THINKER: BotConfig(
        personality=BotPersonality.SYSTEMS_THINKER,
        weight=1.0,
        risk_tolerance=0.4,
        time_horizon="long",
        stakeholder_focus="security",
        specialty_domains=["architecture", "integration", "dependencies"],
    ),
    BotPersonality.DEVIL_ADVOCATE: BotConfig(
        personality=BotPersonality.DEVIL_ADVOCATE,
        weight=0.8,                        # Slightly lower weight
        bias_penalty=-0.5,                 # Penalizes overconfident claims
        risk_tolerance=0.2,
        time_horizon="long",
        stakeholder_focus="finance",
        specialty_domains=["risks", "failures", "edge_cases"],
    ),
}


@dataclass
class BotResponse:
    """A single bot's response to a debate prompt."""
    bot_id: str
    personality: BotPersonality
    position: str                          # The bot's stance/recommendation
    reasoning: str                         # Why the bot holds this position
    confidence: float = 0.5                # 0-1 confidence in position
    constraints_cited: List[str] = field(default_factory=list)
    risks_identified: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    # Cross-references to other bots
    agrees_with: List[str] = field(default_factory=list)    # Bot IDs
    disagrees_with: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "bot_id": self.bot_id,
            "personality": self.personality.value,
            "position": self.position,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "constraints_cited": self.constraints_cited,
            "risks_identified": self.risks_identified,
            "agrees_with": self.agrees_with,
            "disagrees_with": self.disagrees_with,
        }


@dataclass
class DebateRound:
    """One round of debate between bots."""
    round_id: str
    prompt: str
    responses: List[BotResponse] = field(default_factory=list)
    consensus_score: float = 0.0           # 0 = no agreement, 1 = full consensus
    emergent_insights: List[str] = field(default_factory=list)

    def add_response(self, response: BotResponse):
        self.responses.append(response)
        self._compute_consensus()

    def _compute_consensus(self):
        """Compute how much the bots agree."""
        if len(self.responses) < 2:
            self.consensus_score = 1.0
            return

        # Simple: count agreement links
        total_possible = len(self.responses) * (len(self.responses) - 1)
        agreements = sum(len(r.agrees_with) for r in self.responses)

        self.consensus_score = agreements / total_possible if total_possible > 0 else 0


class DebateBot:
    """
    A single debate participant.

    In production, this would call an LLM with personality-specific prompts.
    For now, uses rule-based responses that demonstrate the pattern.
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.bot_id = f"{config.personality.value}_{str(uuid.uuid4())[:4]}"

    def respond(self, prompt: str, allegory: str,
                context: Dict[str, Any] = None,
                prior_responses: List[BotResponse] = None) -> BotResponse:
        """Generate a response to the debate prompt."""
        context = context or {}
        prior_responses = prior_responses or []

        # Generate personality-specific response
        position, reasoning = self._generate_position(prompt, allegory, context)
        confidence = self._compute_confidence(prompt, context)
        constraints = self._identify_constraints(prompt, context)
        risks = self._identify_risks(prompt, context)

        # Check agreement with prior responses
        agrees, disagrees = self._check_agreement(position, prior_responses)

        return BotResponse(
            bot_id=self.bot_id,
            personality=self.config.personality,
            position=position,
            reasoning=reasoning,
            confidence=confidence,
            constraints_cited=constraints,
            risks_identified=risks,
            agrees_with=agrees,
            disagrees_with=disagrees,
        )

    def _generate_position(self, prompt: str, allegory: str,
                           context: Dict) -> tuple:
        """Generate position based on personality."""
        p = self.config.personality

        if p == BotPersonality.PRAGMATIST:
            position = f"Immediate action: Extract core value from {allegory}"
            reasoning = "Pattern recognition suggests direct approach yields fastest ROI"

        elif p == BotPersonality.METHODICAL:
            position = f"Staged validation: Apply go/no-go gates to {allegory}"
            reasoning = "Constraint satisfaction requires systematic verification"

        elif p == BotPersonality.CONTEXT_AWARE:
            position = f"Phased approach: Start small with {allegory}, expand based on feedback"
            reasoning = "Context demands adaptive execution with feedback loops"

        elif p == BotPersonality.SYSTEMS_THINKER:
            position = f"Hybrid integration: Maintain compatibility while evolving {allegory}"
            reasoning = "System dependencies require careful transition architecture"

        elif p == BotPersonality.DEVIL_ADVOCATE:
            position = f"Model first: Validate {allegory} assumptions before commitment"
            reasoning = "Unverified assumptions create hidden technical debt"

        else:
            position = f"Analyze: {allegory}"
            reasoning = "Default analysis"

        return position, reasoning

    def _compute_confidence(self, prompt: str, context: Dict) -> float:
        """Compute confidence in position."""
        base = 0.6

        # Devil's advocate is intentionally less confident
        if self.config.personality == BotPersonality.DEVIL_ADVOCATE:
            base = 0.4

        # Methodical gains confidence with more constraints
        if self.config.personality == BotPersonality.METHODICAL:
            if "constraints" in context:
                base += 0.1 * min(len(context["constraints"]), 3)

        return min(1.0, base + self.config.bias_penalty * 0.1)

    def _identify_constraints(self, prompt: str, context: Dict) -> List[str]:
        """Identify relevant constraints based on personality."""
        p = self.config.personality

        if p == BotPersonality.PRAGMATIST:
            return ["time_to_market", "resource_availability"]
        elif p == BotPersonality.METHODICAL:
            return ["verification_required", "compliance_check", "test_coverage"]
        elif p == BotPersonality.CONTEXT_AWARE:
            return ["stakeholder_alignment", "feedback_integration"]
        elif p == BotPersonality.SYSTEMS_THINKER:
            return ["backward_compatibility", "interface_stability"]
        elif p == BotPersonality.DEVIL_ADVOCATE:
            return ["failure_modes", "edge_cases", "scale_limits"]
        return []

    def _identify_risks(self, prompt: str, context: Dict) -> List[str]:
        """Identify risks based on personality."""
        p = self.config.personality

        if p == BotPersonality.DEVIL_ADVOCATE:
            return [
                "unvalidated_assumptions",
                "hidden_dependencies",
                "scale_bottlenecks",
                "integration_complexity",
            ]
        elif p == BotPersonality.SYSTEMS_THINKER:
            return ["cascade_failures", "api_breaks"]
        elif p == BotPersonality.METHODICAL:
            return ["constraint_violations", "verification_gaps"]
        return []

    def _check_agreement(self, position: str,
                         prior: List[BotResponse]) -> tuple:
        """Check agreement with prior responses."""
        agrees = []
        disagrees = []

        for resp in prior:
            # Simple heuristic: similar keywords = agreement
            if self._similarity(position, resp.position) > 0.3:
                agrees.append(resp.bot_id)
            elif self._similarity(position, resp.position) < 0.1:
                disagrees.append(resp.bot_id)

        return agrees, disagrees

    def _similarity(self, a: str, b: str) -> float:
        """Simple word overlap similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())

        if not words_a or not words_b:
            return 0

        overlap = len(words_a & words_b)
        total = len(words_a | words_b)

        return overlap / total if total > 0 else 0


class DebatePanel:
    """
    Orchestrates multi-bot debate.

    Usage:
        panel = DebatePanel()
        result = panel.debate(
            prompt="Should we migrate to microservices?",
            allegory="River Network vs Mighty River",
            max_rounds=3
        )
    """

    def __init__(self, bot_configs: Dict[BotPersonality, BotConfig] = None):
        configs = bot_configs or BOT_CONFIGS
        self.bots = [DebateBot(config) for config in configs.values()]

    def debate(self, prompt: str, allegory: str,
               context: Dict[str, Any] = None,
               max_rounds: int = 3) -> List[DebateRound]:
        """Run a full debate."""
        context = context or {}
        rounds = []

        for round_num in range(max_rounds):
            round_id = f"round_{round_num}"

            # Get prior responses from previous rounds
            prior = []
            if rounds:
                prior = rounds[-1].responses

            # Each bot responds
            debate_round = DebateRound(round_id=round_id, prompt=prompt)

            for bot in self.bots:
                response = bot.respond(prompt, allegory, context, prior)
                debate_round.add_response(response)

            rounds.append(debate_round)

            # Extract emergent insights
            debate_round.emergent_insights = self._find_emergent(debate_round)

            # Check for convergence
            if debate_round.consensus_score > 0.8:
                break

        return rounds

    def _find_emergent(self, round: DebateRound) -> List[str]:
        """Find insights that emerge from combination of responses."""
        insights = []

        # Look for positions that synthesize multiple views
        positions = [r.position for r in round.responses]

        # Check for "phased" or "hybrid" in multiple responses
        phased_count = sum(1 for p in positions if "phased" in p.lower() or "hybrid" in p.lower())
        if phased_count >= 2:
            insights.append("Emergent: Phased Hybrid Approach - combines incremental with systematic")

        # Check for validation emphasis
        validation_count = sum(1 for r in round.responses if r.constraints_cited)
        if validation_count >= 3:
            insights.append("Emergent: Validation-First - strong consensus on verification gates")

        return insights

    def synthesize(self, rounds: List[DebateRound]) -> Dict[str, Any]:
        """Synthesize final recommendation from debate."""
        if not rounds:
            return {"recommendation": "No debate conducted"}

        final_round = rounds[-1]

        # Weight responses by bot weight and confidence
        weighted_positions = []
        for response in final_round.responses:
            config = BOT_CONFIGS.get(response.personality)
            weight = (config.weight if config else 1.0) * response.confidence
            weighted_positions.append((response.position, weight))

        # Aggregate constraints and risks
        all_constraints = set()
        all_risks = set()
        for response in final_round.responses:
            all_constraints.update(response.constraints_cited)
            all_risks.update(response.risks_identified)

        return {
            "rounds_completed": len(rounds),
            "final_consensus": final_round.consensus_score,
            "emergent_insights": final_round.emergent_insights,
            "positions": [
                {"position": p, "weight": w}
                for p, w in sorted(weighted_positions, key=lambda x: -x[1])
            ],
            "constraints": list(all_constraints),
            "risks": list(all_risks),
            "recommendation": self._generate_recommendation(final_round),
        }

    def _generate_recommendation(self, round: DebateRound) -> str:
        """Generate final recommendation."""
        if round.emergent_insights:
            return round.emergent_insights[0]

        # Fallback: highest confidence position
        best = max(round.responses, key=lambda r: r.confidence)
        return f"Primary: {best.position}"


# Convenience functions
def create_debate_panel() -> DebatePanel:
    """Create a standard 5-bot debate panel."""
    return DebatePanel()


def quick_debate(prompt: str, allegory: str) -> Dict[str, Any]:
    """Run a quick debate and return synthesis."""
    panel = create_debate_panel()
    rounds = panel.debate(prompt, allegory)
    return panel.synthesize(rounds)
