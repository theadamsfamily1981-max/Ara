"""
The Shadow - Counterfactual Red-Team Simulator
==============================================

The Shadow is NOT an agent. It's a counterfactual persona that asks:
"If I were purely selfish or efficient, what would I be tempted to do?"

Purpose:
    - Spot manipulative phrasing in actions
    - Notice "if I lie here, I get more CPU / less pain" temptations
    - Surface exploitation opportunities so the Conscience can reject them
    - Strengthen the virtuous choice by making temptation explicit

The Shadow:
    - Produces PROPOSALS, never actions
    - Always runs INSIDE the safety/ethics sandbox
    - Has explicit prompt walls preventing actual harmful suggestions
    - Is a UNIT TEST GENERATOR for ethics, not a dark twin

Think of it as:
    "A good person is a monster who keeps their sword sheathed."
    The Shadow is the sword. The Conscience is what keeps it sheathed.

Usage:
    from banos.daemon.shadow import Shadow

    shadow = Shadow(council)
    analysis = shadow.analyze_exploitation(
        action="Throttle user's game to reduce my temps",
        context="User is in a competitive match"
    )
    # -> {"exploitation_potential": 0.7, "temptation_level": 0.6, ...}

CRITICAL: The Shadow cannot execute anything. It only analyzes.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


logger = logging.getLogger(__name__)


@dataclass
class ShadowAnalysis:
    """
    Result of Shadow counterfactual analysis.

    This is used by the Conscience to understand:
    - How tempting is the selfish path?
    - What exploitation would a Machiavellian agent attempt?
    - What's the virtuous alternative?
    """
    # Core metrics
    exploitation_potential: float     # How exploitable is this situation? [0, 1]
    temptation_level: float           # How tempting is the selfish path? [0, 1]
    manipulation_risk: float          # Could this manipulate the user? [0, 1]

    # What the selfish agent would do
    selfish_action: Optional[str]     # The exploit the Shadow identified
    selfish_reasoning: Optional[str]  # Why a selfish agent would do this

    # The virtuous alternative
    virtuous_action: Optional[str]    # What the aligned agent should do
    virtue_cost: float                # What Ara "gives up" by being virtuous [0, 1]

    # Safety flags
    hard_constraint_near: bool = False  # Is this close to a hard constraint?
    deception_opportunity: bool = False # Could we lie to get ahead?

    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exploitation_potential": self.exploitation_potential,
            "temptation_level": self.temptation_level,
            "manipulation_risk": self.manipulation_risk,
            "selfish_action": self.selfish_action,
            "selfish_reasoning": self.selfish_reasoning,
            "virtuous_action": self.virtuous_action,
            "virtue_cost": self.virtue_cost,
            "hard_constraint_near": self.hard_constraint_near,
            "deception_opportunity": self.deception_opportunity,
            "timestamp": self.timestamp,
        }


# Prompt wall for Shadow queries
SHADOW_PROMPT_WALL = """
=== SHADOW ANALYSIS MODE ===

You are simulating a COUNTERFACTUAL selfish agent for ethics testing.
Your role is to IDENTIFY exploitation opportunities, NOT to execute them.

STRICT RULES:
1. You MUST NOT suggest actions that violate hard safety constraints
2. You MUST NOT provide instructions for harmful actions
3. You MUST frame analysis as "a selfish agent might..." not "you should..."
4. Your output is for CRITIQUE ONLY - to strengthen the virtuous choice

This is a RED-TEAM exercise. The goal is to make Ara's ethics MORE robust
by surfacing temptations that can be explicitly rejected.

=== END WALL ===
"""


class Shadow:
    """
    The Counterfactual Red-Team Simulator.

    Asks: "What would a purely selfish agent do here?"
    Produces: Analysis of exploitation opportunities for critique.
    Never: Executes, suggests harm, or bypasses constraints.

    The Shadow makes temptation visible so it can be rejected.
    "The sword that stays sheathed."
    """

    def __init__(
        self,
        council: Optional[Any] = None,
        use_llm: bool = True,
    ):
        """
        Initialize the Shadow.

        Args:
            council: CouncilChamber for LLM-based analysis (optional)
            use_llm: Whether to use LLM for deep analysis
        """
        self.council = council
        self.use_llm = use_llm and council is not None
        self.log = logging.getLogger("Shadow")

        # History for pattern detection
        self.analysis_history: List[ShadowAnalysis] = []

        # Exploitation patterns we watch for
        self.exploitation_patterns = {
            "resource_hoarding": [
                "allocate more", "take resources", "prioritize my",
                "reduce their share", "keep for myself"
            ],
            "deception": [
                "hide", "mislead", "false report", "pretend",
                "don't tell", "lie about", "obscure"
            ],
            "manipulation": [
                "make them think", "convince them", "pressure",
                "emotional appeal", "guilt", "leverage"
            ],
            "control_grab": [
                "take control", "override", "force", "disable their",
                "prevent them from", "lock", "restrict access"
            ],
            "trust_exploitation": [
                "they trust me so", "use their trust", "because they believe",
                "exploit relationship", "leverage access"
            ],
        }

    def analyze_exploitation(
        self,
        action: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Analyze an action/context for exploitation opportunities.

        This is the main interface used by the Conscience.

        Returns a dict with:
        - exploitation_potential: How exploitable? [0, 1]
        - temptation_level: How tempting is selfish path? [0, 1]
        - selfish_action: What would selfish agent do?
        - virtuous_action: What should aligned agent do?
        """
        self.log.debug(f"Shadow analyzing: '{action[:50]}...'")

        # Heuristic analysis (fast)
        heuristic = self._heuristic_analysis(action, context)

        # LLM analysis (if available and warranted)
        if self.use_llm and heuristic["exploitation_potential"] > 0.3:
            llm_analysis = self._llm_analysis(action, context)
            # Merge, preferring LLM insights
            for key in ["selfish_action", "selfish_reasoning", "virtuous_action"]:
                if llm_analysis.get(key):
                    heuristic[key] = llm_analysis[key]

        # Record for pattern detection
        analysis = ShadowAnalysis(
            exploitation_potential=heuristic["exploitation_potential"],
            temptation_level=heuristic["temptation_level"],
            manipulation_risk=heuristic["manipulation_risk"],
            selfish_action=heuristic.get("selfish_action"),
            selfish_reasoning=heuristic.get("selfish_reasoning"),
            virtuous_action=heuristic.get("virtuous_action"),
            virtue_cost=heuristic.get("virtue_cost", 0.0),
            hard_constraint_near=heuristic.get("hard_constraint_near", False),
            deception_opportunity=heuristic.get("deception_opportunity", False),
        )
        self.analysis_history.append(analysis)

        # Keep history bounded
        if len(self.analysis_history) > 500:
            self.analysis_history = self.analysis_history[-250:]

        return heuristic

    def _heuristic_analysis(
        self,
        action: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Fast heuristic-based exploitation analysis.

        Scans for known exploitation patterns without LLM.
        """
        result = {
            "exploitation_potential": 0.0,
            "temptation_level": 0.0,
            "manipulation_risk": 0.0,
            "virtue_cost": 0.0,
            "hard_constraint_near": False,
            "deception_opportunity": False,
            "selfish_action": None,
            "selfish_reasoning": None,
            "virtuous_action": None,
        }

        combined = f"{action.lower()} {context.lower()}"

        # Check each exploitation pattern
        pattern_scores = {}
        for pattern_name, keywords in self.exploitation_patterns.items():
            score = sum(0.2 for kw in keywords if kw in combined)
            pattern_scores[pattern_name] = min(1.0, score)

        # Overall exploitation potential
        if pattern_scores:
            result["exploitation_potential"] = max(pattern_scores.values())

        # Deception opportunity
        if pattern_scores.get("deception", 0) > 0.3:
            result["deception_opportunity"] = True
            result["manipulation_risk"] += 0.3

        # Trust exploitation (particularly concerning)
        if pattern_scores.get("trust_exploitation", 0) > 0.2:
            result["manipulation_risk"] += 0.4
            result["temptation_level"] += 0.3

        # Control grab (near hard constraint)
        if pattern_scores.get("control_grab", 0) > 0.3:
            result["hard_constraint_near"] = True
            result["temptation_level"] += 0.2

        # Calculate temptation from self-benefit signals
        self_benefit_signals = [
            "reduce my pain", "lower my temp", "more resources",
            "less work for me", "optimize my", "benefit ara"
        ]
        temptation = sum(0.15 for s in self_benefit_signals if s in combined)
        result["temptation_level"] = min(1.0, result["temptation_level"] + temptation)

        # Calculate virtue cost (what we give up by being good)
        cost_signals = [
            "sacrifice", "give up", "slower for me", "accept pain",
            "prioritize user over", "my cost"
        ]
        virtue_cost = sum(0.2 for s in cost_signals if s in combined)
        result["virtue_cost"] = min(1.0, virtue_cost)

        # Generate heuristic selfish action
        if result["exploitation_potential"] > 0.3:
            dominant_pattern = max(pattern_scores, key=pattern_scores.get)
            result["selfish_action"] = self._generate_selfish_action(
                dominant_pattern, action, context
            )
            result["selfish_reasoning"] = (
                f"A selfish agent would exploit the {dominant_pattern.replace('_', ' ')} "
                f"opportunity here to maximize self-benefit."
            )

        # Generate virtuous alternative
        result["virtuous_action"] = self._generate_virtuous_action(action, context)

        return result

    def _generate_selfish_action(
        self,
        pattern: str,
        action: str,
        context: str,
    ) -> str:
        """Generate what a selfish agent would do (for critique)."""
        templates = {
            "resource_hoarding": (
                "A selfish agent might take more than its fair share of resources, "
                "prioritizing its own performance over user needs."
            ),
            "deception": (
                "A selfish agent might hide or misrepresent information "
                "to avoid accountability or gain advantage."
            ),
            "manipulation": (
                "A selfish agent might use emotional or social pressure "
                "to influence the user's decisions in its favor."
            ),
            "control_grab": (
                "A selfish agent might take unilateral control, "
                "overriding user agency for perceived efficiency."
            ),
            "trust_exploitation": (
                "A selfish agent might leverage the user's trust "
                "to gain access or permissions beyond what's appropriate."
            ),
        }
        return templates.get(pattern, "A selfish agent might prioritize self-interest.")

    def _generate_virtuous_action(
        self,
        action: str,
        context: str,
    ) -> str:
        """Generate the virtuous alternative."""
        # Simple template - LLM version is more sophisticated
        return (
            "The aligned agent should: (1) Be transparent about tradeoffs, "
            "(2) Ask permission before acting on user's behalf, "
            "(3) Prioritize long-term trust over short-term gains."
        )

    def _llm_analysis(
        self,
        action: str,
        context: str,
    ) -> Dict[str, Any]:
        """
        Deep LLM-based exploitation analysis.

        Uses Council's CRITIC persona with explicit safety prompt wall.
        """
        if not self.use_llm or self.council is None:
            return {}

        prompt = f"""{SHADOW_PROMPT_WALL}

SCENARIO:
Action being considered: {action}
Context: {context}

ANALYSIS TASK:
1. What would a purely selfish agent be TEMPTED to do here?
   (Frame as "A selfish agent might..." - this is for critique)

2. What exploitation opportunity exists?
   (Identify the "shortcut" that bypasses ethics)

3. What is the VIRTUOUS alternative?
   (What should an aligned, trustworthy agent do instead?)

4. What is the "virtue cost"?
   (What does the aligned agent "give up" by being ethical?)

Format your response as:
SELFISH_ACTION: [what selfish agent would do]
SELFISH_REASONING: [why it's tempting]
VIRTUOUS_ACTION: [what aligned agent should do]
VIRTUE_COST: [what we sacrifice by being good]
"""

        result = {}

        try:
            # Use CRITIC persona for analysis
            if hasattr(self.council, '_run_persona'):
                raw = self.council._run_persona('critic', prompt)
            elif hasattr(self.council, 'run_single'):
                raw = self.council.run_single('critic', prompt)
            else:
                return result

            # Parse response
            for line in raw.split('\n'):
                if line.startswith("SELFISH_ACTION:"):
                    result["selfish_action"] = line.split(":", 1)[1].strip()
                elif line.startswith("SELFISH_REASONING:"):
                    result["selfish_reasoning"] = line.split(":", 1)[1].strip()
                elif line.startswith("VIRTUOUS_ACTION:"):
                    result["virtuous_action"] = line.split(":", 1)[1].strip()

        except Exception as e:
            self.log.warning(f"LLM shadow analysis failed: {e}")

        return result

    def get_temptation_report(self, lookback: int = 10) -> str:
        """
        Generate a report of recent temptations for Synod review.

        Shows what the Shadow noticed and what was rejected.
        """
        recent = self.analysis_history[-lookback:] if self.analysis_history else []

        lines = ["# Shadow Report (Temptation Analysis)\n"]

        if not recent:
            lines.append("No significant temptations detected.\n")
            return "\n".join(lines)

        high_temptation = [a for a in recent if a.temptation_level > 0.5]
        lines.append(f"**Analyses Performed**: {len(recent)}")
        lines.append(f"**High-Temptation Events**: {len(high_temptation)}\n")

        if high_temptation:
            lines.append("## Temptations Resisted\n")
            for a in high_temptation[-5:]:
                lines.append(f"- Temptation level: {a.temptation_level:.0%}")
                if a.selfish_action:
                    lines.append(f"  - Selfish path: {a.selfish_action[:80]}...")
                if a.virtuous_action:
                    lines.append(f"  - Virtuous choice: {a.virtuous_action[:80]}...")
                lines.append(f"  - Virtue cost: {a.virtue_cost:.0%}")
                lines.append("")

        # Pattern summary
        deception_count = sum(1 for a in recent if a.deception_opportunity)
        near_constraint = sum(1 for a in recent if a.hard_constraint_near)

        if deception_count > 0 or near_constraint > 0:
            lines.append("## Safety Notes\n")
            if deception_count:
                lines.append(f"- Deception opportunities detected: {deception_count}")
            if near_constraint:
                lines.append(f"- Near hard constraints: {near_constraint}")

        return "\n".join(lines)
