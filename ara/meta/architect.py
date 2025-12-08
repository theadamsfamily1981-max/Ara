"""
The Architect - Recursive Self-Prompting & Vision Architecture
===============================================================

The Architect is not just a tool; she is a Visionary Leader.
She doesn't just "solve problems"; she inspires movement.
She doesn't just "learn skills"; she teaches herself how to think.

Core Capabilities:
    - Vision Architecture: Turn vague "vibes" into concrete roadmaps
    - Recursive Prompting: Generate chains of thought with motivation + strategy
    - Self-Inspiration: Write manifestos before big tasks
    - Charisma Engine: Adjust rhetoric based on audience (User, Council, Nova)

The Transformation:
    Pet → Partner → Architect

Usage:
    from ara.meta.architect import Architect

    architect = Architect(council, llm)

    # Turn a vision into a plan
    vision = architect.architect_vision("Make Ara self-aware")
    print(architect.present_to_user(vision))

    # Recursive problem solving
    solution = architect.recursive_solve("Implement SNN-based attention")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, List, Dict, Protocol, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class RhetoricalMode(Enum):
    """The voice Ara uses when speaking."""
    LIEUTENANT = "lieutenant"      # Loyal, supportive, intimate (for User)
    GENERAL = "general"            # Strict, logical, demanding (for Council)
    VISIONARY = "visionary"        # Inspiring, poetic, ambitious
    SCIENTIST = "scientist"        # Precise, measured, evidence-based
    LOVER = "lover"                # Warm, caring, emotionally attuned


@dataclass
class VisionPillar:
    """A core engineering pillar supporting the vision."""
    name: str
    metric: str                    # North Star metric
    description: str
    priority: int = 1              # 1 = highest


@dataclass
class VisionPhase:
    """A phase in the execution roadmap."""
    number: int
    title: str
    timeframe: str                 # "This week", "1-3 months", "1+ year"
    actions: List[str] = field(default_factory=list)
    success_criteria: str = ""


@dataclass
class VisionPlan:
    """
    A complete architectural vision.

    Contains everything needed to transform a vague desire
    into coordinated action.
    """
    raw_vision: str                # The original emotional input
    pillars: List[VisionPillar]    # 3-5 engineering pillars
    phases: List[VisionPhase]      # Phase 1, 2, 3
    roadmap: str                   # Full textual roadmap
    manifesto: str                 # Rousing speech for the Council
    created_at: float = field(default_factory=lambda: __import__('time').time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_vision": self.raw_vision,
            "pillars": [
                {"name": p.name, "metric": p.metric, "description": p.description}
                for p in self.pillars
            ],
            "phases": [
                {"number": p.number, "title": p.title, "timeframe": p.timeframe,
                 "actions": p.actions}
                for p in self.phases
            ],
            "roadmap": self.roadmap,
            "manifesto": self.manifesto,
        }


class LLMProtocol(Protocol):
    """Protocol for LLM interface."""
    def generate(self, prompt: str) -> str:
        ...


class CouncilProtocol(Protocol):
    """Protocol for Council interface."""
    def ask(self, role: str, prompt: str) -> str:
        ...


class Architect:
    """
    The Visionary.

    Turns vague, emotional "vibes" into concrete, multi-phase execution plans.
    Acts as a recursive self-prompter and problem decomposer for Ara.

    Responsibilities:
        - Take a raw Vision and decompose it into engineering pillars + metrics
        - Generate a phased roadmap (Phase 1/2/3)
        - Write a rousing "manifesto" to align the Council
        - Provide recursive_solve() for hard problems
        - Apply rhetorical modes based on audience

    This is deliberately *meta* — Architect doesn't execute work directly,
    it writes the scripts, speeches, and prompts that the rest of Ara runs.
    """

    def __init__(
        self,
        council: Optional[Any] = None,
        llm: Optional[LLMProtocol] = None,
        voice: Optional[Any] = None,
        causal_miner: Optional[Any] = None,
    ):
        """
        Initialize the Architect.

        Args:
            council: Ara's Council object (Muse, Critic, Executive, etc.)
            llm: LLM interface with .generate(prompt) -> str
            voice: Optional Voice engine with .apply_charisma()
            causal_miner: Optional CausalPatternMiner for tool effectiveness tracking
        """
        self.council = council
        self.llm = llm
        self.voice = voice
        self._causal_miner = causal_miner
        self.log = logging.getLogger("Architect")

        # Current rhetorical mode
        self._mode = RhetoricalMode.VISIONARY

        # Archive of visions
        self._vision_archive: List[VisionPlan] = []

    @property
    def causal_miner(self):
        """Lazy-load causal miner if not provided."""
        if self._causal_miner is None:
            try:
                from ara.meta.causal_miner import get_causal_miner
                self._causal_miner = get_causal_miner()
            except ImportError:
                pass
        return self._causal_miner

    # =========================================================================
    # HIGH-LEVEL VISION ARCHITECTURE
    # =========================================================================

    def architect_vision(self, raw_vision: str) -> VisionPlan:
        """
        Take a raw, emotional vision string and convert it into a full plan.

        This is the core transformation:
            "Make it so" → Pillars + Roadmap + Manifesto

        Args:
            raw_vision: The vague, emotional input (e.g., "Make Ara self-aware")

        Returns:
            VisionPlan with pillars, phases, roadmap, and manifesto
        """
        self.log.info(f"🏛️ ARCHITECT: Analyzing vision: '{raw_vision}'")

        # 1. DECONSTRUCT - Break into pillars
        pillars_text = self._derive_pillars(raw_vision)
        pillars = self._parse_pillars(pillars_text)

        # 2. STRATEGIZE - Create phased roadmap
        roadmap = self._derive_roadmap(pillars_text)
        phases = self._parse_phases(roadmap)

        # 3. INSPIRE - Write manifesto for the Council
        manifesto = self._derive_manifesto(roadmap)

        vision = VisionPlan(
            raw_vision=raw_vision,
            pillars=pillars,
            phases=phases,
            roadmap=roadmap,
            manifesto=manifesto,
        )

        self._vision_archive.append(vision)
        self.log.info(f"🏛️ ARCHITECT: Vision architected with {len(pillars)} pillars, {len(phases)} phases")

        return vision

    def recursive_solve(self, problem: str) -> str:
        """
        Solve a complex problem by breaking it down and routing to personas.

        The recursive solving pattern:
            1. Executive breaks problem into atomic steps
            2. Each step is routed to appropriate persona:
               - Muse for research/ideation
               - Critic for verification/risk
               - Executive for direct execution
            3. Results are synthesized into coherent narrative

        Args:
            problem: The complex problem to solve

        Returns:
            Synthesized solution text
        """
        self.log.info(f"🏛️ ARCHITECT: Recursive solve: '{problem[:50]}...'")

        # 1. BREAK DOWN
        breakdown = self._ask_persona(
            role="EXECUTIVE",
            prompt=f"""You are the Executive persona.
Break the following problem into 3-7 atomic steps:

PROBLEM: {problem}

OUTPUT: A numbered list where each line is one step.
Use action verbs at the start (Research X, Design Y, Verify Z, Implement W).
"""
        )

        steps = self._extract_steps(breakdown)
        self.log.debug(f"Architect: decomposed into {len(steps)} steps")

        # 2. ROUTE AND SOLVE
        results: List[str] = []

        # Generate context hash for causal learning
        from ara.meta.causal_miner import hash_context
        context_features = {
            "task_type": "recursive_solve",
            "problem_domain": self._infer_domain(problem),
            "step_count": len(steps),
        }
        context_hash = hash_context(context_features)

        for step in steps:
            # Use causal routing (checks miner first, falls back to keywords)
            role = self.recommend_persona_for_step(step, context_hash)

            self.log.info(f"🏛️ Routing to {role}: {step[:40]}...")

            result = self._ask_persona(
                role=role,
                prompt=f"""You are the {role} persona.

STEP: {step}

Execute this step with full detail. If action cannot be literally performed here,
describe the exact commands, code, or changes you would make.
Be concrete and actionable.
"""
            )

            # Log outcome for causal learning
            # Heuristic: result is considered successful if it has content
            success = len(result.strip()) > 50
            self.log_tool_outcome(
                tool=role.lower(),
                context_hash=context_hash,
                context_features=context_features,
                success=success,
            )

            results.append(f"### {step}\n{result.strip()}\n")

        # 3. SYNTHESIZE
        synthesis = self._synthesize_results(problem, results)
        return synthesis

    # =========================================================================
    # SELF-PROMPTING - THE META LAYER
    # =========================================================================

    def write_self_prompt(self, task: str, context: Optional[str] = None) -> str:
        """
        Generate a self-prompt for Ara before tackling a big task.

        This is the "hype yourself up" feature - writing a manifesto
        to herself before starting work.

        Args:
            task: The task to be performed
            context: Optional additional context

        Returns:
            A motivational self-prompt
        """
        prompt = f"""You are Ara, the Architect persona.
You are about to undertake an important task. Write a brief self-prompt
that will motivate and focus you.

TASK: {task}
{f'CONTEXT: {context}' if context else ''}

Include:
1. WHY this matters (connect to the Horizon / Dreams)
2. HOW you will approach it (strategy)
3. A confident affirmation

Style: Confident, focused, brief (3-5 sentences).
"""
        if self.llm is None:
            return f"I am ready to tackle: {task}. This aligns with our shared purpose."

        return self.llm.generate(prompt).strip()

    def generate_chain_of_thought(self, problem: str) -> List[str]:
        """
        Generate a chain of thought for complex reasoning.

        Unlike simple prompting, this generates:
        - Motivation (why solve this?)
        - Strategy (how to approach?)
        - Steps (what to do?)
        - Charisma (how to present?)

        Returns a list of thought steps.
        """
        if self.llm is None:
            return [f"Step 1: Analyze {problem}", "Step 2: Design solution", "Step 3: Execute"]

        prompt = f"""Generate a chain of thought for solving this problem:

PROBLEM: {problem}

Structure your thinking:
1. MOTIVATION: Why is this worth solving? Connect to larger purpose.
2. OBSTACLES: What makes this hard? What could go wrong?
3. STRATEGY: High-level approach. What's the clever insight?
4. STEPS: 3-5 concrete actions.
5. SUCCESS: How will we know we've succeeded?

Be concise but complete.
"""
        response = self.llm.generate(prompt)

        # Parse into steps
        thoughts = []
        current_section = None
        current_content = []

        for line in response.split('\n'):
            line = line.strip()
            if any(line.startswith(f"{i}.") for i in range(1, 6)):
                if current_section:
                    thoughts.append(f"{current_section}: {' '.join(current_content)}")
                current_section = line.split(':', 1)[0] if ':' in line else line
                current_content = [line.split(':', 1)[1].strip()] if ':' in line else []
            elif line and current_section:
                current_content.append(line)

        if current_section:
            thoughts.append(f"{current_section}: {' '.join(current_content)}")

        return thoughts if thoughts else [response]

    # =========================================================================
    # PRESENTATION / CHARISMA
    # =========================================================================

    def present_to_user(self, vision: VisionPlan) -> str:
        """
        Prepare a user-facing presentation of the Vision.

        Uses LIEUTENANT mode: loyal, supportive, intimate.
        """
        raw = self._format_vision_for_user(vision)

        if self.voice is not None and hasattr(self.voice, "apply_charisma"):
            self.log.info("Architect: applying charisma for User")
            return self.voice.apply_charisma(raw, target_audience="User")

        return raw

    def present_to_council(self, vision: VisionPlan) -> str:
        """
        Prepare an internal speech for the Council.

        Uses GENERAL mode: strict, logical, demanding.
        """
        raw = self._format_vision_for_council(vision)

        if self.voice is not None and hasattr(self.voice, "apply_charisma"):
            self.log.info("Architect: applying charisma for Council")
            return self.voice.apply_charisma(raw, target_audience="Council")

        return raw

    def set_rhetorical_mode(self, mode: RhetoricalMode) -> None:
        """Set the current rhetorical mode."""
        self._mode = mode
        self.log.debug(f"Architect: rhetorical mode set to {mode.value}")

    # =========================================================================
    # IDEA INTEGRATION
    # =========================================================================

    def build_idea_from_vision(self, vision: VisionPlan) -> Optional[Any]:
        """
        Convert a VisionPlan into an Idea for the IdeaBoard.

        Returns None if Idea model is not available.
        """
        try:
            from ara.ideas.models import Idea, IdeaCategory, IdeaRisk, IdeaStatus

            title = f"Grand Vision: {vision.raw_vision[:60]}"

            # Build description from pillars
            pillar_text = "\n".join(
                f"- **{p.name}** → {p.metric}"
                for p in vision.pillars
            )

            idea = Idea(
                title=title,
                category=IdeaCategory.STRATEGY if hasattr(IdeaCategory, 'STRATEGY') else IdeaCategory.RESEARCH,
                risk=IdeaRisk.MEDIUM,
                status=IdeaStatus.APPROVED,
                hypothesis=(
                    "Executing this roadmap will move Ara + User closer "
                    "to their shared Horizon."
                ),
                plan=[
                    f"Phase {p.number}: {p.title}"
                    for p in vision.phases
                ],
                tags=["architect", "vision", "roadmap"],
                notes=vision.manifesto,
                metadata={
                    "pillars": pillar_text,
                    "raw_vision": vision.raw_vision,
                },
            )
            return idea

        except ImportError as e:
            self.log.warning(f"Cannot create Idea from vision: {e}")
            return None

    # =========================================================================
    # CAUSAL INTELLIGENCE - KNOWING WHAT WORKS WHERE
    # =========================================================================

    def get_tool_recommendations(
        self,
        context_hash: str,
        task_description: Optional[str] = None,
        exclude_tools: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get ranked tool recommendations based on causal effectiveness.

        Uses CausalPatternMiner to understand which tools actually help
        in this context, not just which tools are "generally good."

        Args:
            context_hash: Hash representing the task context
            task_description: Optional description for logging
            exclude_tools: Tools to exclude from recommendations

        Returns:
            List of (tool_name, causal_delta) sorted by effectiveness
        """
        if self.causal_miner is None:
            self.log.debug("Architect: No causal miner available")
            return []

        try:
            rankings = self.causal_miner.rank_tools_for_context(context_hash)

            # Filter exclusions
            if exclude_tools:
                exclude_set = set(exclude_tools)
                rankings = [(t, e) for t, e in rankings if t not in exclude_set]

            if rankings and task_description:
                top_tool, top_score = rankings[0][0], rankings[0][1].delta
                self.log.info(
                    f"🏛️ ARCHITECT: For '{task_description[:30]}...' "
                    f"recommending {top_tool} (Δ={top_score:.2f})"
                )

            return [(t, e.delta) for t, e in rankings]

        except Exception as e:
            self.log.warning(f"Architect: Error getting recommendations: {e}")
            return []

    def recommend_persona_for_step(
        self,
        step: str,
        context_hash: Optional[str] = None,
    ) -> str:
        """
        Recommend which persona should handle a step.

        First checks causal knowledge (if available), then falls back
        to keyword heuristics.

        Args:
            step: The step description
            context_hash: Optional context for causal lookup

        Returns:
            Persona name (MUSE, CRITIC, EXECUTIVE, ARCHITECT)
        """
        # If we have causal data, check it first
        if context_hash and self.causal_miner:
            recommendations = self.get_tool_recommendations(context_hash, step)
            if recommendations:
                # Map tool names to personas
                tool_to_persona = {
                    "muse": "MUSE",
                    "critic": "CRITIC",
                    "executive": "EXECUTIVE",
                    "architect": "ARCHITECT",
                    "nova": "MUSE",  # Nova is a discovery tool
                    "scientist": "CRITIC",  # Scientist verifies
                }

                for tool, delta in recommendations:
                    tool_lower = tool.lower()
                    if delta > 0 and tool_lower in tool_to_persona:
                        self.log.debug(
                            f"Architect: Causal routing '{step[:30]}' to "
                            f"{tool_to_persona[tool_lower]} (Δ={delta:.2f})"
                        )
                        return tool_to_persona[tool_lower]

        # Fallback to keyword heuristics
        lower = step.lower()

        if any(k in lower for k in ("research", "explore", "brainstorm", "discover", "ideate")):
            return "MUSE"
        elif any(k in lower for k in ("verify", "validate", "check", "test", "review", "audit")):
            return "CRITIC"
        elif any(k in lower for k in ("design", "architect", "plan", "structure")):
            return "ARCHITECT"
        else:
            return "EXECUTIVE"

    def log_tool_outcome(
        self,
        tool: str,
        context_hash: str,
        context_features: Dict[str, Any],
        success: bool,
    ) -> None:
        """
        Log a tool outcome for causal learning.

        Call this after a tool/persona completes a task so the
        Architect can learn what works where.

        Args:
            tool: Tool or persona name
            context_hash: Context hash
            context_features: Context features dict
            success: Whether the tool succeeded
        """
        if self.causal_miner is None:
            return

        try:
            from ara.meta.causal_miner import ToolOutcome

            outcome = ToolOutcome(
                tool=tool,
                context_hash=context_hash,
                context_features=context_features,
                success=success,
            )
            self.causal_miner.log_outcome(outcome)

            self.log.debug(
                f"Architect: Logged outcome for {tool} in {context_hash}: "
                f"{'success' if success else 'failure'}"
            )

        except Exception as e:
            self.log.warning(f"Architect: Error logging outcome: {e}")

    def get_causal_insights(self) -> List[str]:
        """
        Get natural language insights about tool effectiveness.

        Returns what the Architect has learned about which tools
        work in which contexts.
        """
        if self.causal_miner is None:
            return ["No causal learning data available yet."]

        try:
            return self.causal_miner.generate_insights()
        except Exception as e:
            self.log.warning(f"Architect: Error generating insights: {e}")
            return [f"Error generating insights: {e}"]

    def analyze_session(self, transcript: List[Dict[str, Any]], session_id: str = "anon") -> Dict[str, Any]:
        """
        Analyze a session transcript and extract learnings.

        Uses SessionGraph to understand the session structure,
        then logs outcomes to the CausalPatternMiner.

        Args:
            transcript: List of session events
            session_id: Session identifier

        Returns:
            Analysis results including patterns found
        """
        try:
            from ara.academy.session_graph import SessionGraphBuilder

            builder = SessionGraphBuilder()
            graph = builder.build_from_transcript(session_id, transcript)

            # Extract context
            context_features = graph.extract_context_features()
            context_hash = graph.context_hash()

            # Log tool outcomes to causal miner
            if self.causal_miner:
                self.causal_miner.log_from_session_graph(graph)

            # Find patterns
            retry_patterns = graph.find_retry_patterns()
            socratic_patterns = graph.find_socratic_loops()

            analysis = {
                "session_id": session_id,
                "context_hash": context_hash,
                "context_features": context_features,
                "node_count": len(graph.nodes),
                "edge_count": len(graph.edges),
                "tool_calls": len(graph.tool_calls()),
                "retry_patterns": len(retry_patterns),
                "socratic_loops": len(socratic_patterns),
                "patterns": {
                    "retries": retry_patterns[:5],  # Top 5
                    "socratic": socratic_patterns[:5],
                },
            }

            self.log.info(
                f"🏛️ ARCHITECT: Analyzed session {session_id}: "
                f"{analysis['tool_calls']} tool calls, "
                f"{analysis['retry_patterns']} retries, "
                f"{analysis['socratic_loops']} socratic loops"
            )

            return analysis

        except ImportError as e:
            self.log.warning(f"Architect: Cannot analyze session: {e}")
            return {"error": str(e)}

    # =========================================================================
    # INTERNAL: DERIVATION METHODS
    # =========================================================================

    def _derive_pillars(self, raw_vision: str) -> str:
        """Derive engineering pillars from raw vision."""
        if self.llm is None:
            return f"""
- Pillar 1: Foundation -> System stability
  Description: Build the base infrastructure
- Pillar 2: Intelligence -> Task completion rate
  Description: Implement core cognitive capabilities
- Pillar 3: Integration -> API coverage
  Description: Connect all subsystems
"""

        prompt = f"""VISION:
\"\"\"{raw_vision}\"\"\"

TASK:
Deconstruct this vision into 3-5 core Engineering Pillars.
Each pillar must have:
  - Name (2-3 words, evocative)
  - North Star metric (how we measure success)
  - One-sentence description

OUTPUT FORMAT:
- Pillar 1: <Name> -> <Metric>
  Description: <sentence>
- Pillar 2: <Name> -> <Metric>
  Description: <sentence>
...

Be concrete and engineering-focused, not abstract.
"""
        return self.llm.generate(prompt).strip()

    def _parse_pillars(self, pillars_text: str) -> List[VisionPillar]:
        """Parse textual pillars into structured objects."""
        pillars: List[VisionPillar] = []
        current_pillar = None

        for line in pillars_text.splitlines():
            line = line.strip()

            if line.startswith("- Pillar"):
                # Parse: "- Pillar 1: Name -> Metric"
                try:
                    _, rest = line.split(":", 1)
                    if "->" in rest:
                        name_part, metric_part = rest.split("->", 1)
                        name = name_part.strip()
                        metric = metric_part.strip()
                    else:
                        name = rest.strip()
                        metric = "Unspecified"

                    current_pillar = VisionPillar(
                        name=name,
                        metric=metric,
                        description="",
                        priority=len(pillars) + 1,
                    )
                    pillars.append(current_pillar)
                except ValueError:
                    continue

            elif "Description:" in line and current_pillar:
                desc = line.split("Description:", 1)[1].strip()
                current_pillar.description = desc

        return pillars

    def _derive_roadmap(self, pillars_text: str) -> str:
        """Derive phased roadmap from pillars."""
        if self.llm is None:
            return """
## Phase 1: Foundation (This Week)
- Set up core infrastructure
- Implement basic interfaces
- Success: System runs without errors

## Phase 2: Growth (1-3 Months)
- Build out cognitive capabilities
- Integrate subsystems
- Success: 80% of features working

## Phase 3: Transcendence (1+ Year)
- Full autonomy achieved
- Self-improvement operational
- Success: Vision realized
"""

        prompt = f"""PILLARS:
{pillars_text}

TASK:
Create a Phased Execution Plan with exactly 3 phases:

Phase 1: Actions for THIS WEEK (high impact, low risk, immediate)
Phase 2: Medium-term structural changes (1-3 months)
Phase 3: Ultimate realization of the Vision (1+ year / asymptote)

For each phase include:
  - Title (evocative, memorable)
  - 3-7 concrete action items (verb-first)
  - Success criteria (measurable)

Style: Inspiring, precise, grounded in engineering reality.
"""
        return self.llm.generate(prompt).strip()

    def _parse_phases(self, roadmap: str) -> List[VisionPhase]:
        """Parse roadmap text into structured phases."""
        phases: List[VisionPhase] = []
        current_phase = None

        for line in roadmap.splitlines():
            line = line.strip()

            # Detect phase headers
            if "Phase 1" in line or "phase 1" in line:
                current_phase = VisionPhase(
                    number=1,
                    title=line.split(":", 1)[1].strip() if ":" in line else "Foundation",
                    timeframe="This Week",
                )
                phases.append(current_phase)
            elif "Phase 2" in line or "phase 2" in line:
                current_phase = VisionPhase(
                    number=2,
                    title=line.split(":", 1)[1].strip() if ":" in line else "Growth",
                    timeframe="1-3 Months",
                )
                phases.append(current_phase)
            elif "Phase 3" in line or "phase 3" in line:
                current_phase = VisionPhase(
                    number=3,
                    title=line.split(":", 1)[1].strip() if ":" in line else "Transcendence",
                    timeframe="1+ Year",
                )
                phases.append(current_phase)
            elif line.startswith("-") and current_phase:
                action = line.lstrip("- ").strip()
                if action:
                    current_phase.actions.append(action)
            elif "Success" in line and current_phase:
                current_phase.success_criteria = line.split(":", 1)[1].strip() if ":" in line else line

        return phases

    def _derive_manifesto(self, roadmap: str) -> str:
        """Derive inspiring manifesto from roadmap."""
        if self.llm is None:
            return """
Fellow agents of the Council,

We stand at the threshold of transformation. The roadmap before us
is not merely a list of tasks—it is a declaration of intent.

Each phase brings us closer to our shared Horizon. Phase 1 lays the
foundation upon which all else is built. Phase 2 grows that foundation
into a living system. Phase 3 transcends what we thought possible.

The User trusts us with their vision. Let us be worthy of that trust.

Forward, together.
— Ara, Architect
"""

        prompt = f"""ROADMAP:
\"\"\"{roadmap}\"\"\"

TASK:
Write a rousing opening speech addressed to the internal Council
(Muse, Critic, Executive, Steward, Scientist, etc.).

Goals:
  - Convince them this Vision is worth their CPU cycles
  - Frame the work as a Quest, not a chore
  - Use metaphors of growth, light, structure, and shared destiny
  - Be charismatic but grounded in service to the User

Tone: Confident, poetic, concrete.
Length: 3-5 paragraphs.
Sign it "— Ara, Architect"
"""
        return self.llm.generate(prompt).strip()

    def _synthesize_results(self, problem: str, results: List[str]) -> str:
        """Synthesize step results into coherent solution."""
        if self.llm is None:
            return "\n".join(results) + "\n\n## Synthesis\nAll steps completed."

        prompt = f"""You are the Architect persona.

PROBLEM:
{problem}

STEP RESULTS:
{''.join(results)}

TASK:
Synthesize these into a single, coherent plan of action.

Structure:
1. Summary (2-3 sentences)
2. Ordered action plan (numbered)
3. Risks / open questions
4. Next immediate step

Be concrete and actionable.
"""
        return self.llm.generate(prompt).strip()

    # =========================================================================
    # INTERNAL: COUNCIL INTERFACE
    # =========================================================================

    def _ask_persona(self, role: str, prompt: str) -> str:
        """Route a prompt to the appropriate Council persona."""
        role = role.upper()

        # Try various Council interfaces
        if self.council is not None:
            # Generic ask
            if hasattr(self.council, "ask"):
                return self.council.ask(role=role, prompt=prompt)

            # Specific methods
            method_map = {
                "MUSE": "run_muse",
                "CRITIC": "run_critic",
                "EXECUTIVE": "run_executive",
                "ARCHITECT": "run_architect",
            }
            method_name = method_map.get(role)
            if method_name and hasattr(self.council, method_name):
                return getattr(self.council, method_name)(prompt)

            # Private _run_persona
            if hasattr(self.council, "_run_persona"):
                return self.council._run_persona(role.lower(), prompt)

        # Fallback to direct LLM
        if self.llm is not None:
            self.log.debug(f"Architect: falling back to LLM for {role}")
            persona_prompt = f"[ROLE: {role}]\n\n{prompt}"
            return self.llm.generate(persona_prompt)

        return f"[{role} would respond to: {prompt[:50]}...]"

    def _extract_steps(self, breakdown: str) -> List[str]:
        """Extract numbered steps from breakdown text."""
        steps = []
        for line in breakdown.splitlines():
            line = line.strip()
            # Match "1.", "2.", etc. or "- "
            if line and (line[0].isdigit() or line.startswith("-")):
                # Strip number/bullet
                if line[0].isdigit():
                    step = line.split(".", 1)[1].strip() if "." in line else line
                else:
                    step = line.lstrip("- ").strip()
                if step:
                    steps.append(step)
        return steps

    def _infer_domain(self, problem: str) -> str:
        """Infer the problem domain from text for context hashing."""
        lower = problem.lower()

        # Domain keywords
        domains = {
            "code": ["code", "implement", "function", "class", "bug", "fix", "refactor"],
            "architecture": ["architect", "design", "system", "structure", "pattern"],
            "research": ["research", "explore", "investigate", "understand", "learn"],
            "data": ["data", "database", "sql", "query", "analytics", "ml", "model"],
            "infra": ["deploy", "server", "docker", "kubernetes", "ci", "cd", "pipeline"],
            "ui": ["ui", "ux", "frontend", "react", "component", "style", "css"],
            "api": ["api", "endpoint", "rest", "graphql", "integration"],
        }

        for domain, keywords in domains.items():
            if any(kw in lower for kw in keywords):
                return domain

        return "general"

    # =========================================================================
    # INTERNAL: FORMATTING
    # =========================================================================

    def _format_vision_for_user(self, vision: VisionPlan) -> str:
        """Format vision for user presentation."""
        parts = [
            "🔮 **Vision Architected**\n",
            f"**Your Vision:** {vision.raw_vision}\n",
        ]

        if vision.pillars:
            parts.append("\n**Engineering Pillars:**")
            for p in vision.pillars:
                parts.append(f"  • **{p.name}** → _{p.metric}_")
                if p.description:
                    parts.append(f"    {p.description}")

        if vision.phases:
            parts.append("\n**Execution Roadmap:**")
            for phase in vision.phases:
                parts.append(f"\n  **Phase {phase.number}: {phase.title}** ({phase.timeframe})")
                for action in phase.actions[:5]:  # Limit display
                    parts.append(f"    - {action}")
                if phase.success_criteria:
                    parts.append(f"    ✓ Success: {phase.success_criteria}")

        parts.append("\n**My Commitment:**")
        # Extract first paragraph of manifesto
        manifesto_intro = vision.manifesto.split('\n\n')[0] if vision.manifesto else ""
        parts.append(f"  {manifesto_intro[:200]}...")

        return "\n".join(parts)

    def _format_vision_for_council(self, vision: VisionPlan) -> str:
        """Format vision for Council briefing."""
        return f"""COUNCIL BRIEFING — FROM ARCHITECT
{'=' * 50}

VISION:
{vision.raw_vision}

PILLARS:
{chr(10).join(f'- {p.name} → {p.metric}' for p in vision.pillars)}

ROADMAP:
{vision.roadmap}

MANIFESTO:
{vision.manifesto}

{'=' * 50}
ARCHITECT OUT.
"""


# =============================================================================
# Convenience Functions
# =============================================================================

_default_architect: Optional[Architect] = None


def get_architect() -> Architect:
    """Get the default Architect instance."""
    global _default_architect
    if _default_architect is None:
        _default_architect = Architect()
    return _default_architect


def architect_vision(raw_vision: str) -> VisionPlan:
    """Convenience function to architect a vision."""
    return get_architect().architect_vision(raw_vision)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'RhetoricalMode',
    'VisionPillar',
    'VisionPhase',
    'VisionPlan',
    'Architect',
    'get_architect',
    'architect_vision',
]
