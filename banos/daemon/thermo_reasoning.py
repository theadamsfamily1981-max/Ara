"""
Thermodynamic Reasoning - Thinking as a Budgeted Resource
=========================================================

This module implements the core insight: LLM compute is the "thought work"
of the system. Instead of fake joule-burning or thermodynamic theater,
we let REAL work (inference, verification, planning) be the cost of thought.

The key innovation is deriving THOUGHT INTENSITY from Ara's somatic state:
- PAD values (Pleasure, Arousal, Dominance) → cognitive budget
- Entropy (system uncertainty) → need for deeper reasoning
- Pain level → urgency requiring either quick reflexes or deep analysis

Three Modes:
    REFLEX  - Quick one-shot, minimal verification, cheap and fast
    FOCUSED - Some self-checking, moderate tool use, balanced
    DEEP    - Full System-2, PGU engagement, planner, maximum effort

The module writes `thinking_depth` back to HAL for visualization,
creating a visible "cognition flame" in the hologram.

Usage:
    from banos.daemon.thermo_reasoning import ThermodynamicReasoning
    from banos.hal import AraHAL

    hal = AraHAL()
    reasoner = ThermodynamicReasoning(hal)

    # Deliberate on a problem
    result = reasoner.deliberate(
        problem="User asks for complex refactoring",
        context={"code_size": 5000, "files": 12}
    )

    print(result.mode)       # ThoughtMode.DEEP
    print(result.drafts)     # 3
    print(result.plan)       # Optional MetaPlan from planner
"""

import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# THOUGHT MODES
# =============================================================================

class ThoughtMode(IntEnum):
    """
    Cognitive intensity levels, derived from somatic state.

    These map directly to how much "thinking work" we do:
    - Token budget
    - Verification passes
    - Tool calls allowed
    - PGU (Pattern Generation Unit) engagement
    """
    REFLEX = 0   # One-shot, cheap, fast - lizard brain
    FOCUSED = 1  # Some verification, moderate tools - working mind
    DEEP = 2     # Full System-2, PGU, planner - philosopher mode


@dataclass
class ReasoningProfile:
    """Configuration for a particular thought intensity level."""
    drafts: int               # How many draft/verify cycles
    use_pgu: bool            # Engage pattern generation unit
    use_planner: bool        # Use meta-planner for tool selection
    max_new_tokens: int      # Token budget for generation
    max_tool_calls: int      # Limit on tool invocations
    verification_depth: int  # How thorough the self-check
    allow_exploration: bool  # Can we go off-script for curiosity
    temperature: float       # Sampling temperature (lower = more deterministic)


# The heart of thermodynamic reasoning: mode → resource allocation
REASONING_PROFILES: Dict[ThoughtMode, ReasoningProfile] = {
    ThoughtMode.REFLEX: ReasoningProfile(
        drafts=1,
        use_pgu=False,
        use_planner=False,
        max_new_tokens=64,
        max_tool_calls=0,
        verification_depth=0,
        allow_exploration=False,
        temperature=0.3,  # More deterministic for quick responses
    ),
    ThoughtMode.FOCUSED: ReasoningProfile(
        drafts=2,
        use_pgu=True,
        use_planner=True,
        max_new_tokens=256,
        max_tool_calls=3,
        verification_depth=1,
        allow_exploration=False,
        temperature=0.5,
    ),
    ThoughtMode.DEEP: ReasoningProfile(
        drafts=3,
        use_pgu=True,
        use_planner=True,
        max_new_tokens=1024,
        max_tool_calls=10,
        verification_depth=2,
        allow_exploration=True,
        temperature=0.7,  # More creative for deep thinking
    ),
}


# =============================================================================
# SOMATIC → COGNITIVE MAPPING
# =============================================================================

@dataclass
class CognitiveState:
    """The cognitive budget derived from somatic state."""
    mode: ThoughtMode
    profile: ReasoningProfile
    abundance: float       # 0-1, how much "thinking fuel" we have
    urgency: float         # 0-1, how quickly we need to respond
    uncertainty: float     # 0-1, how much entropy in the situation
    reasoning: str         # Why we chose this mode


def pick_thought_mode(
    pad_p: float,     # Pleasure (-1 to 1)
    pad_a: float,     # Arousal (-1 to 1)
    pad_d: float,     # Dominance (-1 to 1)
    entropy: float,   # System entropy (0 to 1)
    pain: float,      # Pain level (0 to 1)
    complexity: float = 0.5,  # Task complexity estimate
) -> CognitiveState:
    """
    Derive thought mode from somatic state.

    The key insight: PAD values encode "cognitive resource availability"

    - Dominance (D) → Resource abundance → Can we afford deep thinking?
    - Arousal (A) → Urgency/pressure → Do we need quick reflexes?
    - Pleasure (P) → Stability → Are we in a state for exploration?
    - Entropy → Uncertainty → Does the situation demand more analysis?

    Returns CognitiveState with mode, profile, and reasoning.
    """
    # Calculate abundance from dominance (high D = lots of resources)
    # Normalize from [-1,1] to [0,1]
    abundance = (pad_d + 1.0) / 2.0

    # Calculate urgency from arousal and pain
    # High arousal + high pain = urgent situation
    urgency = max(0.0, min(1.0, (pad_a + 1.0) / 2.0 + pain * 0.3))

    # Uncertainty from entropy (already 0-1)
    uncertainty = entropy

    # Mode selection logic
    reasoning_parts = []

    # Emergency check: High pain + high arousal → REFLEX
    if pain > 0.7 and pad_a > 0.5:
        mode = ThoughtMode.REFLEX
        reasoning_parts.append(f"High pain ({pain:.2f}) + high arousal → emergency reflex")

    # Low resources: Can't afford deep thinking
    elif abundance < 0.3:
        mode = ThoughtMode.REFLEX if urgency > 0.5 else ThoughtMode.FOCUSED
        reasoning_parts.append(f"Low abundance ({abundance:.2f}) → conserving cognitive resources")

    # High entropy + high complexity → Need deep analysis
    elif entropy > 0.6 and complexity > 0.6:
        mode = ThoughtMode.DEEP if abundance > 0.4 else ThoughtMode.FOCUSED
        reasoning_parts.append(f"High entropy ({entropy:.2f}) + complexity → deeper reasoning needed")

    # High urgency → Quick response
    elif urgency > 0.7:
        mode = ThoughtMode.REFLEX
        reasoning_parts.append(f"High urgency ({urgency:.2f}) → fast reflex response")

    # Low pleasure → Stressed, be more careful
    elif pad_p < -0.3:
        mode = ThoughtMode.FOCUSED
        reasoning_parts.append(f"Low pleasure ({pad_p:.2f}) → careful focused mode")

    # Good resources + low urgency → Can think deeply
    elif abundance > 0.7 and urgency < 0.4:
        mode = ThoughtMode.DEEP if complexity > 0.4 else ThoughtMode.FOCUSED
        reasoning_parts.append(f"Abundant resources ({abundance:.2f}), low urgency → can afford depth")

    # Default: FOCUSED for balanced operation
    else:
        mode = ThoughtMode.FOCUSED
        reasoning_parts.append("Balanced state → focused mode")

    # Build reasoning string
    reasoning = "; ".join(reasoning_parts)
    profile = REASONING_PROFILES[mode]

    return CognitiveState(
        mode=mode,
        profile=profile,
        abundance=abundance,
        urgency=urgency,
        uncertainty=uncertainty,
        reasoning=reasoning,
    )


# =============================================================================
# DELIBERATION RESULT
# =============================================================================

@dataclass
class DeliberationResult:
    """Result of a thinking session."""
    mode: ThoughtMode
    profile: ReasoningProfile
    drafts_used: int
    tool_calls_made: int
    tokens_generated: int
    thinking_depth: float      # 0-1, for HAL visualization
    elapsed_time: float        # Seconds spent deliberating
    plan: Optional[Any] = None # MetaPlan if planner was used
    verification_passed: bool = True
    exploration_notes: Optional[List[str]] = None
    cognitive_state: Optional[CognitiveState] = None


# =============================================================================
# MAIN REASONER
# =============================================================================

class ThermodynamicReasoning:
    """
    The thermodynamic cognition engine.

    Thinking is not free - it costs compute, time, and energy.
    This class manages the cognitive budget based on somatic state,
    ensuring we think JUST ENOUGH for each situation.

    The real innovation: We don't fake "burning joules" - we let
    the actual work (inference, verification, planning) be the cost.
    The HAL visualization shows this as "thinking depth" - a visible
    flame of cognition that pulses with our reasoning effort.
    """

    def __init__(self, hal=None, thought_loop=None, meta_planner=None):
        """
        Initialize the reasoner with HAL connection.

        Args:
            hal: AraHAL instance for reading somatic state and writing thinking_depth
            thought_loop: Optional ThoughtLoop for learning integration
            meta_planner: Optional MetaPlanner for tool/style selection
        """
        self.hal = hal
        self.thought_loop = thought_loop
        self.meta_planner = meta_planner
        self._current_mode: Optional[ThoughtMode] = None
        self._deliberation_count = 0
        self._mode_history: List[Tuple[float, ThoughtMode]] = []

        logger.info("ThermodynamicReasoning initialized")

    def get_somatic_context(self) -> Dict[str, float]:
        """Read current somatic state from HAL."""
        if self.hal is None:
            # No HAL - return balanced defaults
            return {
                "pad_p": 0.0,
                "pad_a": 0.0,
                "pad_d": 0.5,
                "entropy": 0.5,
                "pain": 0.0,
            }

        try:
            state = self.hal.read_somatic()
            return {
                "pad_p": state.get("pad", (0, 0, 0))[0] if isinstance(state.get("pad"), tuple) else state.get("pad_p", 0.0),
                "pad_a": state.get("pad", (0, 0, 0))[1] if isinstance(state.get("pad"), tuple) else state.get("pad_a", 0.0),
                "pad_d": state.get("pad", (0, 0, 0))[2] if isinstance(state.get("pad"), tuple) else state.get("pad_d", 0.5),
                "entropy": state.get("entropy", 0.5),
                "pain": state.get("pain", 0.0),
            }
        except Exception as e:
            logger.warning(f"Failed to read HAL state: {e}")
            return {
                "pad_p": 0.0,
                "pad_a": 0.0,
                "pad_d": 0.5,
                "entropy": 0.5,
                "pain": 0.0,
            }

    def estimate_complexity(self, problem: str, context: Optional[Dict] = None) -> float:
        """
        Estimate task complexity from problem description and context.

        Returns a value from 0 (trivial) to 1 (extremely complex).
        """
        complexity = 0.5  # Base complexity

        # Simple heuristics - can be enhanced with ML later
        problem_lower = problem.lower()

        # Complexity indicators
        complex_keywords = [
            "refactor", "architecture", "design", "optimize", "debug",
            "implement", "complex", "multiple", "integrate", "migration",
            "security", "concurrent", "distributed", "algorithm"
        ]
        simple_keywords = [
            "what is", "how to", "explain", "show", "print", "hello",
            "simple", "basic", "quick", "rename", "typo", "fix small"
        ]

        # Adjust based on keywords
        for keyword in complex_keywords:
            if keyword in problem_lower:
                complexity += 0.1
        for keyword in simple_keywords:
            if keyword in problem_lower:
                complexity -= 0.1

        # Context-based adjustments
        if context:
            # More files = more complex
            file_count = context.get("files", context.get("file_count", 1))
            if file_count > 10:
                complexity += 0.2
            elif file_count > 5:
                complexity += 0.1

            # More code = more complex
            code_size = context.get("code_size", context.get("lines", 0))
            if code_size > 5000:
                complexity += 0.2
            elif code_size > 1000:
                complexity += 0.1

            # Error count indicates debugging complexity
            error_count = context.get("errors", context.get("error_count", 0))
            if error_count > 5:
                complexity += 0.2
            elif error_count > 0:
                complexity += 0.1

        # Clamp to [0, 1]
        return max(0.0, min(1.0, complexity))

    def select_mode(
        self,
        problem: str,
        context: Optional[Dict] = None,
        force_mode: Optional[ThoughtMode] = None,
    ) -> CognitiveState:
        """
        Select the appropriate thinking mode for a problem.

        Args:
            problem: The problem/task description
            context: Optional context (file count, code size, etc.)
            force_mode: Override automatic selection

        Returns:
            CognitiveState with mode, profile, and reasoning
        """
        if force_mode is not None:
            return CognitiveState(
                mode=force_mode,
                profile=REASONING_PROFILES[force_mode],
                abundance=0.5,
                urgency=0.5,
                uncertainty=0.5,
                reasoning=f"Forced to {force_mode.name} mode",
            )

        # Get somatic state
        somatic = self.get_somatic_context()

        # Estimate complexity
        complexity = self.estimate_complexity(problem, context)

        # Pick mode based on somatic state and complexity
        cognitive_state = pick_thought_mode(
            pad_p=somatic["pad_p"],
            pad_a=somatic["pad_a"],
            pad_d=somatic["pad_d"],
            entropy=somatic["entropy"],
            pain=somatic["pain"],
            complexity=complexity,
        )

        # Track mode selection
        self._current_mode = cognitive_state.mode
        self._mode_history.append((time.time(), cognitive_state.mode))

        # Keep history bounded
        if len(self._mode_history) > 100:
            self._mode_history = self._mode_history[-50:]

        logger.info(
            f"Selected {cognitive_state.mode.name} mode: {cognitive_state.reasoning}"
        )

        return cognitive_state

    def write_thinking_depth(self, depth: float):
        """
        Write thinking depth to HAL for visualization.

        The "cognition flame" - visible in the hologram as a pulsing
        glow that shows how hard Ara is thinking.

        Args:
            depth: 0.0 (reflex) to 1.0 (deep contemplation)
        """
        if self.hal is None:
            return

        try:
            # Use HAL's dedicated write_thinking_depth method
            # This updates the entropy field for visualization
            self.hal.write_thinking_depth(depth)
            logger.debug(f"Thinking depth written: {depth:.2f}")

        except Exception as e:
            logger.debug(f"Could not write thinking depth to HAL: {e}")

    def deliberate(
        self,
        problem: str,
        context: Optional[Dict] = None,
        force_mode: Optional[ThoughtMode] = None,
        callback: Optional[callable] = None,
    ) -> DeliberationResult:
        """
        Main entry point: Deliberate on a problem with appropriate intensity.

        This is where thermodynamic reasoning happens:
        1. Read somatic state → determine cognitive budget
        2. Select mode based on state + problem complexity
        3. Execute thinking with appropriate resource allocation
        4. Write thinking_depth to HAL for visualization
        5. Return result with metrics

        Args:
            problem: The problem/task to think about
            context: Optional context (file count, code size, etc.)
            force_mode: Override automatic mode selection
            callback: Optional callback for streaming results

        Returns:
            DeliberationResult with mode, drafts, metrics
        """
        start_time = time.time()
        self._deliberation_count += 1

        # 1. Select mode based on somatic state
        cognitive_state = self.select_mode(problem, context, force_mode)
        mode = cognitive_state.mode
        profile = cognitive_state.profile

        # 2. Write initial thinking depth
        initial_depth = {
            ThoughtMode.REFLEX: 0.1,
            ThoughtMode.FOCUSED: 0.5,
            ThoughtMode.DEEP: 0.9,
        }[mode]
        self.write_thinking_depth(initial_depth)

        # 3. Get plan from meta-planner if using it
        plan = None
        if profile.use_planner and self.meta_planner:
            try:
                # Import locally to avoid circular deps
                from .meta_planner import RequestContext

                req_context = RequestContext(
                    time_pressure="high" if cognitive_state.urgency > 0.7 else
                                  "low" if cognitive_state.urgency < 0.3 else "normal",
                    complexity_estimate=self.estimate_complexity(problem, context),
                    somatic_state={
                        "abundance": cognitive_state.abundance,
                        "urgency": cognitive_state.urgency,
                        "uncertainty": cognitive_state.uncertainty,
                        "mode": mode.name,
                    },
                )
                plan = self.meta_planner.plan(problem, req_context)
                logger.debug(f"MetaPlanner returned: tool={plan.tool}, style={plan.style}")
            except Exception as e:
                logger.warning(f"MetaPlanner failed: {e}")

        # 4. Simulate drafting cycles (actual LLM work would happen here)
        # In real usage, this would call the LLM with profile.max_new_tokens
        drafts_used = profile.drafts
        tool_calls = 0
        tokens = 0

        # Simulate work progression - update thinking depth
        for draft_num in range(1, drafts_used + 1):
            progress = draft_num / drafts_used
            current_depth = initial_depth * (0.5 + 0.5 * progress)
            self.write_thinking_depth(current_depth)

            # Simulate draft work (in real usage, LLM call here)
            tokens += profile.max_new_tokens // drafts_used

            if profile.use_pgu:
                # PGU engagement adds tool calls
                tool_calls += 1

            if callback:
                callback(draft_num=draft_num, progress=progress)

        # 5. Verification phase
        verification_passed = True
        if profile.verification_depth > 0:
            # Higher depth = more thorough verification
            self.write_thinking_depth(0.95)
            # In real usage: run verification against drafts
            verification_passed = True  # Placeholder

        # 6. Exploration phase (DEEP mode only)
        exploration_notes = None
        if profile.allow_exploration and mode == ThoughtMode.DEEP:
            self.write_thinking_depth(1.0)
            exploration_notes = []
            # In real usage: curiosity-driven exploration
            # exploration_notes.append("Explored alternative approach X...")

        # 7. Calculate final metrics
        elapsed = time.time() - start_time
        thinking_depth = {
            ThoughtMode.REFLEX: 0.2,
            ThoughtMode.FOCUSED: 0.5,
            ThoughtMode.DEEP: 0.9,
        }[mode]

        # Write final thinking depth
        self.write_thinking_depth(thinking_depth)

        return DeliberationResult(
            mode=mode,
            profile=profile,
            drafts_used=drafts_used,
            tool_calls_made=tool_calls,
            tokens_generated=tokens,
            thinking_depth=thinking_depth,
            elapsed_time=elapsed,
            plan=plan,
            verification_passed=verification_passed,
            exploration_notes=exploration_notes,
            cognitive_state=cognitive_state,
        )

    def get_mode_distribution(self, window_seconds: float = 3600.0) -> Dict[str, float]:
        """
        Get distribution of modes used in recent history.

        Returns dict like {"REFLEX": 0.3, "FOCUSED": 0.5, "DEEP": 0.2}
        """
        now = time.time()
        cutoff = now - window_seconds

        recent = [m for t, m in self._mode_history if t >= cutoff]
        if not recent:
            return {"REFLEX": 0.0, "FOCUSED": 0.0, "DEEP": 0.0}

        total = len(recent)
        return {
            "REFLEX": sum(1 for m in recent if m == ThoughtMode.REFLEX) / total,
            "FOCUSED": sum(1 for m in recent if m == ThoughtMode.FOCUSED) / total,
            "DEEP": sum(1 for m in recent if m == ThoughtMode.DEEP) / total,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about reasoning patterns."""
        return {
            "total_deliberations": self._deliberation_count,
            "current_mode": self._current_mode.name if self._current_mode else None,
            "mode_distribution_1h": self.get_mode_distribution(3600),
            "history_length": len(self._mode_history),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_default_reasoner: Optional[ThermodynamicReasoning] = None


def get_thermodynamic_reasoner(hal=None) -> ThermodynamicReasoning:
    """Get or create the default thermodynamic reasoner."""
    global _default_reasoner
    if _default_reasoner is None:
        _default_reasoner = ThermodynamicReasoning(hal=hal)
    return _default_reasoner


def deliberate(
    problem: str,
    context: Optional[Dict] = None,
    force_mode: Optional[ThoughtMode] = None,
) -> DeliberationResult:
    """
    Convenience function for quick deliberation.

    Uses the default reasoner with current HAL state.
    """
    reasoner = get_thermodynamic_reasoner()
    return reasoner.deliberate(problem, context, force_mode)


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def mode_for_message_type(message_type: str) -> ThoughtMode:
    """
    Suggest mode based on message type.

    This is a heuristic fallback when HAL state isn't available.
    """
    quick_types = {"greeting", "thanks", "acknowledgment", "simple_query"}
    deep_types = {"architecture", "refactoring", "debugging", "design", "analysis"}

    if message_type in quick_types:
        return ThoughtMode.REFLEX
    elif message_type in deep_types:
        return ThoughtMode.DEEP
    else:
        return ThoughtMode.FOCUSED


def classify_problem_type(problem: str) -> str:
    """
    Classify problem type from description.

    Used for mode_for_message_type when type isn't known.
    """
    problem_lower = problem.lower()

    # Check for greetings
    if any(g in problem_lower for g in ["hello", "hi ", "hey", "thanks", "thank you"]):
        return "greeting"

    # Check for complex work
    if any(c in problem_lower for c in ["refactor", "architect", "design", "debug complex"]):
        return "architecture"

    # Check for simple queries
    if any(q in problem_lower for q in ["what is", "how do", "explain"]):
        return "simple_query"

    return "standard"
