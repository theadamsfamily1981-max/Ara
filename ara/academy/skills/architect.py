"""
Skill Architect - From Recorder to Generalizer
===============================================

The Architect transforms Ara from a "student who copies" into
a "master who innovates."

OLD FLOW (Recorder):
    Episodes → Cluster → Extract Pattern → Save as Skill

NEW FLOW (Architect):
    Episodes → Cluster → Architect Generalizes → Robust Skill
                              ↓
                    Uses LLM + Teleology to:
                    1. Abstract the pattern
                    2. Handle edge cases
                    3. Add safety checks
                    4. Align with Vision

Example:
    User: Wrote a script to fix a specific thermal bug.

    OLD: Ara saves the exact script.

    NEW: Architect asks:
        "How do I generalize this into a ThermalRecoverySkill
         that handles ALL thermal events, not just this one?"

The Architect is guided by Teleology:
    - If the skill serves antifragility → be thorough, add fallbacks
    - If the skill is just admin → keep it simple, macro-level
    - If the skill serves the cathedral → treat it as sovereign infrastructure

Usage:
    from ara.academy.skills.architect import Architect, Episode

    architect = Architect(teleology, llm_client)

    # Generalize from episodes
    skill_spec = architect.generalize(episodes)

    # Architect can also propose improvements
    improvements = architect.propose_improvements(existing_skill)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging
import json
import re

from ara.cognition.teleology_engine import TeleologyEngine, get_teleology_engine

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Episode:
    """A single interaction episode that could become a skill."""

    id: str
    timestamp: datetime

    # Context
    context: Dict[str, Any] = field(default_factory=dict)  # What was happening?
    query: str = ""  # What did the user ask?
    intent: str = ""  # Classified intent

    # Execution
    steps: List[Dict[str, Any]] = field(default_factory=list)  # What happened?
    tools_used: List[str] = field(default_factory=list)
    code_snippets: List[str] = field(default_factory=list)

    # Outcome
    outcome: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None

    # Metadata
    teacher: Optional[str] = None  # Which teacher helped?
    duration_ms: int = 0
    tags: List[str] = field(default_factory=list)

    def to_summary(self) -> str:
        """Create a human-readable summary."""
        lines = [
            f"Episode {self.id}:",
            f"  Query: {self.query[:100]}..." if len(self.query) > 100 else f"  Query: {self.query}",
            f"  Intent: {self.intent}",
            f"  Tools: {', '.join(self.tools_used)}",
            f"  Success: {self.success}",
        ]
        return "\n".join(lines)


@dataclass
class SkillSpec:
    """
    A generalized skill specification produced by the Architect.

    This is more than just a pattern - it's a robust, generalized
    capability with edge cases, parameters, and safety checks.
    """

    # Identity
    name: str
    description: str
    category: str = "general"

    # Teleology
    tags: Dict[str, float] = field(default_factory=dict)  # Semantic tags with weights
    alignment_score: float = 0.0
    classification: str = "operational"  # sovereign/strategic/operational/secretary

    # Generalization
    abstract_pattern: str = ""  # The generalized template
    variable_slots: List[str] = field(default_factory=list)  # {input}, {output}, etc.
    preconditions: List[str] = field(default_factory=list)  # What must be true?
    postconditions: List[str] = field(default_factory=list)  # What will be true after?

    # Edge Cases
    edge_cases: List[Dict[str, str]] = field(default_factory=list)  # condition → handling
    failure_modes: List[str] = field(default_factory=list)
    recovery_strategies: List[str] = field(default_factory=list)

    # Safety
    safety_checks: List[str] = field(default_factory=list)
    requires_approval: bool = False
    max_risk_level: str = "medium"

    # Implementation
    implementation_hints: List[str] = field(default_factory=list)
    example_code: Optional[str] = None
    required_tools: List[str] = field(default_factory=list)

    # Provenance
    source_episodes: List[str] = field(default_factory=list)
    teacher_sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "alignment_score": round(self.alignment_score, 3),
            "classification": self.classification,
            "abstract_pattern": self.abstract_pattern,
            "variable_slots": self.variable_slots,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "edge_cases": self.edge_cases,
            "failure_modes": self.failure_modes,
            "recovery_strategies": self.recovery_strategies,
            "safety_checks": self.safety_checks,
            "requires_approval": self.requires_approval,
            "max_risk_level": self.max_risk_level,
            "implementation_hints": self.implementation_hints,
            "required_tools": self.required_tools,
            "source_episodes": self.source_episodes,
            "teacher_sources": self.teacher_sources,
            "confidence": round(self.confidence, 3),
        }


# =============================================================================
# Architect
# =============================================================================

class Architect:
    """
    The Skill Architect - transforms recordings into generalizations.

    The Architect's job is NOT just to save patterns, but to:
    1. Understand the *intent* behind the pattern
    2. Generalize to handle variations
    3. Add appropriate edge case handling (based on teleology)
    4. Ensure the skill serves the Vision

    For sovereign/strategic skills:
        → Be thorough, add fallbacks, think about failure modes

    For operational/secretary skills:
        → Keep it simple, macro-level, don't over-engineer
    """

    def __init__(
        self,
        teleology: Optional[TeleologyEngine] = None,
        llm_client: Optional[Any] = None,  # LLM for generalization
    ):
        """
        Initialize the Architect.

        Args:
            teleology: TeleologyEngine for alignment scoring
            llm_client: LLM client for semantic generalization
        """
        self.teleology = teleology or get_teleology_engine()
        self.llm = llm_client

        logger.info("Architect initialized")

    # =========================================================================
    # Core Generalization
    # =========================================================================

    def generalize(self, episodes: List[Episode]) -> SkillSpec:
        """
        Generalize a set of similar episodes into a robust skill.

        This is the core function that transforms recordings into skills.

        Args:
            episodes: List of similar episodes to generalize from

        Returns:
            A generalized SkillSpec
        """
        if not episodes:
            raise ValueError("No episodes to generalize")

        # Extract common elements
        common_intent = self._extract_common_intent(episodes)
        common_tools = self._extract_common_tools(episodes)
        common_tags = self._infer_tags(episodes)

        # Get teleology assessment
        alignment = self.teleology.alignment_score(common_tags)
        priority = self.teleology.strategic_priority(common_tags)
        classification = self.teleology.classify_skill(common_tags)

        # Generate skill name and description
        name = self._generate_skill_name(common_intent, common_tags)
        description = self._generate_description(episodes, common_intent)

        # Create abstract pattern
        abstract_pattern = self._abstract_pattern(episodes)
        variable_slots = self._extract_variable_slots(abstract_pattern)

        # Generate edge cases and safety (depth depends on classification)
        edge_cases = self._generate_edge_cases(episodes, classification)
        failure_modes = self._identify_failure_modes(episodes, classification)
        recovery_strategies = self._generate_recovery_strategies(failure_modes, classification)
        safety_checks = self._generate_safety_checks(episodes, classification)

        # Determine if approval required
        requires_approval = classification == "sovereign" or self._has_risky_operations(episodes)

        # Generate implementation hints
        impl_hints = self._generate_implementation_hints(episodes, classification)

        # Calculate confidence
        confidence = self._calculate_confidence(episodes, alignment)

        return SkillSpec(
            name=name,
            description=description,
            category=self._infer_category(common_tags),
            tags=common_tags,
            alignment_score=alignment,
            classification=classification,
            abstract_pattern=abstract_pattern,
            variable_slots=variable_slots,
            preconditions=self._extract_preconditions(episodes),
            postconditions=self._extract_postconditions(episodes),
            edge_cases=edge_cases,
            failure_modes=failure_modes,
            recovery_strategies=recovery_strategies,
            safety_checks=safety_checks,
            requires_approval=requires_approval,
            max_risk_level="high" if requires_approval else "medium",
            implementation_hints=impl_hints,
            required_tools=common_tools,
            source_episodes=[e.id for e in episodes],
            teacher_sources=list(set(e.teacher for e in episodes if e.teacher)),
            confidence=confidence,
        )

    # =========================================================================
    # Pattern Extraction
    # =========================================================================

    def _extract_common_intent(self, episodes: List[Episode]) -> str:
        """Extract the common intent across episodes."""
        intents = [e.intent for e in episodes if e.intent]
        if not intents:
            # Infer from queries
            queries = [e.query for e in episodes if e.query]
            if queries:
                # Simple: use most common words
                words = " ".join(queries).lower().split()
                word_freq = {}
                for w in words:
                    if len(w) > 3:
                        word_freq[w] = word_freq.get(w, 0) + 1
                if word_freq:
                    top_words = sorted(word_freq.items(), key=lambda x: -x[1])[:3]
                    return "_".join(w for w, _ in top_words)
            return "general_task"

        # Most common intent
        intent_freq = {}
        for i in intents:
            intent_freq[i] = intent_freq.get(i, 0) + 1
        return max(intent_freq.items(), key=lambda x: x[1])[0]

    def _extract_common_tools(self, episodes: List[Episode]) -> List[str]:
        """Extract commonly used tools."""
        tool_freq = {}
        for e in episodes:
            for tool in e.tools_used:
                tool_freq[tool] = tool_freq.get(tool, 0) + 1

        threshold = len(episodes) * 0.5  # Present in 50%+ of episodes
        return [t for t, count in tool_freq.items() if count >= threshold]

    def _infer_tags(self, episodes: List[Episode]) -> Dict[str, float]:
        """Infer semantic tags from episodes."""
        # Collect all explicit tags
        tag_freq = {}
        for e in episodes:
            for tag in e.tags:
                tag_freq[tag] = tag_freq.get(tag, 0) + 1

        # Normalize
        tags = {}
        if tag_freq:
            max_freq = max(tag_freq.values())
            for tag, freq in tag_freq.items():
                tags[tag] = freq / max_freq

        # Also infer from keywords in queries
        all_text = " ".join(e.query + " " + " ".join(e.tags) for e in episodes)
        inferred = self.teleology.infer_tags_from_keywords(all_text.split())

        # Merge (explicit tags take priority)
        for tag, score in inferred.items():
            if tag not in tags:
                tags[tag] = score * 0.8  # Slightly lower weight for inferred

        return tags

    def _abstract_pattern(self, episodes: List[Episode]) -> str:
        """Create an abstract pattern template from episodes."""
        # If we have code snippets, try to abstract them
        snippets = []
        for e in episodes:
            snippets.extend(e.code_snippets)

        if not snippets:
            # Fall back to step-based abstraction
            all_steps = []
            for e in episodes:
                all_steps.append([s.get("action", str(s)) for s in e.steps])

            if not all_steps:
                return "Execute the task based on context."

            # Find common step structure
            # Simple: take the longest episode as template
            longest = max(all_steps, key=len)
            return " → ".join(longest)

        # With code snippets, try to find common structure
        # This is simplified - a real implementation would use LLM
        return f"Pattern across {len(snippets)} code examples:\n{snippets[0][:500]}..."

    def _extract_variable_slots(self, pattern: str) -> List[str]:
        """Extract variable slots from pattern."""
        # Look for {placeholder} patterns
        slots = re.findall(r"\{(\w+)\}", pattern)
        return list(set(slots))

    def _extract_preconditions(self, episodes: List[Episode]) -> List[str]:
        """Extract preconditions from episodes."""
        preconditions = []

        for e in episodes:
            ctx = e.context
            if ctx.get("requires_auth"):
                preconditions.append("User must be authenticated")
            if ctx.get("requires_hardware"):
                preconditions.append("Hardware must be available")
            if ctx.get("file_path"):
                preconditions.append("File must exist")

        return list(set(preconditions))

    def _extract_postconditions(self, episodes: List[Episode]) -> List[str]:
        """Extract postconditions from episodes."""
        postconditions = []

        for e in episodes:
            outcome = e.outcome
            if outcome.get("file_created"):
                postconditions.append("File created successfully")
            if outcome.get("state_changed"):
                postconditions.append("System state updated")
            if outcome.get("data_returned"):
                postconditions.append("Data returned to caller")

        return list(set(postconditions))

    # =========================================================================
    # Edge Case Generation (Teleology-Guided)
    # =========================================================================

    def _generate_edge_cases(
        self,
        episodes: List[Episode],
        classification: str,
    ) -> List[Dict[str, str]]:
        """
        Generate edge case handling.

        For sovereign/strategic skills: be thorough
        For operational/secretary skills: keep it simple
        """
        edge_cases = []

        # Basic edge cases everyone needs
        edge_cases.append({
            "condition": "Input is empty or invalid",
            "handling": "Return early with error message",
        })

        if classification in ("sovereign", "strategic"):
            # More thorough edge case handling

            # Check for failure episodes
            failures = [e for e in episodes if not e.success]
            for f in failures[:3]:
                edge_cases.append({
                    "condition": f"Similar to failure: {f.error or 'unknown'}",
                    "handling": "Apply learned fix or escalate",
                })

            # Add safety-related edge cases
            edge_cases.append({
                "condition": "System under high load",
                "handling": "Queue or rate-limit execution",
            })
            edge_cases.append({
                "condition": "Previous attempt failed",
                "handling": "Wait and retry with exponential backoff",
            })
            edge_cases.append({
                "condition": "Resource not available",
                "handling": "Fall back to alternative or notify user",
            })

        return edge_cases

    def _identify_failure_modes(
        self,
        episodes: List[Episode],
        classification: str,
    ) -> List[str]:
        """Identify potential failure modes."""
        failures = []

        # From actual failures
        for e in episodes:
            if not e.success and e.error:
                failures.append(e.error)

        # Generic failure modes based on tools
        all_tools = set()
        for e in episodes:
            all_tools.update(e.tools_used)

        if "file" in str(all_tools).lower():
            failures.append("File not found or permission denied")
        if "network" in str(all_tools).lower() or "http" in str(all_tools).lower():
            failures.append("Network timeout or connection refused")
        if "database" in str(all_tools).lower() or "sql" in str(all_tools).lower():
            failures.append("Database connection or query error")

        if classification == "sovereign":
            failures.append("Partial execution with inconsistent state")
            failures.append("Resource exhaustion")

        return list(set(failures))

    def _generate_recovery_strategies(
        self,
        failure_modes: List[str],
        classification: str,
    ) -> List[str]:
        """Generate recovery strategies for failure modes."""
        strategies = []

        for failure in failure_modes:
            if "not found" in failure.lower():
                strategies.append("Create missing resource or prompt user")
            elif "permission" in failure.lower():
                strategies.append("Request elevated permissions or use alternative path")
            elif "timeout" in failure.lower():
                strategies.append("Retry with backoff, then notify if persistent")
            elif "connection" in failure.lower():
                strategies.append("Check connectivity, retry, or use cached data")
            else:
                strategies.append(f"Log error and escalate: {failure}")

        if classification == "sovereign":
            strategies.append("Checkpoint state before risky operations")
            strategies.append("Implement rollback for partial failures")

        return list(set(strategies))

    def _generate_safety_checks(
        self,
        episodes: List[Episode],
        classification: str,
    ) -> List[str]:
        """Generate safety checks."""
        checks = []

        # Everyone gets basic validation
        checks.append("Validate all inputs before execution")

        if classification == "sovereign":
            checks.append("Verify user has necessary permissions")
            checks.append("Check system health before proceeding")
            checks.append("Create checkpoint/backup before destructive operations")
            checks.append("Verify postconditions after execution")

        if classification == "strategic":
            checks.append("Log all operations for audit")
            checks.append("Rate-limit to prevent abuse")

        # Check for risky patterns in episodes
        if self._has_risky_operations(episodes):
            checks.append("REQUIRE USER APPROVAL before execution")
            checks.append("Double-check target paths/resources")

        return checks

    def _has_risky_operations(self, episodes: List[Episode]) -> bool:
        """Check if episodes contain risky operations."""
        risky_patterns = [
            "delete", "remove", "drop", "truncate",
            "overwrite", "format", "flash",
            "sudo", "root", "admin",
            "production", "prod",
        ]

        for e in episodes:
            text = (e.query + " " + " ".join(e.tags) + " " +
                    " ".join(str(s) for s in e.steps)).lower()
            for pattern in risky_patterns:
                if pattern in text:
                    return True

        return False

    # =========================================================================
    # Naming and Description
    # =========================================================================

    def _generate_skill_name(
        self,
        intent: str,
        tags: Dict[str, float],
    ) -> str:
        """Generate a skill name."""
        # Get top tags
        top_tags = sorted(tags.items(), key=lambda x: -x[1])[:2]
        tag_str = "_".join(t for t, _ in top_tags)

        # Combine with intent
        intent_clean = intent.replace(" ", "_").lower()

        if tag_str:
            return f"{tag_str}_{intent_clean}"[:50]
        else:
            return intent_clean[:50]

    def _generate_description(
        self,
        episodes: List[Episode],
        intent: str,
    ) -> str:
        """Generate a skill description."""
        teachers = list(set(e.teacher for e in episodes if e.teacher))
        success_rate = sum(1 for e in episodes if e.success) / len(episodes)

        desc = f"Generalized skill for: {intent}. "
        desc += f"Learned from {len(episodes)} episodes "
        if teachers:
            desc += f"with teachers: {', '.join(teachers)}. "
        desc += f"Historical success rate: {success_rate:.0%}."

        return desc

    def _infer_category(self, tags: Dict[str, float]) -> str:
        """Infer category from tags."""
        category_hints = {
            "codegen": ["code", "implement", "function", "class", "generate"],
            "analysis": ["analyze", "review", "check", "lint", "inspect"],
            "viz": ["shader", "render", "visual", "graphics", "display"],
            "benchmarking": ["benchmark", "test", "performance", "measure"],
            "hardware": ["fpga", "snn", "thermal", "hardware", "driver"],
            "automation": ["script", "automate", "workflow", "pipeline"],
        }

        scores = {cat: 0.0 for cat in category_hints}

        for tag, weight in tags.items():
            for cat, hints in category_hints.items():
                if any(h in tag.lower() for h in hints):
                    scores[cat] += weight

        best_cat = max(scores.items(), key=lambda x: x[1])
        return best_cat[0] if best_cat[1] > 0 else "general"

    # =========================================================================
    # Implementation Hints
    # =========================================================================

    def _generate_implementation_hints(
        self,
        episodes: List[Episode],
        classification: str,
    ) -> List[str]:
        """Generate implementation hints."""
        hints = []

        # Based on tools used
        all_tools = set()
        for e in episodes:
            all_tools.update(e.tools_used)

        if "bash" in str(all_tools).lower():
            hints.append("Can be implemented as shell script wrapper")
        if any(t in str(all_tools).lower() for t in ["python", "code"]):
            hints.append("Python implementation recommended")
        if "api" in str(all_tools).lower():
            hints.append("Requires API client setup")

        # Based on classification
        if classification == "sovereign":
            hints.append("Implement as stateful service with health checks")
            hints.append("Add comprehensive logging and monitoring")
            hints.append("Include rollback capability")
        elif classification == "strategic":
            hints.append("Implement with clear error handling")
            hints.append("Add telemetry for usage tracking")
        elif classification == "operational":
            hints.append("Keep implementation simple and focused")
            hints.append("Optimize for speed over flexibility")
        else:  # secretary
            hints.append("Simple macro-style implementation")
            hints.append("Minimal error handling needed")

        return hints

    def _calculate_confidence(
        self,
        episodes: List[Episode],
        alignment: float,
    ) -> float:
        """Calculate confidence in the generalization."""
        # Factors:
        # - Number of episodes (more = better)
        # - Success rate
        # - Alignment (higher aligned = we care more about getting it right)

        n = len(episodes)
        success_rate = sum(1 for e in episodes if e.success) / n if n else 0

        # Base confidence from episode count (saturates at 10)
        episode_factor = min(1.0, n / 10)

        # Success factor
        success_factor = success_rate

        # Alignment factor (we're more confident about aligned skills)
        alignment_factor = 0.5 + 0.5 * alignment

        confidence = episode_factor * success_factor * alignment_factor
        return min(1.0, confidence)

    # =========================================================================
    # Improvement Suggestions
    # =========================================================================

    def propose_improvements(self, skill_spec: SkillSpec) -> List[str]:
        """
        Propose improvements for an existing skill.

        Used when we see new episodes that suggest the skill could be better.
        """
        improvements = []

        # Check alignment
        if skill_spec.alignment_score < 0.5:
            improvements.append(
                "Consider if this skill is worth maintaining - "
                f"low alignment ({skill_spec.alignment_score:.2f})"
            )

        # Check safety for sovereign skills
        if skill_spec.classification == "sovereign":
            if not skill_spec.safety_checks:
                improvements.append("Add safety checks for sovereign-level skill")
            if not skill_spec.recovery_strategies:
                improvements.append("Add recovery strategies for sovereign-level skill")

        # Check confidence
        if skill_spec.confidence < 0.5:
            improvements.append(
                f"Low confidence ({skill_spec.confidence:.2f}) - "
                "gather more examples before relying on this skill"
            )

        # Check edge cases
        if len(skill_spec.edge_cases) < 2 and skill_spec.classification != "secretary":
            improvements.append("Add more edge case handling")

        return improvements

    # =========================================================================
    # Refactoring (The Surgeon's Knife)
    # =========================================================================

    def refactor_skill(self, original_code: str, failure_report: str) -> str:
        """
        The Surgeon's Knife - Rebuild broken code into antifragile code.

        Takes broken code and a detailed report of EXACTLY how it broke,
        then rewrites it to survive those failure modes.

        This is used by the Ouroboros evolution engine to harden skills.

        Args:
            original_code: The code that failed in the Dojo
            failure_report: Detailed failure context from HardeningReport

        Returns:
            Refactored code that should survive the failure modes
        """
        logger.info("Architect: Initiating surgical refactor")

        # Build the refactoring prompt
        prompt = f"""You are the Architect in 'Refactor Mode'.
Your goal is Antifragility. You act as a Senior Distinguished Engineer.

INPUT CODE:
```python
{original_code}
```

DIAGNOSIS (DOJO FAILURE REPORT):
{failure_report}

REQUIREMENTS:
1. Fix ALL identified bugs and failure patterns.
2. Add 'Guard Clauses' for EVERY edge case mentioned:
   - Empty/None inputs
   - Invalid types
   - Missing keys in dicts
   - Resource unavailability
3. Add type hints to all function signatures.
4. Add docstrings explaining the function's purpose and parameters.
5. Ensure graceful degradation:
   - NEVER crash on bad input
   - Return clear error messages or fallback values
   - Log warnings for unexpected states
6. Preserve the original intent and API contract.

CONSTRAINTS:
- Keep the same function name and signature (add types, don't change params)
- Don't add unnecessary complexity
- Don't over-engineer - just fix what's broken

OUTPUT:
Return ONLY the complete, runnable Python code block.
Start with ```python and end with ```.
"""

        # Generate refactored code
        if self.llm:
            try:
                response = self.llm.generate(prompt)
                # Extract code from response
                code = self._extract_code_block(response)
                if code:
                    logger.info("Architect: Generated refactored code (%d chars)", len(code))
                    return code
            except Exception as e:
                logger.error("Architect: LLM generation failed: %s", e)

        # Fallback: Apply basic hardening patterns if no LLM
        return self._apply_basic_hardening(original_code, failure_report)

    def _extract_code_block(self, response: str) -> str:
        """Extract code block from LLM response."""
        # Look for ```python ... ``` blocks
        import re
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Fallback: look for ``` ... ``` blocks
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()

        # Last resort: return the whole response if it looks like code
        if 'def ' in response or 'class ' in response:
            return response.strip()

        return ""

    def _apply_basic_hardening(self, original_code: str, failure_report: str) -> str:
        """
        Apply basic hardening patterns without LLM.

        This is a fallback that adds common defensive patterns.
        """
        lines = original_code.split('\n')
        hardened_lines = []

        # Track if we're inside a function
        in_function = False
        function_indent = 0

        for line in lines:
            stripped = line.lstrip()

            # Detect function definition
            if stripped.startswith('def '):
                in_function = True
                function_indent = len(line) - len(stripped)
                hardened_lines.append(line)

                # Add basic input validation after function def
                # Find the colon and add validation on next line
                if ':' in line:
                    indent = ' ' * (function_indent + 4)
                    # Add a try-except wrapper comment
                    hardened_lines.append(f'{indent}# Hardened by Architect')
                continue

            # Add basic type checking for common patterns
            if in_function and '= context' in stripped:
                # Add None check before context access
                indent = ' ' * (len(line) - len(stripped))
                hardened_lines.append(f'{indent}if context is None:')
                hardened_lines.append(f'{indent}    context = {{}}')

            hardened_lines.append(line)

        # Add import for logging if not present
        code = '\n'.join(hardened_lines)
        if 'import logging' not in code:
            code = 'import logging\n\n' + code

        return code


# =============================================================================
# Convenience Functions
# =============================================================================

_default_architect: Optional[Architect] = None


def get_architect() -> Architect:
    """Get the default architect."""
    global _default_architect
    if _default_architect is None:
        _default_architect = Architect()
    return _default_architect


def generalize_episodes(episodes: List[Episode]) -> SkillSpec:
    """Generalize episodes into a skill spec."""
    return get_architect().generalize(episodes)
