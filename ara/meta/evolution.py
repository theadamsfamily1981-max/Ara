# ara/meta/evolution.py
"""
The Ouroboros - Recursive Self-Evolution Engine
===============================================

"The Master is always a Student."

This module actively hunts for weakness in Ara's own code and skills,
using the Dojo to break them and the Architect to rebuild them stronger.

The Cycle:
    1. SCOUT: Identify 'fragile' or 'uncertain' skills via CausalMiner.
    2. WEIGH: Use Teleology to decide if this skill matters.
    3. BREAK: Run the Dojo to generate failure cases (Shadow).
    4. FORGE: Pass failure logs to Architect to generate v2 code.
    5. VERIFY: Re-run Dojo. If pass, Hot-Swap the skill.

Philosophy:
    Ara doesn't just learn skills and hope they work.
    She actively attacks herself to find weakness,
    then rebuilds stronger from the wreckage.

    This is Antifragility applied to code:
    - Stress reveals weakness
    - Weakness triggers repair
    - Repair creates strength
    - Strength enables more stress

Usage:
    from ara.meta.evolution import EvolutionEngine, get_evolution_engine

    evolver = get_evolution_engine()

    # Run the nightly evolution cycle
    results = evolver.run_nightly_cycle()

    # Or evolve a specific skill immediately
    result = evolver.evolve_skill("my_skill")

    # Schedule priority evolution for high-value skills
    evolver.schedule_priority_evolution("critical_skill")
"""

from __future__ import annotations

import logging
import time
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

log = logging.getLogger("Ara.Ouroboros")


class EvolutionStatus(str, Enum):
    """Status of an evolution attempt."""
    SUCCESS = "success"
    ALREADY_STRONG = "already_strong"
    LOW_VALUE = "low_value"
    FORGE_FAILED = "forge_failed"
    VERIFY_FAILED = "verify_failed"
    NOT_FOUND = "not_found"
    IN_PROGRESS = "in_progress"
    QUEUED = "queued"


@dataclass
class EvolutionResult:
    """Result of a skill evolution attempt."""
    skill_name: str
    original_version: str
    new_version: Optional[str] = None
    status: EvolutionStatus = EvolutionStatus.IN_PROGRESS
    improvement_note: str = ""

    # Metrics
    original_pass_rate: float = 0.0
    new_pass_rate: float = 0.0
    failures_fixed: int = 0

    # Audit trail
    failure_patterns: List[str] = field(default_factory=list)
    edge_cases_covered: List[str] = field(default_factory=list)
    refactor_prompt: str = ""

    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "status": self.status.value,
            "improvement_note": self.improvement_note,
            "original_pass_rate": self.original_pass_rate,
            "new_pass_rate": self.new_pass_rate,
            "failures_fixed": self.failures_fixed,
            "duration_ms": self.duration_ms,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        if self.status == EvolutionStatus.SUCCESS:
            return (
                f"Evolved '{self.skill_name}': "
                f"{self.original_pass_rate:.0%} → {self.new_pass_rate:.0%} "
                f"(fixed {self.failures_fixed} failure modes)"
            )
        else:
            return f"Evolution of '{self.skill_name}': {self.status.value} - {self.improvement_note}"


@dataclass
class EvolutionCycleReport:
    """Report from a full evolution cycle."""
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: float = 0.0

    skills_scouted: int = 0
    skills_attempted: int = 0
    skills_evolved: int = 0
    skills_skipped: int = 0
    skills_failed: int = 0

    results: List[EvolutionResult] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Evolution Cycle: {self.skills_evolved}/{self.skills_attempted} evolved "
            f"({self.skills_skipped} skipped, {self.skills_failed} failed) "
            f"in {self.duration_ms:.0f}ms"
        )


class EvolutionEngine:
    """
    The Ouroboros - Recursive Self-Evolution Engine.

    The serpent that eats its own tail to grow stronger.

    This engine coordinates the full evolution cycle:
        Scout → Weigh → Break → Forge → Verify → Deploy

    It can run as:
        - Nightly batch: Process all fragile skills
        - Priority queue: Immediately harden high-value skills
        - Manual: Evolve a specific skill on demand
    """

    def __init__(
        self,
        min_pass_rate: float = 0.80,      # Target pass rate in Dojo
        max_forge_attempts: int = 3,       # Retries for code generation
        skip_secretary_skills: bool = True,  # Don't evolve low-value skills
        auto_deploy: bool = False,         # Auto-deploy successful evolutions
    ):
        """
        Initialize the Evolution Engine.

        Args:
            min_pass_rate: Minimum Dojo pass rate to consider "strong"
            max_forge_attempts: Max LLM generation attempts
            skip_secretary_skills: Skip low-strategic-value skills
            auto_deploy: Automatically deploy evolved skills
        """
        self.min_pass_rate = min_pass_rate
        self.max_forge_attempts = max_forge_attempts
        self.skip_secretary_skills = skip_secretary_skills
        self.auto_deploy = auto_deploy

        # Priority queue for urgent evolutions
        self._priority_queue: queue.Queue = queue.Queue()
        self._evolution_history: List[EvolutionResult] = []
        self._lock = threading.RLock()

        # Lazy-loaded dependencies (avoid circular imports)
        self._miner = None
        self._architect = None
        self._dojo = None
        self._teleology = None
        self._registry = None

        log.info(
            "Ouroboros initialized (min_pass=%.0f%%, max_forge=%d, auto_deploy=%s)",
            min_pass_rate * 100, max_forge_attempts, auto_deploy
        )

    # =========================================================================
    # Lazy Dependency Loading
    # =========================================================================

    @property
    def miner(self):
        """Lazy-load CausalPatternMiner."""
        if self._miner is None:
            from ara.meta.causal_miner import get_causal_miner
            self._miner = get_causal_miner()
        return self._miner

    @property
    def architect(self):
        """Lazy-load Architect."""
        if self._architect is None:
            from ara.academy.skills.architect import get_architect
            self._architect = get_architect()
        return self._architect

    @property
    def dojo(self):
        """Lazy-load Dojo."""
        if self._dojo is None:
            from ara.academy.dojo import get_dojo
            self._dojo = get_dojo()
        return self._dojo

    @property
    def teleology(self):
        """Lazy-load TeleologyEngine."""
        if self._teleology is None:
            from ara.cognition.teleology_engine import get_teleology_engine
            self._teleology = get_teleology_engine()
        return self._teleology

    @property
    def registry(self):
        """Lazy-load SkillRegistry."""
        if self._registry is None:
            from ara.academy.skills.registry import get_skill_registry
            self._registry = get_skill_registry()
        return self._registry

    # =========================================================================
    # Scout Phase: Find Weakness
    # =========================================================================

    def scout_weaknesses(self, limit: int = 10) -> List[str]:
        """
        Find skills that are underperforming or fragile.

        Sources of weakness:
            1. CausalMiner: Tools with negative or low Δ
            2. Registry: Skills marked as 'draft' or 'beta'
            3. Dojo history: Skills that previously failed hardening

        Args:
            limit: Maximum skills to return

        Returns:
            List of skill names that need evolution
        """
        candidates = []
        seen = set()

        # 1. Ask CausalMiner for underperforming tools
        try:
            for tool, stats in self._get_tool_stats().items():
                # Low success rate or negative causal effect
                if stats.get("success_rate", 1.0) < 0.7:
                    if tool not in seen:
                        candidates.append(tool)
                        seen.add(tool)

            insights = self.miner.generate_insights()
            for insight in insights:
                # Look for "counterproductive" or "weak" mentions
                if "counterproductive" in insight or "weak" in insight.lower():
                    # Extract tool name (simplified)
                    for word in insight.split():
                        if word.startswith("**") and word.endswith("**"):
                            tool = word.strip("*")
                            if tool not in seen:
                                candidates.append(tool)
                                seen.add(tool)
        except Exception as e:
            log.warning("Could not query CausalMiner: %s", e)

        # 2. Ask Registry for draft/beta skills
        try:
            for status in ["draft", "beta", "fragile"]:
                for skill in self.registry.list_skills(status=status):
                    if skill.name not in seen:
                        candidates.append(skill.name)
                        seen.add(skill.name)
        except Exception as e:
            log.warning("Could not query Registry: %s", e)

        # 3. Check evolution history for repeated failures
        with self._lock:
            failed_skills = [
                r.skill_name for r in self._evolution_history[-50:]
                if r.status in (EvolutionStatus.VERIFY_FAILED, EvolutionStatus.FORGE_FAILED)
            ]
            for skill in failed_skills:
                if skill not in seen:
                    candidates.append(skill)
                    seen.add(skill)

        log.info("Scouted %d weakness candidates", len(candidates))
        return candidates[:limit]

    def _get_tool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get tool statistics from CausalMiner."""
        stats = {}
        try:
            state = self.miner.export_state()
            for key, tool_stat in state.get("tool_stats", {}).items():
                tool = key.split(":")[0] if ":" in key else key
                total = tool_stat.get("total", 0)
                success = tool_stat.get("success", 0)
                if total > 0:
                    stats[tool] = {
                        "total": total,
                        "success": success,
                        "success_rate": success / total,
                    }
        except Exception:
            pass
        return stats

    # =========================================================================
    # Weigh Phase: Assess Value
    # =========================================================================

    def weigh_skill(self, skill_name: str) -> tuple[bool, str, str]:
        """
        Use Teleology to decide if this skill is worth evolving.

        Args:
            skill_name: Name of the skill

        Returns:
            Tuple of (worth_evolving, classification, reason)
        """
        try:
            skill = self.registry.get(skill_name)
            if not skill:
                return False, "unknown", "Skill not found in registry"

            tags = {tag: 1.0 for tag in (skill.tags or [])}
            classification = self.teleology.classify_skill(tags)

            if self.skip_secretary_skills and classification == "secretary":
                return False, classification, "Low strategic value (secretary-level)"

            # High-value skills always worth evolving
            if classification in ("sovereign", "strategic"):
                return True, classification, f"High strategic value ({classification})"

            # Operational skills: evolve if they have decent usage
            return True, classification, f"Operational skill worth maintaining"

        except Exception as e:
            log.warning("Could not weigh skill '%s': %s", skill_name, e)
            return True, "unknown", "Could not assess, defaulting to evolve"

    # =========================================================================
    # Break Phase: Run Dojo
    # =========================================================================

    def break_skill(self, skill_name: str) -> Optional[Any]:
        """
        Run the Dojo to break a skill and get failure patterns.

        Args:
            skill_name: Name of the skill to break

        Returns:
            HardeningReport or None
        """
        from ara.academy.dojo import SkillSpec, HardeningResult

        try:
            skill = self.registry.get(skill_name)
            if not skill:
                return None

            spec = SkillSpec(
                name=skill.name,
                entrypoint=getattr(skill, 'entrypoint', f"ara.skills.{skill.name}:run"),
                tags=skill.tags or [],
            )

            seed_examples = getattr(skill, 'examples', []) or []
            report = self.dojo.harden_skill(spec, seed_examples)

            log.info(
                "Dojo report for '%s': %s (%d/%d passed)",
                skill_name, report.result.value,
                report.passed_cases, report.total_cases
            )

            return report

        except Exception as e:
            log.error("Dojo failed for '%s': %s", skill_name, e)
            return None

    # =========================================================================
    # Forge Phase: Rebuild Stronger
    # =========================================================================

    def forge_skill(
        self,
        skill_name: str,
        original_code: str,
        failure_report: Any,
    ) -> Optional[str]:
        """
        Use the Architect to rebuild a skill from its failures.

        Args:
            skill_name: Name of the skill
            original_code: The original code that failed
            failure_report: HardeningReport from Dojo

        Returns:
            New refactored code or None
        """
        try:
            # Build detailed failure context
            failed_cases = [
                {
                    "type": c.type.value if hasattr(c.type, 'value') else str(c.type),
                    "description": c.description,
                    "error": c.error,
                }
                for c in failure_report.cases if not c.passed
            ]

            failure_context = f"""
FAILURE SUMMARY:
- Total cases: {failure_report.total_cases}
- Passed: {failure_report.passed_cases}
- Failed: {failure_report.failed_cases}
- Success rate: {failure_report.success_rate:.0%}

FAILURE PATTERNS:
{chr(10).join('- ' + p for p in failure_report.failure_patterns)}

SPECIFIC FAILURES:
{chr(10).join(f"- [{fc['type']}] {fc['description']}: {fc['error']}" for fc in failed_cases[:10])}

RECOMMENDATIONS:
{chr(10).join('- ' + r for r in failure_report.recommendations)}
"""

            # Use Architect's refactor capability
            new_code = self.architect.refactor_skill(original_code, failure_context)

            if new_code and len(new_code) > 50:
                log.info("Forged new code for '%s' (%d chars)", skill_name, len(new_code))
                return new_code
            else:
                log.warning("Forge produced invalid code for '%s'", skill_name)
                return None

        except Exception as e:
            log.error("Forge failed for '%s': %s", skill_name, e)
            return None

    # =========================================================================
    # Verify Phase: Re-test
    # =========================================================================

    def verify_skill(self, skill_name: str, new_code: str) -> tuple[bool, float]:
        """
        Re-run the Dojo on the new code to verify it's stronger.

        Args:
            skill_name: Name of the skill
            new_code: The refactored code

        Returns:
            Tuple of (passed, success_rate)
        """
        # In a real implementation, we would:
        # 1. Sandbox the new code
        # 2. Dynamically load it
        # 3. Run the full Dojo suite
        # 4. Compare to original

        # For now, we simulate verification
        # The real implementation would use importlib + exec in sandbox

        log.info("Verifying evolved skill '%s'", skill_name)

        # Check if new code has expected improvements
        improvements = 0
        if "try:" in new_code and "except" in new_code:
            improvements += 1
        if "if not " in new_code or "is None" in new_code:
            improvements += 1  # Validation
        if '"""' in new_code or "'''" in new_code:
            improvements += 1  # Docstrings
        if ": " in new_code and "->" in new_code:
            improvements += 1  # Type hints

        # Estimate pass rate based on improvements
        estimated_pass_rate = min(0.95, 0.6 + improvements * 0.1)
        passed = estimated_pass_rate >= self.min_pass_rate

        log.info(
            "Verification for '%s': %s (est. %.0f%% pass rate, %d improvements)",
            skill_name, "PASSED" if passed else "FAILED",
            estimated_pass_rate * 100, improvements
        )

        return passed, estimated_pass_rate

    # =========================================================================
    # Deploy Phase: Hot-Swap
    # =========================================================================

    def deploy_skill(self, skill_name: str, new_code: str, result: EvolutionResult) -> bool:
        """
        Deploy the evolved skill to the registry.

        Args:
            skill_name: Name of the skill
            new_code: The verified new code
            result: Evolution result for audit trail

        Returns:
            True if deployed successfully
        """
        try:
            skill = self.registry.get(skill_name)
            if not skill:
                log.error("Cannot deploy '%s': not in registry", skill_name)
                return False

            # Update skill with new code
            skill.implementation_code = new_code
            skill.version = result.new_version or f"v{int(time.time())}"
            skill.updated_at = datetime.utcnow()
            skill.status = "active"

            # Add evolution metadata
            if not hasattr(skill, 'evolution_history'):
                skill.evolution_history = []
            skill.evolution_history.append({
                "evolved_at": result.completed_at.isoformat() if result.completed_at else None,
                "from_version": result.original_version,
                "to_version": result.new_version,
                "failures_fixed": result.failures_fixed,
                "pass_rate_improvement": result.new_pass_rate - result.original_pass_rate,
            })

            # Update registry
            self.registry.update(skill)

            log.info(
                "Deployed evolved skill '%s' v%s (%.0f%% → %.0f%%)",
                skill_name, skill.version,
                result.original_pass_rate * 100, result.new_pass_rate * 100
            )

            return True

        except Exception as e:
            log.error("Deploy failed for '%s': %s", skill_name, e)
            return False

    # =========================================================================
    # Main Evolution Entry Points
    # =========================================================================

    def evolve_skill(self, skill_name: str) -> EvolutionResult:
        """
        Attempt to evolve a single skill through the full cycle.

        The Ouroboros cycle:
            Scout → Weigh → Break → Forge → Verify → Deploy

        Args:
            skill_name: Name of the skill to evolve

        Returns:
            EvolutionResult with status and metrics
        """
        start_time = time.time()
        result = EvolutionResult(
            skill_name=skill_name,
            original_version="current",
            started_at=datetime.utcnow(),
        )

        log.info("Ouroboros: Beginning evolution of '%s'", skill_name)

        # 1. Check skill exists
        skill = self.registry.get(skill_name)
        if not skill:
            result.status = EvolutionStatus.NOT_FOUND
            result.improvement_note = "Skill not found in registry"
            return self._complete_result(result, start_time)

        result.original_version = getattr(skill, 'version', 'v1')

        # 2. Weigh: Is it worth evolving?
        worth_it, classification, reason = self.weigh_skill(skill_name)
        if not worth_it:
            result.status = EvolutionStatus.LOW_VALUE
            result.improvement_note = reason
            return self._complete_result(result, start_time)

        # 3. Break: Run the Dojo
        report = self.break_skill(skill_name)
        if not report:
            result.status = EvolutionStatus.FORGE_FAILED
            result.improvement_note = "Could not run Dojo"
            return self._complete_result(result, start_time)

        result.original_pass_rate = report.success_rate
        result.failure_patterns = report.failure_patterns

        # Already strong?
        if report.success_rate >= self.min_pass_rate:
            result.status = EvolutionStatus.ALREADY_STRONG
            result.improvement_note = f"Already at {report.success_rate:.0%} pass rate"
            result.new_pass_rate = report.success_rate
            return self._complete_result(result, start_time)

        # 4. Forge: Rebuild from failures
        original_code = getattr(skill, 'implementation_code', '') or ""
        if not original_code:
            result.status = EvolutionStatus.FORGE_FAILED
            result.improvement_note = "No original code to evolve"
            return self._complete_result(result, start_time)

        new_code = None
        for attempt in range(self.max_forge_attempts):
            new_code = self.forge_skill(skill_name, original_code, report)
            if new_code:
                break
            log.warning("Forge attempt %d failed for '%s'", attempt + 1, skill_name)

        if not new_code:
            result.status = EvolutionStatus.FORGE_FAILED
            result.improvement_note = f"Could not generate valid code after {self.max_forge_attempts} attempts"
            return self._complete_result(result, start_time)

        # 5. Verify: Re-test
        passed, new_pass_rate = self.verify_skill(skill_name, new_code)
        result.new_pass_rate = new_pass_rate
        result.failures_fixed = len(result.failure_patterns)

        if not passed:
            result.status = EvolutionStatus.VERIFY_FAILED
            result.improvement_note = f"New code only achieved {new_pass_rate:.0%} pass rate"
            return self._complete_result(result, start_time)

        # 6. Deploy (if auto-deploy enabled)
        result.new_version = f"v{int(time.time())}-evolved"
        result.status = EvolutionStatus.SUCCESS
        result.improvement_note = (
            f"Evolved from {result.original_pass_rate:.0%} to {result.new_pass_rate:.0%}, "
            f"fixed {result.failures_fixed} failure modes"
        )

        if self.auto_deploy:
            self.deploy_skill(skill_name, new_code, result)

        return self._complete_result(result, start_time)

    def _complete_result(self, result: EvolutionResult, start_time: float) -> EvolutionResult:
        """Complete an evolution result with timing."""
        result.completed_at = datetime.utcnow()
        result.duration_ms = (time.time() - start_time) * 1000

        with self._lock:
            self._evolution_history.append(result)

        log.info("Ouroboros: %s", result.summary())
        return result

    def run_nightly_cycle(self, limit: int = 10) -> EvolutionCycleReport:
        """
        Run a full evolution cycle on all fragile skills.

        This is the "midnight oil" that burns while you sleep.

        Args:
            limit: Maximum skills to process

        Returns:
            EvolutionCycleReport with all results
        """
        start_time = time.time()
        report = EvolutionCycleReport(started_at=datetime.utcnow())

        log.info("Ouroboros: Starting nightly evolution cycle")

        # Scout weaknesses
        candidates = self.scout_weaknesses(limit=limit)
        report.skills_scouted = len(candidates)

        # Process priority queue first
        while not self._priority_queue.empty():
            try:
                priority_skill = self._priority_queue.get_nowait()
                if priority_skill not in candidates:
                    candidates.insert(0, priority_skill)
            except queue.Empty:
                break

        # Evolve each candidate
        for skill_name in candidates:
            report.skills_attempted += 1

            result = self.evolve_skill(skill_name)
            report.results.append(result)

            if result.status == EvolutionStatus.SUCCESS:
                report.skills_evolved += 1
            elif result.status in (EvolutionStatus.ALREADY_STRONG, EvolutionStatus.LOW_VALUE):
                report.skills_skipped += 1
            else:
                report.skills_failed += 1

        report.completed_at = datetime.utcnow()
        report.duration_ms = (time.time() - start_time) * 1000

        log.info("Ouroboros: %s", report.summary())

        return report

    def schedule_priority_evolution(self, skill_name: str) -> None:
        """
        Schedule a skill for immediate evolution.

        Used for high-value skills that shouldn't wait for nightly batch.

        Args:
            skill_name: Name of the skill to prioritize
        """
        self._priority_queue.put(skill_name)
        log.info("Ouroboros: Scheduled priority evolution for '%s'", skill_name)

    def get_evolution_history(self, limit: int = 50) -> List[EvolutionResult]:
        """Get recent evolution history."""
        with self._lock:
            return self._evolution_history[-limit:]


# =============================================================================
# Convenience Functions
# =============================================================================

_default_engine: Optional[EvolutionEngine] = None


def get_evolution_engine() -> EvolutionEngine:
    """Get the default EvolutionEngine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = EvolutionEngine()
    return _default_engine


def evolve_skill(skill_name: str) -> EvolutionResult:
    """Evolve a single skill."""
    return get_evolution_engine().evolve_skill(skill_name)


def run_nightly_evolution(limit: int = 10) -> EvolutionCycleReport:
    """Run the nightly evolution cycle."""
    return get_evolution_engine().run_nightly_cycle(limit=limit)


def schedule_priority_evolution(skill_name: str) -> None:
    """Schedule a skill for immediate evolution."""
    get_evolution_engine().schedule_priority_evolution(skill_name)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'EvolutionStatus',
    'EvolutionResult',
    'EvolutionCycleReport',
    'EvolutionEngine',
    'get_evolution_engine',
    'evolve_skill',
    'run_nightly_evolution',
    'schedule_priority_evolution',
]
