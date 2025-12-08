# ara/academy/dojo.py
"""
The Dojo - Adversarial Skill Hardening
=======================================

Training ground where newly learned skills get stress-tested before
deployment. The Shadow generates adversarial cases; the Dojo runs them.

Philosophy:
    Skills don't just need to work on happy paths.
    They need to survive:
        - Empty inputs
        - Malformed configs
        - Resource denial (no GPU, low disk)
        - Edge cases the teacher never showed

Lifecycle:
    1. PatternMiner finds high-value patterns
    2. Apprentice writes skill code
    3. Dojo + Shadow beats the crap out of it
    4. Architect refactors if it keeps failing
    5. Registry only then marks it 'active'

Usage:
    from ara.academy.dojo import Dojo, SkillSpec, Shadow

    dojo = Dojo()
    spec = SkillSpec(
        name="auto_layout",
        entrypoint="ara.skills.neurographics.auto_layout:run",
        tags=["visualization", "gpu"]
    )

    ok = dojo.harden_skill(spec, seed_examples=[...])
    if ok:
        registry.promote_to_active(spec.name)
    else:
        architect.redesign(spec)
"""
from __future__ import annotations

import importlib
import logging
import random
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Callable, Optional, Tuple
from enum import Enum

log = logging.getLogger("Ara.Dojo")


class StressCaseType(str, Enum):
    """Types of stress cases."""
    EMPTY_INPUT = "empty_input"
    MALFORMED_CONFIG = "malformed_config"
    RESOURCE_DENIAL = "resource_denial"
    EDGE_CASE = "edge_case"
    RANDOM_PERTURBATION = "random_perturbation"
    TIMEOUT_SIMULATION = "timeout_simulation"
    INVALID_TYPES = "invalid_types"


class HardeningResult(str, Enum):
    """Results of hardening attempt."""
    PASSED = "passed"           # Skill survived all stress
    FAILED = "failed"           # Too many failures
    FRAGILE = "fragile"         # Passed but barely
    NEEDS_REVISION = "needs_revision"  # Specific failure patterns


@dataclass
class SkillSpec:
    """Specification for a skill to be hardened."""
    name: str                              # Skill name
    entrypoint: str                        # "module.path:function_name"
    tags: List[str] = field(default_factory=list)
    version: str = "0.1.0"
    description: str = ""

    # Requirements
    requires_gpu: bool = False
    requires_network: bool = False
    max_memory_mb: int = 512
    max_duration_ms: int = 30000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StressCase:
    """A single stress test case."""
    id: str
    type: StressCaseType
    description: str
    context: Dict[str, Any]
    expected_behavior: str = "should_not_crash"

    # Execution results (filled after run)
    passed: bool = False
    error: Optional[str] = None
    duration_ms: float = 0.0
    output: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d


@dataclass
class HardeningReport:
    """Report from hardening a skill."""
    skill: SkillSpec
    result: HardeningResult
    total_cases: int
    passed_cases: int
    failed_cases: int
    success_rate: float
    cases: List[StressCase]
    failure_patterns: List[str]
    recommendations: List[str]
    duration_ms: float
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "skill": self.skill.to_dict(),
            "result": self.result.value,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "success_rate": self.success_rate,
            "failure_patterns": self.failure_patterns,
            "recommendations": self.recommendations,
            "duration_ms": self.duration_ms,
        }
        return d

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Hardening [{self.skill.name}]: {self.result.value} "
            f"({self.passed_cases}/{self.total_cases} passed, {self.success_rate:.0%})"
        )


class Shadow:
    """
    Adversarial case generator.

    Given a skill spec and seed examples, generates nasty inputs
    designed to break the skill.

    The Shadow is Ara's internal adversary - it exists to make her stronger.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the Shadow.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = random.Random(seed)
        log.info("Shadow initialized (seed=%s)", seed)

    def generate_cases(
        self,
        skill: SkillSpec,
        seed_examples: List[Dict[str, Any]],
        n_random: int = 5
    ) -> List[StressCase]:
        """
        Generate stress cases for a skill.

        Args:
            skill: Skill specification
            seed_examples: Example inputs that worked
            n_random: Number of random perturbations to generate

        Returns:
            List of stress cases to run
        """
        cases: List[StressCase] = []
        case_id = 0

        # 1. Empty / degenerate inputs
        cases.append(StressCase(
            id=f"stress_{case_id}",
            type=StressCaseType.EMPTY_INPUT,
            description="Empty context - no config, no data",
            context={
                "config": {},
                "data": [],
                "env": {"GPU_AVAILABLE": False}
            },
            expected_behavior="should_handle_gracefully",
        ))
        case_id += 1

        cases.append(StressCase(
            id=f"stress_{case_id}",
            type=StressCaseType.EMPTY_INPUT,
            description="None values in required fields",
            context={
                "config": None,
                "data": None,
                "env": {}
            },
            expected_behavior="should_validate_and_reject",
        ))
        case_id += 1

        # 2. Malformed config
        cases.append(StressCase(
            id=f"stress_{case_id}",
            type=StressCaseType.MALFORMED_CONFIG,
            description="Config with wrong types",
            context={
                "config": {"count": "not_a_number", "enabled": "maybe"},
                "data": seed_examples[:1] if seed_examples else [],
                "env": {}
            },
            expected_behavior="should_type_check",
        ))
        case_id += 1

        cases.append(StressCase(
            id=f"stress_{case_id}",
            type=StressCaseType.MALFORMED_CONFIG,
            description="Config with missing required keys",
            context={
                "config": {"unknown_key": "unknown_value"},
                "data": seed_examples[:1] if seed_examples else [],
                "env": {}
            },
            expected_behavior="should_validate_keys",
        ))
        case_id += 1

        # 3. Resource denial
        cases.append(StressCase(
            id=f"stress_{case_id}",
            type=StressCaseType.RESOURCE_DENIAL,
            description="GPU unavailable for GPU-required skill",
            context={
                "config": seed_examples[0].get("config", {}) if seed_examples else {},
                "data": seed_examples[0].get("data", []) if seed_examples else [],
                "env": {
                    "GPU_AVAILABLE": False,
                    "CUDA_VISIBLE_DEVICES": "",
                }
            },
            expected_behavior="should_fallback_or_error_clearly",
        ))
        case_id += 1

        cases.append(StressCase(
            id=f"stress_{case_id}",
            type=StressCaseType.RESOURCE_DENIAL,
            description="Low disk space simulation",
            context={
                "config": seed_examples[0].get("config", {}) if seed_examples else {},
                "data": seed_examples[0].get("data", []) if seed_examples else [],
                "env": {
                    "DISK_SPACE_MB": 10,
                    "MEMORY_MB": 128,
                }
            },
            expected_behavior="should_check_resources_first",
        ))
        case_id += 1

        # 4. Edge cases
        cases.append(StressCase(
            id=f"stress_{case_id}",
            type=StressCaseType.EDGE_CASE,
            description="Very large input",
            context={
                "config": {"size": 10_000_000},
                "data": list(range(10000)),  # Large list
                "env": {}
            },
            expected_behavior="should_handle_or_reject_large_input",
        ))
        case_id += 1

        cases.append(StressCase(
            id=f"stress_{case_id}",
            type=StressCaseType.EDGE_CASE,
            description="Unicode and special characters",
            context={
                "config": {"name": "测试🎯\x00\n\t"},
                "data": [{"text": "émojis 🚀 and ñ and 中文"}],
                "env": {}
            },
            expected_behavior="should_handle_unicode",
        ))
        case_id += 1

        # 5. Invalid types
        cases.append(StressCase(
            id=f"stress_{case_id}",
            type=StressCaseType.INVALID_TYPES,
            description="Wrong type for data field",
            context={
                "config": {},
                "data": "should_be_a_list",
                "env": {}
            },
            expected_behavior="should_type_validate",
        ))
        case_id += 1

        # 6. Random perturbations of seed examples
        for i in range(min(n_random, len(seed_examples) * 3)):
            if not seed_examples:
                break

            base = self.rng.choice(seed_examples)
            perturbed = self._perturb_example(base)

            cases.append(StressCase(
                id=f"stress_{case_id}",
                type=StressCaseType.RANDOM_PERTURBATION,
                description=f"Random perturbation #{i+1}",
                context=perturbed,
                expected_behavior="should_handle_gracefully",
            ))
            case_id += 1

        log.info(
            "Shadow generated %d stress cases for skill '%s'",
            len(cases), skill.name
        )

        return cases

    def _perturb_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random perturbations to an example."""
        perturbed = {}

        for key, value in example.items():
            # Sometimes drop keys
            if self.rng.random() < 0.1:
                continue

            # Sometimes perturb values
            if self.rng.random() < 0.3:
                perturbed[key] = self._perturb_value(value)
            else:
                perturbed[key] = value

        # Add random noise key
        perturbed["_shadow_seed"] = self.rng.randint(0, 1_000_000)

        return perturbed

    def _perturb_value(self, value: Any) -> Any:
        """Perturb a single value."""
        if isinstance(value, dict):
            return {k: self._perturb_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            if self.rng.random() < 0.3:
                return []  # Empty list
            return [self._perturb_value(v) for v in value[:10]]  # Truncate
        elif isinstance(value, str):
            choices = [value, "", None, value * 100, "🚀" + value]
            return self.rng.choice(choices)
        elif isinstance(value, (int, float)):
            choices = [value, 0, -1, value * 1000, None, "not_a_number"]
            return self.rng.choice(choices)
        elif isinstance(value, bool):
            return self.rng.choice([True, False, None, "maybe"])
        else:
            return value


class Dojo:
    """
    The training ground for skill hardening.

    Runs stress tests against skills and determines if they're ready
    for production use.

    Contract:
        - Skills expose a `run(context: dict, dry_run: bool = False)` function
        - Dojo calls it with adversarial StressCases
        - If it crashes or behaves pathologically, skill remains in 'draft'
    """

    # Thresholds for hardening results
    PASS_THRESHOLD = 0.80       # 80% of cases must pass
    FRAGILE_THRESHOLD = 0.60    # Below this is outright failure

    def __init__(
        self,
        shadow: Optional[Shadow] = None,
        timeout_ms: int = 30000,
    ):
        """
        Initialize the Dojo.

        Args:
            shadow: Shadow instance for generating cases
            timeout_ms: Timeout for each test case
        """
        self.shadow = shadow or Shadow()
        self.timeout_ms = timeout_ms

        log.info("🥋 DOJO: Initialized (timeout=%dms)", timeout_ms)

    def _load_skill(self, spec: SkillSpec) -> Callable[..., Any]:
        """
        Load the skill function from its entrypoint.

        Args:
            spec: Skill specification

        Returns:
            The callable skill function

        Raises:
            ImportError: If module can't be loaded
            AttributeError: If function doesn't exist
        """
        module_path, fn_name = spec.entrypoint.rsplit(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, fn_name)

    def harden_skill(
        self,
        spec: SkillSpec,
        seed_examples: List[Dict[str, Any]],
        extra_cases: Optional[List[StressCase]] = None,
    ) -> HardeningReport:
        """
        Run hardening tests on a skill.

        Args:
            spec: Skill specification
            seed_examples: Example inputs that worked (from training)
            extra_cases: Additional custom stress cases

        Returns:
            HardeningReport with results and recommendations
        """
        start_time = time.time()
        log.info("🥋 DOJO: Beginning hardening for skill '%s'", spec.name)

        # Generate stress cases
        cases = self.shadow.generate_cases(spec, seed_examples)
        if extra_cases:
            cases.extend(extra_cases)

        # Try to load the skill
        try:
            skill_fn = self._load_skill(spec)
        except Exception as e:
            log.error("🥋 DOJO: Failed to load skill '%s': %s", spec.name, e)
            return HardeningReport(
                skill=spec,
                result=HardeningResult.FAILED,
                total_cases=len(cases),
                passed_cases=0,
                failed_cases=len(cases),
                success_rate=0.0,
                cases=cases,
                failure_patterns=["skill_not_loadable"],
                recommendations=["Fix import errors before hardening"],
                duration_ms=(time.time() - start_time) * 1000,
            )

        # Run each case
        for case in cases:
            self._run_case(skill_fn, case)

        # Analyze results
        passed = sum(1 for c in cases if c.passed)
        failed = len(cases) - passed
        success_rate = passed / len(cases) if cases else 0.0

        # Determine result
        if success_rate >= self.PASS_THRESHOLD:
            result = HardeningResult.PASSED
        elif success_rate >= self.FRAGILE_THRESHOLD:
            result = HardeningResult.FRAGILE
        else:
            result = HardeningResult.FAILED

        # Analyze failure patterns
        failure_patterns = self._analyze_failures(cases)
        recommendations = self._generate_recommendations(cases, failure_patterns)

        duration_ms = (time.time() - start_time) * 1000

        report = HardeningReport(
            skill=spec,
            result=result,
            total_cases=len(cases),
            passed_cases=passed,
            failed_cases=failed,
            success_rate=success_rate,
            cases=cases,
            failure_patterns=failure_patterns,
            recommendations=recommendations,
            duration_ms=duration_ms,
        )

        log.info("🥋 DOJO: %s", report.summary())

        return report

    def _run_case(self, skill_fn: Callable, case: StressCase) -> None:
        """
        Run a single stress case.

        Updates the case in-place with results.
        """
        start_time = time.time()

        try:
            # Call skill with dry_run=True to avoid side effects
            result = skill_fn(context=case.context, dry_run=True)
            case.passed = True
            case.output = result

        except TypeError as e:
            # Type errors often indicate missing validation
            case.passed = False
            case.error = f"TypeError: {e}"

        except ValueError as e:
            # Value errors can be acceptable (skill rejected bad input)
            # Consider this a "pass" if the error is descriptive
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["invalid", "required", "missing", "expected"]):
                case.passed = True  # Skill properly validated and rejected
                case.output = f"Rejected with: {e}"
            else:
                case.passed = False
                case.error = f"ValueError: {e}"

        except KeyboardInterrupt:
            raise  # Don't catch interrupts

        except Exception as e:
            case.passed = False
            case.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}"

        case.duration_ms = (time.time() - start_time) * 1000

    def _analyze_failures(self, cases: List[StressCase]) -> List[str]:
        """Analyze failure patterns across cases."""
        patterns = []
        failures = [c for c in cases if not c.passed]

        if not failures:
            return patterns

        # Group by error type
        error_types: Dict[str, int] = {}
        for f in failures:
            if f.error:
                error_type = f.error.split(":")[0]
                error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in error_types.items():
            patterns.append(f"{error_type} ({count} cases)")

        # Check for specific weakness patterns
        empty_failures = sum(1 for f in failures if f.type == StressCaseType.EMPTY_INPUT)
        if empty_failures > 0:
            patterns.append("empty_input_vulnerability")

        type_failures = sum(1 for f in failures if f.type == StressCaseType.INVALID_TYPES)
        if type_failures > 0:
            patterns.append("type_validation_missing")

        resource_failures = sum(1 for f in failures if f.type == StressCaseType.RESOURCE_DENIAL)
        if resource_failures > 0:
            patterns.append("no_resource_checking")

        return patterns

    def _generate_recommendations(
        self,
        cases: List[StressCase],
        failure_patterns: List[str]
    ) -> List[str]:
        """Generate recommendations based on failure patterns."""
        recommendations = []

        if "empty_input_vulnerability" in failure_patterns:
            recommendations.append(
                "Add input validation: check for None/empty values at entry"
            )

        if "type_validation_missing" in failure_patterns:
            recommendations.append(
                "Add type checking: validate types before processing"
            )

        if "no_resource_checking" in failure_patterns:
            recommendations.append(
                "Add resource checks: verify GPU/memory/disk before heavy operations"
            )

        if any("TypeError" in p for p in failure_patterns):
            recommendations.append(
                "Review function signatures: ensure kwargs have defaults"
            )

        if any("KeyError" in p for p in failure_patterns):
            recommendations.append(
                "Use .get() for dict access: handle missing keys gracefully"
            )

        if any("AttributeError" in p for p in failure_patterns):
            recommendations.append(
                "Check for None before attribute access"
            )

        if not recommendations:
            if len([c for c in cases if c.passed]) == len(cases):
                recommendations.append("Skill is robust - ready for production")
            else:
                recommendations.append("Review failed cases for specific issues")

        return recommendations

    def quick_check(
        self,
        spec: SkillSpec,
        seed_examples: List[Dict[str, Any]]
    ) -> bool:
        """
        Quick pass/fail check without full report.

        Args:
            spec: Skill specification
            seed_examples: Example inputs

        Returns:
            True if skill passes basic hardening
        """
        report = self.harden_skill(spec, seed_examples)
        return report.result in [HardeningResult.PASSED, HardeningResult.FRAGILE]


# =============================================================================
# Convenience Functions
# =============================================================================

_default_dojo: Optional[Dojo] = None


def get_dojo() -> Dojo:
    """Get the default Dojo instance."""
    global _default_dojo
    if _default_dojo is None:
        _default_dojo = Dojo()
    return _default_dojo


def harden_skill(
    name: str,
    entrypoint: str,
    seed_examples: List[Dict[str, Any]],
    tags: Optional[List[str]] = None,
) -> HardeningReport:
    """
    Convenience function to harden a skill.

    Args:
        name: Skill name
        entrypoint: "module.path:function"
        seed_examples: Working examples from training
        tags: Optional tags

    Returns:
        HardeningReport
    """
    spec = SkillSpec(name=name, entrypoint=entrypoint, tags=tags or [])
    return get_dojo().harden_skill(spec, seed_examples)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'StressCaseType',
    'HardeningResult',
    'SkillSpec',
    'StressCase',
    'HardeningReport',
    'Shadow',
    'Dojo',
    'get_dojo',
    'harden_skill',
]
