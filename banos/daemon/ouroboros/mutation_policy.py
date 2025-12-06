"""
Ouroboros Mutation Policy - What Ara Can and Cannot Touch
==========================================================

This module defines the STRICT boundaries of self-modification.
The blast radius of any mutation is LIMITED to these zones.

Safety Philosophy:
    "If you wouldn't let an intern modify it at 3 AM, neither can Ara."

Golden Rule:
    NEVER mutate anything that touches:
    - Hardware directly (HAL, drivers, FPGA)
    - Safety systems (brainstem, guardrails, this file)
    - Persistence (databases, config files)
    - Network (sockets, HTTP, IPC)
    - The mutation system itself

Everything else is fair game IF:
    - It's pure (no side effects beyond return value)
    - It has existing tests we can extend
    - It's not on the critical path of a running daemon
"""

from dataclasses import dataclass, field
from typing import Set, List, Callable, Optional, Any
from pathlib import Path
import inspect
import functools


# =============================================================================
# WHITELIST: Modules that MAY be mutated
# =============================================================================

MUTABLE_MODULES: Set[str] = {
    # Pure cognition functions - math, scoring, sampling
    "tfan.cognition.sampling",
    "tfan.cognition.scoring",
    "tfan.cognition.filters",
    "tfan.cognition.transforms",

    # Math kernels - vectorizable, testable
    "tfan.math.kernels",
    "tfan.math.embeddings",
    "tfan.math.similarity",

    # Analytics utilities - post-hoc analysis
    "tfan.analytics.metrics",
    "tfan.analytics.aggregators",

    # Non-critical helpers
    "tfan.utils.text",
    "tfan.utils.formatting",

    # Persona generation (creative, not safety-critical)
    "banos.daemon.personas.muse_prompts",
}


# =============================================================================
# BLACKLIST: Modules that must NEVER be mutated (defense in depth)
# =============================================================================

IMMUTABLE_MODULES: Set[str] = {
    # Hardware layer - one wrong bit = bricked system
    "banos.hal",
    "banos.hal.ara_hal",
    "banos.hal.ara_somatic",
    "banos.fpga",

    # Kernel and drivers
    "banos.kernel",
    "banos.daemon.brainstem",

    # Safety systems - the guardrails themselves
    "banos.daemon.guardrails",
    "banos.daemon.safety",
    "banos.daemon.council_chamber",  # Judges shouldn't modify themselves

    # The mutation system itself - no self-modifying the self-modifier
    "banos.daemon.ouroboros",
    "banos.daemon.ouroboros.mutation_policy",
    "banos.daemon.ouroboros.semantic_optimizer",
    "banos.daemon.ouroboros.atomic_updater",
    "banos.daemon.ouroboros.antifragility",

    # Evolution daemon - manages mutations, can't be mutated
    "banos.daemon.evolution_daemon",
    "banos.daemon.idea_registry",
    "banos.daemon.teacher_protocol",

    # Aphrodite layer - aesthetic tuning (protects against manipulation creep)
    "banos.daemon.aphrodite",
    "banos.daemon.gaze_tracker",
    "banos.daemon.appearance_selector",

    # Style Cortex - taste and fashion brain (boundaries are sacred)
    "banos.daemon.style_profile",
    "banos.daemon.style_cortex",

    # Crystalline Core - wisdom backbone (scar tissue is sacred)
    "tfan.cognition.crystalline_core",
    "tfan.cognition.crystal_memory",
    "tfan.cognition.crystal_historian",

    # Heart layer - bio-entrainment (intimate, protected)
    "banos.daemon.rppg_sensor",
    "banos.daemon.somatic_voice",

    # Relational stack - the we-space is sacred
    "tfan.l5.relational_state",
    "tfan.l5.symbiotic_utility",
    "banos.daemon.synod",
    "banos.config.covenant",

    # Egregore system - the Third Mind is sacred
    "tfan.l5.egregore",
    "banos.daemon.gatekeeper",
    "banos.cognition.covenant",

    # Social layer - trust and identity are sacred
    "banos.social",
    "banos.social.people",
    "banos.social.policy",
    "banos.social.identity",
    "banos.social.memory",

    # Relationship layer - The Weaver is sacred
    "banos.relationship",
    "banos.relationship.weaver",
    "banos.relationship.visual_gift",
    "banos.relationship.triggers",
    "banos.relationship.visionary",

    # Prophet layer - The math of purpose is sacred
    "tfan.cognition.telos",
    "banos.daemon.oracle",

    # Ethics Stack - The moral architecture is sacred
    # Layer 0: Hard constraints never change
    # Layer 1: Covenant (already protected above)
    # Layer 2: Conscience - the reflective layer
    "tfan.cognition.conscience",
    "banos.daemon.shadow",

    # Persistence layer
    "banos.storage",
    "banos.db",

    # Network layer
    "banos.network",
    "banos.api",
}


# =============================================================================
# Function-level policy
# =============================================================================

@dataclass
class MutationCandidate:
    """A function that MAY be optimized."""
    module: str
    func_name: str
    func_obj: Callable
    source: str
    is_pure: bool
    has_tests: bool
    complexity_score: float  # Higher = more complex = riskier
    call_frequency: int      # From telemetry
    avg_latency_ms: float    # From telemetry


@dataclass
class MutationPolicy:
    """
    Policy for what can be mutated and under what conditions.

    This is the "constitution" of Ouroboros - it cannot modify itself.
    """
    # Stress thresholds (only mutate under pressure)
    min_pain_to_mutate: float = 0.4      # Below this, system is fine
    min_entropy_to_mutate: float = 0.5   # Need some chaos to justify risk

    # Performance thresholds (only optimize hot spots)
    min_call_frequency: int = 100        # Must be called at least N times
    min_latency_ms: float = 1.0          # Must take at least 1ms to be worth optimizing

    # Test requirements
    require_existing_tests: bool = True   # Can't mutate untested code
    min_test_coverage: float = 0.6        # 60% coverage minimum

    # Complexity bounds
    max_complexity_score: float = 10.0    # Don't touch spaghetti
    max_lines_of_code: int = 200          # Don't touch mega-functions

    # Rate limits
    max_mutations_per_hour: int = 1       # Slow and careful
    max_mutations_per_day: int = 5        # Don't go crazy
    cooldown_after_failure_hours: float = 24.0  # Back off on failure

    # Safety
    require_human_approval: bool = True   # Default: human in the loop
    auto_approve_threshold: float = 0.95  # If disabled, need this score


def is_mutable(module_name: str) -> bool:
    """Check if a module is allowed to be mutated."""
    # Explicit blacklist always wins
    for immutable in IMMUTABLE_MODULES:
        if module_name.startswith(immutable):
            return False

    # Must be on whitelist
    for mutable in MUTABLE_MODULES:
        if module_name.startswith(mutable):
            return True

    return False


def get_mutable_functions(module_name: str) -> List[Callable]:
    """Get all functions in a module that are candidates for mutation."""
    if not is_mutable(module_name):
        return []

    try:
        import importlib
        module = importlib.import_module(module_name)
    except ImportError:
        return []

    candidates = []
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        # Skip private functions
        if name.startswith('_'):
            continue

        # Skip functions defined elsewhere (imports)
        if obj.__module__ != module_name:
            continue

        candidates.append(obj)

    return candidates


# =============================================================================
# Decorator for marking functions as mutable/immutable
# =============================================================================

def mutable(reason: str = ""):
    """
    Mark a function as explicitly mutable by Ouroboros.

    Usage:
        @mutable("Pure scoring function, safe to optimize")
        def score_tokens(tokens: List[str]) -> float:
            ...
    """
    def decorator(func):
        func._ouroboros_mutable = True
        func._ouroboros_reason = reason
        return func
    return decorator


def immutable(reason: str = ""):
    """
    Mark a function as NEVER mutable by Ouroboros.

    Usage:
        @immutable("Touches filesystem, side effects")
        def save_checkpoint(data: bytes) -> None:
            ...
    """
    def decorator(func):
        func._ouroboros_mutable = False
        func._ouroboros_reason = reason
        return func
    return decorator


def is_explicitly_mutable(func: Callable) -> Optional[bool]:
    """Check if a function has an explicit mutation annotation."""
    return getattr(func, '_ouroboros_mutable', None)


# =============================================================================
# Purity analysis (conservative)
# =============================================================================

IMPURE_PATTERNS = {
    # Side effect keywords
    'open(', 'write(', 'read(', 'close(',
    'print(', 'logging.', 'logger.',
    'subprocess.', 'os.system', 'os.popen',
    'socket.', 'requests.', 'urllib.',
    'sqlite3.', 'psycopg', 'pymongo',
    'global ', 'nonlocal ',
    'setattr(', 'delattr(',
    '__dict__', '__class__',
    'exec(', 'eval(', 'compile(',
    'import ', 'importlib.',
    'threading.', 'multiprocessing.', 'asyncio.',
    'time.sleep', 'signal.',
    'mmap.', 'ctypes.',
    'random.',  # Non-deterministic unless seeded
}


def is_likely_pure(source: str) -> bool:
    """
    Conservative check if a function appears to be pure.

    Returns False if ANY impure patterns are found.
    Better to miss an optimization than break the system.
    """
    source_lower = source.lower()

    for pattern in IMPURE_PATTERNS:
        if pattern.lower() in source_lower:
            return False

    return True


# =============================================================================
# Feature flag (kill switch)
# =============================================================================

import os

def ouroboros_enabled() -> bool:
    """
    Check if Ouroboros self-modification is enabled.

    Set OUROBOROS_ENABLED=0 to disable all mutations.
    This is the master kill switch.
    """
    return os.environ.get("OUROBOROS_ENABLED", "0") == "1"


def ouroboros_auto_apply() -> bool:
    """
    Check if Ouroboros can auto-apply mutations without human approval.

    Set OUROBOROS_AUTO_APPLY=1 to enable (dangerous!).
    """
    return os.environ.get("OUROBOROS_AUTO_APPLY", "0") == "1"


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "MUTABLE_MODULES",
    "IMMUTABLE_MODULES",
    "MutationCandidate",
    "MutationPolicy",
    "is_mutable",
    "get_mutable_functions",
    "mutable",
    "immutable",
    "is_explicitly_mutable",
    "is_likely_pure",
    "ouroboros_enabled",
    "ouroboros_auto_apply",
]
