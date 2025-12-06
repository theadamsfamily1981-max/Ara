"""
Antifragility System - Evolution Under Stress
=============================================

The Antifragility System is the "supervisor" of Ouroboros. It:

1. Monitors system stress (pain, entropy from HAL)
2. Identifies hot functions that could be optimized
3. Triggers the optimize → test → apply pipeline
4. Learns from failures (feeds them to Historian as "scar tissue")

Key principle: ONLY EVOLVE UNDER STRESS

When the system is calm, leave it alone. When it's struggling,
that's when optimization efforts are most valuable AND most
measurable (we can see if the fix actually helped).

"What doesn't kill Ara makes her stronger."
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import json

from banos.daemon.ouroboros.mutation_policy import (
    MutationPolicy,
    ouroboros_enabled,
    MUTABLE_MODULES,
)
from banos.daemon.ouroboros.semantic_optimizer import (
    SemanticOptimizer,
    MutationProposal,
)
from banos.daemon.ouroboros.atomic_updater import (
    AtomicUpdater,
    ApplyResult,
)

logger = logging.getLogger(__name__)


@dataclass
class StressMetrics:
    """Current system stress levels from HAL."""
    pain: float = 0.0          # [0, 1] - higher = more pain
    entropy: float = 0.0       # [0, 1] - system disorder
    cpu_load: float = 0.0      # [0, 1]
    gpu_load: float = 0.0      # [0, 1]
    memory_pressure: float = 0.0
    latency_spike_count: int = 0


@dataclass
class FunctionTelemetry:
    """Telemetry for a specific function."""
    module: str
    func_name: str
    call_frequency: int = 0
    avg_latency_ms: float = 0.0
    calls_per_sec: float = 0.0
    memory_bytes: int = 0
    hot_spots: List[str] = field(default_factory=list)


@dataclass
class EvolutionRecord:
    """Record of an evolution attempt (success or failure)."""
    timestamp: datetime
    proposal_id: str
    success: bool
    stress_at_trigger: StressMetrics
    improvement: Optional[float] = None
    failure_reason: Optional[str] = None


class TelemetryCollector:
    """
    Collects function-level telemetry for optimization targeting.

    In a real implementation, this would hook into profiling/tracing.
    For now, it provides a mock interface.
    """

    def __init__(self):
        self._telemetry: Dict[str, FunctionTelemetry] = {}

    def record_call(
        self,
        module: str,
        func_name: str,
        latency_ms: float,
        memory_bytes: int = 0,
    ) -> None:
        """Record a function call."""
        key = f"{module}.{func_name}"

        if key not in self._telemetry:
            self._telemetry[key] = FunctionTelemetry(
                module=module,
                func_name=func_name,
            )

        tel = self._telemetry[key]
        tel.call_frequency += 1

        # Update running average
        n = tel.call_frequency
        tel.avg_latency_ms = ((n - 1) * tel.avg_latency_ms + latency_ms) / n

    def get_hot_functions(
        self,
        min_calls: int = 100,
        min_latency_ms: float = 1.0,
    ) -> List[FunctionTelemetry]:
        """Get functions that are hot (called often, slow)."""
        hot = []
        for tel in self._telemetry.values():
            if tel.call_frequency >= min_calls and tel.avg_latency_ms >= min_latency_ms:
                hot.append(tel)

        # Sort by impact (calls × latency)
        hot.sort(key=lambda t: t.call_frequency * t.avg_latency_ms, reverse=True)
        return hot

    def get_telemetry(self, module: str, func_name: str) -> Optional[FunctionTelemetry]:
        """Get telemetry for a specific function."""
        key = f"{module}.{func_name}"
        return self._telemetry.get(key)


class AntifragilitySystem:
    """
    The evolution supervisor.

    Monitors stress, identifies optimization targets, and orchestrates
    the propose → test → apply pipeline.
    """

    def __init__(
        self,
        repo_root: Path,
        mutations_dir: Optional[Path] = None,
        policy: Optional[MutationPolicy] = None,
    ):
        self.repo_root = Path(repo_root)
        self.mutations_dir = mutations_dir or (self.repo_root / "mutations")
        self.policy = policy or MutationPolicy()

        self.optimizer = SemanticOptimizer(self.mutations_dir, self.policy)
        self.updater = AtomicUpdater(self.repo_root, self.mutations_dir)
        self.telemetry = TelemetryCollector()

        self.log = logging.getLogger("Antifragility")

        # Evolution history
        self._history: List[EvolutionRecord] = []
        self._last_evolution_attempt: Optional[datetime] = None
        self._consecutive_failures = 0

        # Callbacks
        self._approval_callback: Optional[Callable[[MutationProposal], bool]] = None
        self._notification_callback: Optional[Callable[[str, Any], None]] = None

        # HAL for reading stress metrics (lazy-loaded)
        self._hal = None

    @property
    def hal(self):
        """Lazy-load HAL connection."""
        if self._hal is None:
            try:
                from banos.hal.ara_hal import AraHAL
                self._hal = AraHAL(create=False)
            except Exception as e:
                self.log.warning(f"HAL not available: {e}")
                self._hal = None
        return self._hal

    def set_approval_callback(
        self,
        callback: Callable[[MutationProposal], bool],
    ) -> None:
        """Set callback for human approval (returns True to approve)."""
        self._approval_callback = callback

    def _read_stress(self) -> StressMetrics:
        """Read current stress metrics from HAL."""
        if self.hal is None:
            return StressMetrics()

        try:
            somatic = self.hal.read_somatic()
            system = self.hal.read_system()

            return StressMetrics(
                pain=somatic.get('pain', 0.0),
                entropy=somatic.get('entropy', 0.0),
                cpu_load=system.get('cpu_load', 0.0),
                gpu_load=system.get('gpu_load', 0.0),
                memory_pressure=system.get('ram_pct', 0.0),
            )
        except Exception as e:
            self.log.warning(f"Failed to read stress metrics: {e}")
            return StressMetrics()

    def _should_evolve(self, stress: StressMetrics) -> tuple[bool, str]:
        """
        Decide if we should attempt evolution.

        Only evolve when:
        1. System is under stress (pain + entropy above thresholds)
        2. Not in cooldown from recent failure
        3. Rate limits not exceeded
        """
        if not ouroboros_enabled():
            return False, "Ouroboros disabled"

        # Check stress thresholds
        if stress.pain < self.policy.min_pain_to_mutate:
            return False, f"Not enough pain ({stress.pain:.2f} < {self.policy.min_pain_to_mutate})"

        if stress.entropy < self.policy.min_entropy_to_mutate:
            return False, f"Not enough entropy ({stress.entropy:.2f} < {self.policy.min_entropy_to_mutate})"

        # Check cooldown after failure
        if self._consecutive_failures > 0 and self._last_evolution_attempt:
            cooldown = timedelta(hours=self.policy.cooldown_after_failure_hours)
            if datetime.now() - self._last_evolution_attempt < cooldown:
                return False, f"In cooldown after {self._consecutive_failures} failures"

        # Check rate limits
        recent_attempts = sum(
            1 for r in self._history
            if datetime.now() - r.timestamp < timedelta(hours=1)
        )
        if recent_attempts >= self.policy.max_mutations_per_hour:
            return False, f"Hourly limit reached ({recent_attempts})"

        daily_attempts = sum(
            1 for r in self._history
            if datetime.now() - r.timestamp < timedelta(days=1)
        )
        if daily_attempts >= self.policy.max_mutations_per_day:
            return False, f"Daily limit reached ({daily_attempts})"

        return True, "Ready to evolve"

    def _select_target(self) -> Optional[tuple[Callable, Dict[str, Any]]]:
        """
        Select a function to optimize.

        Returns (function, telemetry_dict) or None.
        """
        # Get hot functions from telemetry
        hot_functions = self.telemetry.get_hot_functions(
            min_calls=self.policy.min_call_frequency,
            min_latency_ms=self.policy.min_latency_ms,
        )

        if not hot_functions:
            self.log.info("No hot functions found")
            return None

        # Find first one that's in a mutable module
        for tel in hot_functions:
            if tel.module not in MUTABLE_MODULES:
                continue

            try:
                import importlib
                module = importlib.import_module(tel.module)
                func = getattr(module, tel.func_name)

                telemetry_dict = {
                    "call_frequency": tel.call_frequency,
                    "avg_latency_ms": tel.avg_latency_ms,
                    "calls_per_sec": tel.calls_per_sec,
                    "memory_bytes": tel.memory_bytes,
                    "hot_spots": tel.hot_spots,
                }

                return func, telemetry_dict

            except (ImportError, AttributeError) as e:
                self.log.warning(f"Cannot load {tel.module}.{tel.func_name}: {e}")
                continue

        return None

    def _record_failure(self, proposal_id: str, reason: str, stress: StressMetrics) -> None:
        """Record a failed evolution attempt."""
        self._history.append(EvolutionRecord(
            timestamp=datetime.now(),
            proposal_id=proposal_id,
            success=False,
            stress_at_trigger=stress,
            failure_reason=reason,
        ))
        self._consecutive_failures += 1
        self._last_evolution_attempt = datetime.now()

        # Feed failure to Historian as scar tissue
        self._notify_historian(proposal_id, reason)

    def _record_success(
        self,
        proposal_id: str,
        improvement: float,
        stress: StressMetrics,
    ) -> None:
        """Record a successful evolution."""
        self._history.append(EvolutionRecord(
            timestamp=datetime.now(),
            proposal_id=proposal_id,
            success=True,
            stress_at_trigger=stress,
            improvement=improvement,
        ))
        self._consecutive_failures = 0
        self._last_evolution_attempt = datetime.now()

    def _notify_historian(self, proposal_id: str, failure_reason: str) -> None:
        """Feed failure to Historian for learning."""
        try:
            # This would integrate with the Council's Historian persona
            self.log.info(f"Recording scar tissue: {proposal_id} - {failure_reason}")

            # In a full implementation, this would:
            # 1. Save to a "bad mutations" database
            # 2. Include the failure in future optimization prompts
            # 3. Build a "what NOT to do" memory

        except Exception as e:
            self.log.warning(f"Failed to notify Historian: {e}")

    async def attempt_evolution(self) -> Optional[ApplyResult]:
        """
        Attempt one evolution cycle.

        1. Check if we should evolve
        2. Select a target function
        3. Generate optimization proposal
        4. Test and benchmark
        5. Apply if approved

        Returns ApplyResult if mutation was applied, None otherwise.
        """
        # Read stress
        stress = self._read_stress()

        # Should we evolve?
        should, reason = self._should_evolve(stress)
        if not should:
            self.log.debug(f"Not evolving: {reason}")
            return None

        self.log.info("Stress detected, attempting evolution...")

        # Select target
        target = self._select_target()
        if not target:
            self.log.info("No suitable optimization target found")
            return None

        func, telemetry = target
        self.log.info(f"Selected target: {func.__module__}.{func.__name__}")

        # Generate proposal
        proposal = self.optimizer.propose_optimization(func, telemetry)
        if not proposal:
            self.log.warning("Optimizer failed to generate proposal")
            return None

        # Write files
        proposal = self.optimizer.write_proposal_files(proposal)

        # Evaluate (test + benchmark)
        proposal = self.updater.evaluate_proposal(proposal)

        if not proposal.tests_passed:
            self.log.warning(f"Proposal failed tests: {proposal.error_message}")
            self._record_failure(proposal.proposal_id, proposal.error_message, stress)
            return None

        if proposal.actual_speedup and proposal.actual_speedup < 1.0:
            reason = f"No improvement: {proposal.actual_speedup:.2f}x"
            self.log.warning(reason)
            self._record_failure(proposal.proposal_id, reason, stress)
            return None

        # Approval gate
        if self.policy.require_human_approval:
            self.log.info(f"Proposal {proposal.proposal_id} ready for human approval")

            if self._approval_callback:
                approved = self._approval_callback(proposal)
                if not approved:
                    self.log.info("Human rejected proposal")
                    self._record_failure(proposal.proposal_id, "Rejected by human", stress)
                    return None
            else:
                self.log.info("No approval callback, waiting for manual approval")
                return None

        # Apply!
        result = self.updater.apply_mutation(proposal, force=True)

        if result.success:
            self._record_success(
                proposal.proposal_id,
                proposal.actual_speedup or 1.0,
                stress,
            )
            self.log.info(f"Evolution successful: {proposal.proposal_id}")
        else:
            self._record_failure(proposal.proposal_id, result.message, stress)
            self.log.error(f"Evolution failed: {result.message}")

        return result

    async def monitor_loop(self, interval_seconds: float = 10.0) -> None:
        """
        Continuous monitoring loop.

        Checks stress every interval and triggers evolution when appropriate.
        """
        self.log.info("Antifragility monitor started")

        while True:
            try:
                result = await self.attempt_evolution()

                if result and result.success:
                    self.log.info(f"Applied mutation: {result.message}")

            except Exception as e:
                self.log.error(f"Monitor loop error: {e}")

            await asyncio.sleep(interval_seconds)

    def get_status(self) -> Dict[str, Any]:
        """Get current antifragility status."""
        stress = self._read_stress()
        should_evolve, reason = self._should_evolve(stress)

        return {
            "enabled": ouroboros_enabled(),
            "stress": {
                "pain": stress.pain,
                "entropy": stress.entropy,
                "cpu_load": stress.cpu_load,
            },
            "evolution_ready": should_evolve,
            "evolution_reason": reason,
            "consecutive_failures": self._consecutive_failures,
            "total_attempts": len(self._history),
            "successful_mutations": sum(1 for r in self._history if r.success),
            "applied_mutations": self.updater.get_applied_mutations(),
        }

    def emergency_rollback(self) -> List[ApplyResult]:
        """Emergency: rollback all applied mutations."""
        self.log.warning("EMERGENCY ROLLBACK - reverting all mutations")
        return self.updater.rollback_all()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AntifragilitySystem",
    "StressMetrics",
    "TelemetryCollector",
    "FunctionTelemetry",
]
