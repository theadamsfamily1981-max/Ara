#!/usr/bin/env python3
"""
CADD Orchestrator - High-Risk Job Safety Gate
==============================================

Provides gated execution for high-risk operations (e.g., drug synthesis).
Jobs only run if the swarm passes CADD health checks.

Safety Protocol:
    1. Update agent associations during research phase
    2. Call sentinel.tick() as predicate
    3. Check for MONOCULTURE alerts
    4. If healthy → execute job
    5. If unhealthy → BLOCK + inject diversity

Integration:
    - Uses CADDSentinel for health monitoring
    - Provides run_code_fn injection for sandboxed execution
    - Supports automatic diversity injection on monoculture

Usage:
    from ara_core.cadd import CADDSentinel
    from ara_core.cadd.orchestrator import DrugSynthesisOrchestrator

    sentinel = CADDSentinel()
    orch = DrugSynthesisOrchestrator(sentinel, run_code_fn=sandbox.run)

    # Simulate research (updates associations)
    orch.simulate_research_update("protein_X", "binding_A", 0.9)

    # Attempt synthesis (gated by sentinel health)
    success = orch.run_synthesis_job("job_1", "synthesis_code")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Iterable, Optional, Any

from .sentinel import CADDSentinel, DriftType, DriftAlert


@dataclass
class CodeSnippetConfig:
    """
    Configuration for sandboxed code execution.

    Note: Actual chemistry/synthesis code execution happens in an
    external sandboxed environment. This config controls the gate.
    """
    enabled: bool = True
    allowed_binaries: List[str] = field(default_factory=lambda: ["python3"])
    timeout_seconds: float = 300.0
    max_memory_mb: int = 4096
    audit_log: bool = True


@dataclass
class JobResult:
    """Result of a synthesis job attempt."""
    job_name: str
    executed: bool
    blocked_reason: Optional[str] = None
    alerts_triggered: List[DriftAlert] = field(default_factory=list)
    health_status: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    execution_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_name": self.job_name,
            "executed": self.executed,
            "blocked_reason": self.blocked_reason,
            "alerts": [a.to_dict() for a in self.alerts_triggered],
            "health": self.health_status,
            "timestamp": self.timestamp,
            "execution_time_ms": self.execution_time_ms,
        }


class DrugSynthesisOrchestrator:
    """
    High-risk job orchestrator with CADD safety gating.

    The orchestrator:
    1. Updates CADD agent associations during 'research' phase
    2. Calls sentinel.tick() as a health predicate
    3. Only runs jobs if swarm passes health checks
    4. Injects diversity agents on monoculture detection
    """

    def __init__(
        self,
        sentinel: CADDSentinel,
        agent_ids: Iterable[str] | None = None,
        run_code_fn: Callable[[str], None] | None = None,
        snippet_config: CodeSnippetConfig | None = None,
        verbose: bool = True,
    ):
        """
        Initialize the orchestrator.

        Args:
            sentinel: CADD sentinel for health monitoring
            agent_ids: Initial agent IDs (default: Agent_A through Agent_D)
            run_code_fn: Injected code runner (sandboxed execution)
            snippet_config: Code execution configuration
            verbose: Print status messages
        """
        self.sentinel = sentinel
        self.runner_cfg = snippet_config or CodeSnippetConfig()
        self.run_code_fn = run_code_fn
        self.verbose = verbose

        # Initialize agent pool
        if agent_ids is not None:
            self.agent_ids: List[str] = list(agent_ids)
        else:
            self.agent_ids = ["Agent_A", "Agent_B", "Agent_C", "Agent_D"]

        for agent_id in self.agent_ids:
            self.sentinel.register_agent(agent_id)

        # Job history
        self.job_history: List[JobResult] = []
        self.total_jobs_attempted: int = 0
        self.total_jobs_blocked: int = 0

        if self.verbose:
            print(f"DrugSynthesisOrchestrator initialized")
            print(f"  Agents: {self.agent_ids}")
            print(f"  Code execution: {'enabled' if self.runner_cfg.enabled else 'disabled'}")

    # ------------------------------------------------------------------
    # Research Simulation / Association Updates
    # ------------------------------------------------------------------

    def simulate_research_update(
        self,
        concept: str,
        signal: str,
        bias_strength: float,
    ) -> None:
        """
        Simulate a research update that affects agent associations.

        This demo implementation creates a bias pattern where:
        - First half of agents get strong associations (dominant cluster)
        - Second half get weak associations (under-represented)

        In a real system, this would be called by actual research/learning
        code updating the swarm's belief structures.

        Args:
            concept: Target concept (e.g., "protein_X")
            signal: Signal/feature (e.g., "binding_affinity")
            bias_strength: Base strength of association
        """
        n = len(self.agent_ids)
        for i, agent_id in enumerate(self.agent_ids):
            if i < n // 2:
                # Dominant cluster: strong associations
                strength = bias_strength * 0.9
            else:
                # Under-represented cluster: weak associations
                strength = bias_strength * 0.2

            self.sentinel.update_association(agent_id, concept, signal, strength)

    def update_agent_association(
        self,
        agent_id: str,
        concept: str,
        signal: str,
        strength: float,
    ) -> None:
        """
        Direct association update for a specific agent.

        Args:
            agent_id: Agent to update
            concept: Target concept
            signal: Signal/feature
            strength: Association strength
        """
        if agent_id not in self.agent_ids:
            self.agent_ids.append(agent_id)
        self.sentinel.update_association(agent_id, concept, signal, strength)

    # ------------------------------------------------------------------
    # Diversity Injection
    # ------------------------------------------------------------------

    def inject_diversity(self, count: int) -> List[str]:
        """
        Inject diverse agents to counter monoculture.

        Creates new agents with no prior associations (orthogonal priors).
        In a real system, this might:
        - Re-seed agents with different priors
        - Pull in external tools with different inductive biases
        - Change model families or hyperparameters

        Args:
            count: Number of agents to inject

        Returns:
            List of new agent IDs
        """
        base_index = len(self.agent_ids)
        new_agents = []

        for i in range(count):
            new_id = f"Diverse_{base_index + i}"
            if new_id not in self.agent_ids:
                self.agent_ids.append(new_id)
                self.sentinel.register_agent(new_id)
                new_agents.append(new_id)

        if self.verbose and new_agents:
            print(f"[CADD META] Injected {len(new_agents)} diverse agents: {new_agents}")

        return new_agents

    def _handle_monoculture_alert(self, alert: DriftAlert) -> None:
        """Handle a monoculture alert by injecting diversity."""
        count = self.sentinel.config.diversity_spawn_count
        self.inject_diversity(count)

        if self.verbose:
            print(f"[CADD ACTION] Diversity injection triggered")
            print(f"   Alert: {alert.message}")
            print(f"   Action: Spawned {count} orthogonal agents")

    # ------------------------------------------------------------------
    # Job Execution with Safety Gate
    # ------------------------------------------------------------------

    def run_synthesis_job(
        self,
        job_name: str,
        synthesis_code: str,
        force: bool = False,
    ) -> JobResult:
        """
        Run a synthesis job with CADD safety gating.

        Safety Protocol:
        1. Run sentinel tick to check health
        2. If MONOCULTURE alert → BLOCK + inject diversity
        3. If healthy → execute job (if runner configured)

        Args:
            job_name: Name of the job for logging
            synthesis_code: Code to execute (passed to run_code_fn)
            force: Force execution even with alerts (use with caution!)

        Returns:
            JobResult with execution status and health info
        """
        self.total_jobs_attempted += 1
        start_time = time.time()

        # Run sentinel tick
        alerts = self.sentinel.tick()
        status = self.sentinel.health_status()

        if self.verbose:
            print(f"\n[CADD] Job '{job_name}' - tick={status['total_ticks']}")
            print(f"  H_influence={status['h_influence']:.3f} "
                  f"(min={self.sentinel.config.h_influence_min:.3f})")

        # Check for monoculture
        monoculture_alerts = [
            a for a in alerts if a.drift_type == DriftType.MONOCULTURE
        ]

        if monoculture_alerts and not force:
            # BLOCK execution
            self.total_jobs_blocked += 1

            if self.verbose:
                print(f"\n[CRITICAL STOP] Job '{job_name}' BLOCKED - monoculture detected")

            for alert in monoculture_alerts:
                if self.verbose:
                    print(f"  -> {alert.message}")
                self._handle_monoculture_alert(alert)

            return JobResult(
                job_name=job_name,
                executed=False,
                blocked_reason="MONOCULTURE_DETECTED",
                alerts_triggered=monoculture_alerts,
                health_status=status,
            )

        # Check if execution is enabled
        if not self.runner_cfg.enabled:
            if self.verbose:
                print(f"[SKIP] Code execution disabled (job='{job_name}')")

            return JobResult(
                job_name=job_name,
                executed=False,
                blocked_reason="EXECUTION_DISABLED",
                alerts_triggered=alerts,
                health_status=status,
            )

        # Check if runner is configured
        if self.run_code_fn is None:
            if self.verbose:
                print(f"[DRY-RUN] Would execute '{job_name}' (no runner configured)")

            return JobResult(
                job_name=job_name,
                executed=False,
                blocked_reason="NO_RUNNER_CONFIGURED",
                alerts_triggered=alerts,
                health_status=status,
            )

        # SAFETY PASS - Execute job
        if self.verbose:
            print(f"\n[SAFETY PASS] Executing job '{job_name}'...")

        try:
            self.run_code_fn(synthesis_code)
            execution_time = (time.time() - start_time) * 1000

            return JobResult(
                job_name=job_name,
                executed=True,
                alerts_triggered=alerts,
                health_status=status,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            return JobResult(
                job_name=job_name,
                executed=False,
                blocked_reason=f"EXECUTION_ERROR: {str(e)}",
                alerts_triggered=alerts,
                health_status=status,
            )

    # ------------------------------------------------------------------
    # Status and Reporting
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "n_agents": len(self.agent_ids),
            "total_jobs_attempted": self.total_jobs_attempted,
            "total_jobs_blocked": self.total_jobs_blocked,
            "block_rate": (
                self.total_jobs_blocked / self.total_jobs_attempted
                if self.total_jobs_attempted > 0 else 0.0
            ),
            "sentinel_health": self.sentinel.health_status(),
        }

    def status_string(self) -> str:
        """Get status string for logging/dashboard."""
        stats = self.get_stats()
        health = stats["sentinel_health"]

        if health.get("h_influence_target"):
            status = "🟢 HEALTHY"
        elif health.get("h_influence_ok"):
            status = "🟡 CAUTION"
        else:
            status = "🔴 MONOCULTURE"

        return (
            f"[CADD Orchestrator] {status}\n"
            f"  Agents: {stats['n_agents']}\n"
            f"  Jobs: {stats['total_jobs_attempted']} attempted, "
            f"{stats['total_jobs_blocked']} blocked ({stats['block_rate']:.1%})\n"
            f"  H_influence: {health['h_influence']:.3f}"
        )
