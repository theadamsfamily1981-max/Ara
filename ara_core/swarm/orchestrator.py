"""
Swarm Orchestrator - Job Execution Engine
=========================================

Runs jobs through layered agent hierarchy with safety enforcement.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging
import time

from .schema import (
    AgentLayer,
    RiskLevel,
    JobOutcome,
    AgentRun,
    JobFix,
    JobRecord,
    LAYER_CAPABILITIES,
    get_layer_capabilities,
)
from .patterns import Pattern, PatternStep, get_registry, select_pattern
from .stats import save_jobs_to_jsonl


logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Result from an agent execution."""
    success: bool
    output: Any
    cost: float
    latency_ms: float
    error: Optional[str] = None
    needs_escalation: bool = False
    proposed_action: Optional[Dict[str, Any]] = None  # For side-effectful actions


class Agent:
    """Base agent class - override execute() for real implementations."""

    def __init__(self, agent_id: str, layer: AgentLayer, specialty: str):
        self.agent_id = agent_id
        self.layer = layer
        self.specialty = specialty
        self.capabilities = get_layer_capabilities(layer)

    def can_execute(self, action: str) -> bool:
        """Check if this agent can execute an action."""
        action_map = {
            "use_tools": self.capabilities["can_use_tools"],
            "write_files": self.capabilities["can_write_files"],
            "network": self.capabilities["can_network"],
            "spend_money": self.capabilities["can_spend_money"],
            "write_prod_db": self.capabilities["can_write_prod_db"],
        }
        return action_map.get(action, False)

    def execute(self, task: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Execute a task. Override in subclasses."""
        # Default stub implementation
        start = time.time()

        # Simulate some work
        time.sleep(0.01)

        latency_ms = (time.time() - start) * 1000
        cost = 10.0 * (self.layer.value + 1)  # Higher layers cost more

        return AgentResult(
            success=True,
            output={"status": "completed", "agent": self.agent_id},
            cost=cost,
            latency_ms=latency_ms,
        )


class AgentPool:
    """Pool of available agents."""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self._init_default_agents()

    def _init_default_agents(self):
        """Initialize default agent pool."""
        # Create some default agents per layer/specialty
        specialties = ["general", "code", "research", "review", "staticcheck",
                       "analyze", "synthesize", "prepare", "plan", "validate", "approve"]

        for layer in AgentLayer:
            for specialty in specialties:
                agent_id = f"L{layer.value}_{specialty}"
                self.agents[agent_id] = Agent(agent_id, layer, specialty)

    def get(self, agent_id: str) -> Optional[Agent]:
        """Get agent by ID."""
        return self.agents.get(agent_id)

    def find(self, layer: AgentLayer, specialty: str) -> Optional[Agent]:
        """Find an agent by layer and specialty."""
        # Try exact match first
        agent_id = f"L{layer.value}_{specialty}"
        if agent_id in self.agents:
            return self.agents[agent_id]

        # Fall back to general agent at that layer
        fallback_id = f"L{layer.value}_general"
        return self.agents.get(fallback_id)

    def register(self, agent: Agent):
        """Register a new agent."""
        self.agents[agent.agent_id] = agent


class Orchestrator:
    """Main orchestrator for running jobs through the swarm."""

    def __init__(
        self,
        job_log_path: str = "logs/jobs.jsonl",
        max_budget: float = 1000.0,
        require_approval_callback: Optional[Callable[[Dict], bool]] = None,
    ):
        self.agent_pool = AgentPool()
        self.job_log_path = job_log_path
        self.max_budget = max_budget
        self.require_approval_callback = require_approval_callback

        # Current job state
        self.current_job: Optional[JobRecord] = None
        self.current_cost: float = 0.0

    def pick_agent(self, step: PatternStep) -> Optional[Agent]:
        """Select an agent for a step."""
        return self.agent_pool.find(step.layer, step.specialty)

    def check_budget(self, estimated_cost: float) -> bool:
        """Check if we have budget for an operation."""
        return (self.current_cost + estimated_cost) <= self.max_budget

    def escalate_to_higher_layer(
        self,
        current_layer: AgentLayer,
        reason: str,
        context: Dict[str, Any],
    ) -> Optional[AgentResult]:
        """Escalate to next higher layer."""
        if current_layer >= AgentLayer.L3_GOVERNOR:
            # Already at top, need human
            logger.warning(f"Escalation from L3 requires human: {reason}")
            return None

        next_layer = AgentLayer(current_layer.value + 1)
        agent = self.agent_pool.find(next_layer, "review")

        if agent:
            return agent.execute(
                {"action": "review_escalation", "reason": reason},
                context,
            )
        return None

    def request_approval(self, action: Dict[str, Any]) -> bool:
        """Request approval for side-effectful action."""
        if self.require_approval_callback:
            return self.require_approval_callback(action)

        # Default: auto-approve low risk, reject high risk
        risk = action.get("risk", "high")
        return risk == "low"

    def run_job(
        self,
        job_type: str,
        risk: RiskLevel,
        input_data: Optional[Dict[str, Any]] = None,
        pattern_id: Optional[str] = None,
    ) -> JobRecord:
        """Run a job through the swarm hierarchy."""

        # Select pattern
        if pattern_id:
            pattern = get_registry().get(pattern_id)
        else:
            pattern = select_pattern(job_type, risk)

        if not pattern:
            # No pattern found, use minimal fallback
            pattern = Pattern(
                pattern_id="P_fallback",
                job_types=[job_type],
                risk_levels=[risk],
                steps=[
                    PatternStep(AgentLayer.L1_SPECIALIST, "general", "Execute task"),
                    PatternStep(AgentLayer.L2_PLANNER, "review", "Review result"),
                ],
            )

        # Create job record
        job = JobRecord.create(job_type, risk, pattern.pattern_id)
        self.current_job = job
        self.current_cost = 0.0

        context = {
            "job_id": job.job_id,
            "job_type": job_type,
            "risk": risk.value,
            "input": input_data or {},
            "results": [],
        }

        # Execute pattern steps
        for step in pattern.steps:
            agent = self.pick_agent(step)
            if not agent:
                logger.warning(f"No agent for step: {step.description}")
                continue

            # Budget check
            estimated_cost = 10.0 * (step.layer.value + 1)
            if not self.check_budget(estimated_cost):
                logger.warning(f"Budget exceeded at step: {step.description}")
                job.finalize(JobOutcome.ABORTED)
                break

            # Execute step
            result = agent.execute(
                {"step": step.description, "specialty": step.specialty},
                context,
            )

            # Record agent run
            run = AgentRun(
                agent_id=agent.agent_id,
                layer=agent.layer,
                cost=result.cost,
                latency_ms=result.latency_ms,
                outcome=JobOutcome.SUCCESS if result.success else JobOutcome.FAIL,
                error=result.error,
            )
            job.add_agent_run(run)
            self.current_cost += result.cost

            # Update context with result
            context["results"].append({
                "step": step.description,
                "agent": agent.agent_id,
                "output": result.output,
                "success": result.success,
            })

            # Handle failures
            if not result.success:
                if result.needs_escalation:
                    # Try to escalate
                    escalation = self.escalate_to_higher_layer(
                        agent.layer,
                        result.error or "Step failed",
                        context,
                    )
                    if escalation and escalation.success:
                        # Record the fix
                        fix = JobFix(
                            from_layer=AgentLayer(agent.layer.value + 1),
                            to_layer=agent.layer,
                            reason=result.error or "escalation_fix",
                            cost=escalation.cost,
                        )
                        job.add_fix(fix)
                        self.current_cost += escalation.cost
                    else:
                        # Escalation failed
                        if step.required:
                            job.finalize(JobOutcome.ESCALATED)
                            break
                elif step.required:
                    job.finalize(JobOutcome.FAIL)
                    break

            # Handle side-effectful actions
            if result.proposed_action:
                # L0/L1 cannot execute side effects directly
                if agent.layer < AgentLayer.L2_PLANNER:
                    logger.info(f"L{agent.layer.value} proposed action, escalating")
                    continue

                # L2 can propose, L3+ can approve
                if agent.layer == AgentLayer.L2_PLANNER:
                    if not self.request_approval(result.proposed_action):
                        logger.warning(f"Action rejected: {result.proposed_action}")
                        continue

                # Execute the approved action (placeholder)
                logger.info(f"Executing approved action: {result.proposed_action}")

        # Finalize job if not already done
        if job.outcome == JobOutcome.SUCCESS:
            job.finalize(JobOutcome.SUCCESS)

        # Log job
        self._log_job(job)

        # Update pattern stats
        won = job.outcome == JobOutcome.SUCCESS
        get_registry().update_stats(
            pattern.pattern_id,
            won=won,
            cost=job.total_cost,
            latency_ms=job.total_latency_ms,
        )

        self.current_job = None
        return job

    def _log_job(self, job: JobRecord):
        """Log job record to JSONL file."""
        try:
            save_jobs_to_jsonl([job], self.job_log_path)
        except Exception as e:
            logger.error(f"Failed to log job: {e}")


# =============================================================================
# SINGLETON AND CONVENIENCE
# =============================================================================

_orchestrator: Optional[Orchestrator] = None


def get_orchestrator() -> Orchestrator:
    """Get the global orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator


def run_job(
    job_type: str,
    risk: RiskLevel = RiskLevel.LOW,
    input_data: Optional[Dict[str, Any]] = None,
    pattern_id: Optional[str] = None,
) -> JobRecord:
    """Run a job through the swarm."""
    return get_orchestrator().run_job(job_type, risk, input_data, pattern_id)
