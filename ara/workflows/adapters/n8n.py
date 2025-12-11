# ara/workflows/adapters/n8n.py
"""
n8n Workflow Engine Adapter
===========================

Connects Ara to n8n (https://n8n.io) via HTTP API.

n8n provides:
    - Visual workflow builder
    - 400+ integrations
    - Webhook triggers
    - Self-hosted option

Ara provides:
    - Decision-making (which path to take?)
    - User guidance (what to tell them?)
    - Learning (outcomes → MEIS/QUANTA)

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                 AraSelfGuidedOrchestrator                │
    │                        │                                 │
    │                        ▼                                 │
    │                    N8NAdapter                            │
    │                        │                                 │
    │          ┌─────────────┴─────────────┐                  │
    │          ▼                           ▼                  │
    │   Workflow API              Webhook API                  │
    │   (list, execute)           (trigger, receive)           │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                      n8n Server                          │
    │   (executes workflows, manages credentials)              │
    └─────────────────────────────────────────────────────────┘

Usage:
    from ara.workflows.adapters.n8n import N8NAdapter

    adapter = N8NAdapter(
        base_url="http://localhost:5678",
        api_key="your-n8n-api-key",
    )

    # Ara orchestrator uses this adapter
    orchestrator = AraSelfGuidedOrchestrator(engine=adapter)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import (
    BaseEngineAdapter,
    StepDefinition,
    StepResult,
    StepStatus,
    StepType,
    WorkflowDefinition,
    WorkflowExecution,
    WorkflowStatus,
)

log = logging.getLogger("Ara.Workflows.N8N")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class N8NConfig:
    """Configuration for n8n connection."""
    base_url: str = "http://localhost:5678"
    api_key: Optional[str] = None

    # Webhook settings
    webhook_base_url: Optional[str] = None  # For receiving callbacks

    # Timeouts
    request_timeout: int = 30
    execution_timeout: int = 300

    # Polling
    poll_interval_seconds: float = 1.0
    max_poll_attempts: int = 300


# =============================================================================
# N8N API Types
# =============================================================================

@dataclass
class N8NWorkflow:
    """n8n workflow as returned by API."""
    id: str
    name: str
    active: bool
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    connections: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "N8NWorkflow":
        return cls(
            id=str(data.get("id", "")),
            name=data.get("name", ""),
            active=data.get("active", False),
            nodes=data.get("nodes", []),
            connections=data.get("connections", {}),
            settings=data.get("settings", {}),
            tags=data.get("tags", []),
        )

    def to_workflow_definition(self) -> WorkflowDefinition:
        """Convert n8n workflow to Ara's WorkflowDefinition."""
        steps = []

        for node in self.nodes:
            step_type = self._infer_step_type(node)
            steps.append(StepDefinition(
                id=node.get("name", str(uuid.uuid4())),
                name=node.get("name", "Unknown"),
                type=step_type,
                description=node.get("notes", ""),
                meta={
                    "n8n_type": node.get("type", ""),
                    "n8n_params": node.get("parameters", {}),
                    "n8n_position": node.get("position", []),
                },
            ))

        return WorkflowDefinition(
            id=self.id,
            name=self.name,
            description=f"n8n workflow: {self.name}",
            steps=steps,
            meta={
                "source": "n8n",
                "active": self.active,
                "tags": self.tags,
            },
        )

    def _infer_step_type(self, node: Dict[str, Any]) -> StepType:
        """Infer step type from n8n node type."""
        n8n_type = node.get("type", "").lower()

        if "trigger" in n8n_type or "webhook" in n8n_type:
            return StepType.SIGNAL
        elif "if" in n8n_type or "switch" in n8n_type:
            return StepType.DECISION
        elif "wait" in n8n_type or "delay" in n8n_type:
            return StepType.TIMER
        elif "form" in n8n_type or "respondtowebhook" in n8n_type:
            return StepType.HUMAN
        elif "executeworkflow" in n8n_type:
            return StepType.SUBWORKFLOW
        else:
            return StepType.ACTIVITY


@dataclass
class N8NExecution:
    """n8n execution as returned by API."""
    id: str
    workflow_id: str
    finished: bool
    mode: str
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    status: str = "unknown"
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> "N8NExecution":
        return cls(
            id=str(data.get("id", "")),
            workflow_id=str(data.get("workflowId", "")),
            finished=data.get("finished", False),
            mode=data.get("mode", "manual"),
            status=data.get("status", "unknown"),
            data=data.get("data", {}),
        )


# =============================================================================
# N8N Adapter
# =============================================================================

class N8NAdapter(BaseEngineAdapter):
    """
    Adapter that connects Ara to n8n workflow engine via HTTP API.

    This is a simulation/mock for development.
    In production, replace HTTP calls with actual aiohttp requests.
    """

    def __init__(
        self,
        config: Optional[N8NConfig] = None,
    ):
        super().__init__(name="n8n")
        self.config = config or N8NConfig()

        # Simulation: cached n8n workflows
        self._n8n_workflows: Dict[str, N8NWorkflow] = {}
        self._n8n_executions: Dict[str, N8NExecution] = {}

        log.info(
            "N8NAdapter initialized: base_url=%s",
            self.config.base_url,
        )

    # =========================================================================
    # n8n API Simulation
    # =========================================================================

    async def _api_get(self, endpoint: str) -> Dict[str, Any]:
        """Simulate GET request to n8n API."""
        # In production: use aiohttp
        log.debug("API GET: %s%s", self.config.base_url, endpoint)
        return {}

    async def _api_post(
        self,
        endpoint: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simulate POST request to n8n API."""
        log.debug("API POST: %s%s", self.config.base_url, endpoint)
        return {}

    # =========================================================================
    # EngineClient Implementation
    # =========================================================================

    async def sync_workflows(self) -> int:
        """
        Sync workflow definitions from n8n.

        In production: calls GET /api/v1/workflows
        Returns number of workflows synced.
        """
        # Simulate API call
        # In production:
        # response = await self._api_get("/api/v1/workflows")
        # workflows = response.get("data", [])

        # For simulation, use registered workflows
        count = 0
        for workflow_id, workflow in self._n8n_workflows.items():
            workflow_def = workflow.to_workflow_definition()
            self._workflows[workflow_id] = workflow_def
            count += 1

        log.info("Synced %d workflows from n8n", count)
        return count

    async def start_workflow(
        self,
        workflow_id: str,
        initial_state: Dict[str, Any],
    ) -> WorkflowExecution:
        """
        Start a new workflow execution.

        In production: calls POST /api/v1/workflows/{id}/execute
        """
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            # Try to find in n8n cache
            n8n_workflow = self._n8n_workflows.get(workflow_id)
            if n8n_workflow:
                workflow = n8n_workflow.to_workflow_definition()
                self._workflows[workflow_id] = workflow
            else:
                raise ValueError(f"Workflow not found: {workflow_id}")

        execution_id = f"n8n-{uuid.uuid4().hex[:8]}"

        # Simulate n8n execution
        n8n_exec = N8NExecution(
            id=execution_id,
            workflow_id=workflow_id,
            finished=False,
            mode="manual",
            started_at=datetime.utcnow(),
            status="running",
            data={"inputData": initial_state},
        )
        self._n8n_executions[execution_id] = n8n_exec

        # Create Ara execution wrapper
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            execution_id=execution_id,
            status=WorkflowStatus.RUNNING,
            state=initial_state.copy(),
            started_at=datetime.utcnow(),
        )
        self._executions[execution_id] = execution

        log.info(
            "Started n8n workflow: %s (execution=%s)",
            workflow_id, execution_id,
        )

        return execution

    async def run_step(
        self,
        execution_id: str,
        step_id: str,
        state: Dict[str, Any],
    ) -> StepResult:
        """
        Execute a single node in the workflow.

        Note: n8n typically runs entire workflows at once.
        This method simulates step-by-step execution for Ara's control.

        In production: might use n8n's execution API with breakpoints.
        """
        execution = self._executions.get(execution_id)
        if not execution:
            raise ValueError(f"Execution not found: {execution_id}")

        workflow = self._workflows.get(execution.workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {execution.workflow_id}")

        step = workflow.get_step(step_id)
        if not step:
            raise ValueError(f"Step not found: {step_id}")

        result = StepResult(
            step_id=step_id,
            status=StepStatus.RUNNING,
            started_at=datetime.utcnow(),
        )

        execution.current_step = step_id

        try:
            # Simulate n8n node execution based on type
            n8n_type = step.meta.get("n8n_type", "")

            if step.type == StepType.ACTIVITY:
                output = await self._execute_n8n_node(step, state)
                result.output = output
                result.status = StepStatus.COMPLETED

            elif step.type == StepType.DECISION:
                branch = await self._evaluate_n8n_if(step, state)
                result.branch_taken = branch
                result.status = StepStatus.COMPLETED

            elif step.type == StepType.HUMAN:
                # n8n form or webhook response
                result.output = {"awaiting_input": True}
                result.status = StepStatus.COMPLETED

            elif step.type == StepType.TIMER:
                # n8n Wait node
                await asyncio.sleep(0.1)  # Simulate brief wait
                result.status = StepStatus.COMPLETED

            else:
                result.status = StepStatus.COMPLETED

        except Exception as e:
            log.error("n8n step %s failed: %s", step_id, e)
            result.status = StepStatus.FAILED
            result.error = str(e)

        result.completed_at = datetime.utcnow()

        # Update execution
        if result.success:
            execution.completed_steps.append(step_id)
            if result.output and isinstance(result.output, dict):
                execution.state.update(result.output)

        execution.step_results.append(result)
        execution.current_step = None

        log.info(
            "n8n step %s completed: status=%s",
            step_id, result.status.value,
        )

        return result

    async def cancel_execution(
        self,
        execution_id: str,
        reason: str = "",
    ) -> bool:
        """Cancel a running n8n execution."""
        # In production: calls DELETE /api/v1/executions/{id}

        if execution_id in self._n8n_executions:
            self._n8n_executions[execution_id].finished = True
            self._n8n_executions[execution_id].status = "cancelled"

        if execution_id in self._executions:
            self._executions[execution_id].status = WorkflowStatus.CANCELLED
            self._executions[execution_id].completed_at = datetime.utcnow()
            self._executions[execution_id].error = reason

        log.info("Cancelled n8n execution: %s", execution_id)
        return True

    async def send_signal(
        self,
        execution_id: str,
        signal_name: str,
        payload: Any = None,
    ) -> bool:
        """
        Send data to a waiting n8n workflow.

        In production: might use webhook or n8n's REST API.
        """
        log.info("Signal to n8n: %s -> %s", signal_name, execution_id)

        # Store for waiting steps
        execution = self._executions.get(execution_id)
        if execution:
            execution.state[f"signal_{signal_name}"] = payload

        return True

    # =========================================================================
    # n8n Node Execution (Simulation)
    # =========================================================================

    async def _execute_n8n_node(
        self,
        step: StepDefinition,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Simulate execution of an n8n node.

        In production: n8n handles this internally.
        """
        n8n_type = step.meta.get("n8n_type", "").lower()
        params = step.meta.get("n8n_params", {})

        # Simulate common node types
        if "http" in n8n_type:
            return {"http_response": "simulated", "status": 200}
        elif "set" in n8n_type:
            # Set node - copy values
            return params.get("values", {})
        elif "code" in n8n_type:
            return {"code_executed": True}
        elif "slack" in n8n_type or "email" in n8n_type:
            return {"message_sent": True}
        else:
            return {"executed": True, "node_type": n8n_type}

    async def _evaluate_n8n_if(
        self,
        step: StepDefinition,
        state: Dict[str, Any],
    ) -> str:
        """Evaluate n8n IF node condition."""
        params = step.meta.get("n8n_params", {})
        conditions = params.get("conditions", {})

        # Simplified: just return "true" or "false" branch
        # In production: evaluate actual conditions
        return "true"

    # =========================================================================
    # n8n Workflow Registration
    # =========================================================================

    def register_n8n_workflow(self, workflow_data: Dict[str, Any]) -> N8NWorkflow:
        """
        Register an n8n workflow from JSON export.

        Usage:
            with open("my_workflow.json") as f:
                data = json.load(f)
            adapter.register_n8n_workflow(data)
        """
        workflow = N8NWorkflow.from_api(workflow_data)
        self._n8n_workflows[workflow.id] = workflow

        # Also register as Ara workflow
        workflow_def = workflow.to_workflow_definition()
        self._workflows[workflow.id] = workflow_def

        log.info("Registered n8n workflow: %s (%s)", workflow.name, workflow.id)
        return workflow


# =============================================================================
# Convenience
# =============================================================================

_default_adapter: Optional[N8NAdapter] = None


def get_n8n_adapter(config: Optional[N8NConfig] = None) -> N8NAdapter:
    """Get the default n8n adapter."""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = N8NAdapter(config)
    return _default_adapter


__all__ = [
    "N8NConfig",
    "N8NAdapter",
    "N8NWorkflow",
    "N8NExecution",
    "get_n8n_adapter",
]
