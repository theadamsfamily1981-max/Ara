"""
Orchestrator
============

Decides which agents run where based on supply and demand.
Maintains running agent state and handles lifecycle.
"""

from __future__ import annotations

import subprocess
import threading
import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple

from ara_soft_kernel.models.supply import SupplyProfile, DeviceInfo
from ara_soft_kernel.models.demand import DemandProfile, Goal
from ara_soft_kernel.models.agent import (
    AgentSpec, AgentInstance, AgentState, AgentHealth,
    AgentPriority, ResourceUsage,
)
from ara_soft_kernel.models.workspace import WorkspaceSpec

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Manages agent lifecycle and placement decisions.

    Thread-safe: state protected by RLock.
    """

    def __init__(self):
        self._lock = threading.RLock()

        # Running agents
        self._agents: Dict[str, AgentInstance] = {}

        # Agent specs (registered but not necessarily running)
        self._specs: Dict[str, AgentSpec] = {}

        # Metrics
        self._spawn_count = 0
        self._stop_count = 0
        self._migration_count = 0

    def register_spec(self, spec: AgentSpec) -> None:
        """Register an agent specification."""
        with self._lock:
            self._specs[spec.name] = spec
        logger.debug(f"Registered agent spec: {spec.name}")

    def get_spec(self, name: str) -> Optional[AgentSpec]:
        """Get agent spec by name."""
        with self._lock:
            return self._specs.get(name)

    def list_specs(self) -> List[AgentSpec]:
        """List all registered agent specs."""
        with self._lock:
            return list(self._specs.values())

    def get_running_agents(self) -> List[AgentInstance]:
        """Get all running agent instances."""
        with self._lock:
            return list(self._agents.values())

    def get_agent(self, instance_id: str) -> Optional[AgentInstance]:
        """Get agent instance by ID."""
        with self._lock:
            return self._agents.get(instance_id)

    def spawn_agent(
        self,
        spec: AgentSpec,
        device_id: str,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> AgentInstance:
        """
        Spawn a new agent instance.

        Returns the instance (state may be PENDING or STARTING).
        """
        instance_id = f"{spec.name}-{uuid.uuid4().hex[:8]}"

        instance = AgentInstance.from_spec(spec, instance_id, device_id)
        instance.state = AgentState.STARTING
        instance.started_at = time.time()

        with self._lock:
            self._agents[instance_id] = instance
            self._spawn_count += 1

        # Actually start the process
        try:
            if spec.command:
                self._start_process(instance, config_overrides or {})
                instance.state = AgentState.RUNNING
                instance.health = AgentHealth.HEALTHY
                logger.info(f"Spawned agent: {instance_id} on {device_id}")
            else:
                # No command = virtual agent (for testing/planning)
                instance.state = AgentState.RUNNING
                logger.info(f"Spawned virtual agent: {instance_id} on {device_id}")
        except Exception as e:
            instance.state = AgentState.FAILED
            instance.error = str(e)
            logger.error(f"Failed to spawn agent {instance_id}: {e}")

        return instance

    def _start_process(
        self,
        instance: AgentInstance,
        config_overrides: Dict[str, Any],
    ) -> None:
        """Start the actual agent process."""
        spec = instance.spec

        # Build environment
        env = dict(spec.env)
        env["ARA_INSTANCE_ID"] = instance.instance_id
        env["ARA_DEVICE_ID"] = instance.device_id

        # Build command
        cmd = [spec.command] + spec.args

        # Start process
        proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=spec.working_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        instance.pid = proc.pid

    def stop_agent(
        self,
        instance_id: str,
        reason: str = "requested",
        force: bool = False,
    ) -> bool:
        """Stop an agent instance."""
        with self._lock:
            instance = self._agents.get(instance_id)
            if not instance:
                logger.warning(f"Agent not found: {instance_id}")
                return False

            instance.state = AgentState.STOPPING

        # Stop the process
        if instance.pid:
            try:
                import signal
                import os
                if force:
                    os.kill(instance.pid, signal.SIGKILL)
                else:
                    os.kill(instance.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # Already dead
            except Exception as e:
                logger.error(f"Failed to stop process {instance.pid}: {e}")

        with self._lock:
            instance.state = AgentState.STOPPED
            instance.stopped_at = time.time()
            self._stop_count += 1

        logger.info(f"Stopped agent: {instance_id} ({reason})")
        return True

    def remove_agent(self, instance_id: str) -> bool:
        """Remove a stopped agent from tracking."""
        with self._lock:
            if instance_id in self._agents:
                instance = self._agents[instance_id]
                if instance.state not in (AgentState.STOPPED, AgentState.FAILED):
                    logger.warning(f"Cannot remove running agent: {instance_id}")
                    return False
                del self._agents[instance_id]
                return True
            return False

    def migrate_agent(
        self,
        instance_id: str,
        to_device: str,
    ) -> bool:
        """Migrate an agent to a new device."""
        with self._lock:
            instance = self._agents.get(instance_id)
            if not instance:
                logger.warning(f"Agent not found: {instance_id}")
                return False

            if not instance.spec.placement.can_migrate:
                logger.warning(f"Agent cannot migrate: {instance_id}")
                return False

            from_device = instance.device_id

        # Stop on old device
        self.stop_agent(instance_id, reason="migration")

        # Spawn on new device
        new_instance = self.spawn_agent(instance.spec, to_device)

        with self._lock:
            self._migration_count += 1

        logger.info(f"Migrated agent {instance_id} from {from_device} to {to_device}")
        return True

    def health_check(self, instance_id: str) -> AgentHealth:
        """Check health of an agent."""
        with self._lock:
            instance = self._agents.get(instance_id)
            if not instance:
                return AgentHealth.UNKNOWN

            # Check if process is alive
            if instance.pid:
                try:
                    import os
                    os.kill(instance.pid, 0)
                    instance.health = AgentHealth.HEALTHY
                except ProcessLookupError:
                    instance.health = AgentHealth.UNHEALTHY
                    instance.state = AgentState.FAILED

            instance.last_health_check = time.time()
            return instance.health

    def update_resource_usage(
        self,
        instance_id: str,
        usage: ResourceUsage,
    ) -> None:
        """Update resource usage for an agent."""
        with self._lock:
            instance = self._agents.get(instance_id)
            if instance:
                instance.resource_usage = usage

    def plan_from_goals(
        self,
        goals: List[Goal],
        supply: SupplyProfile,
        demand: DemandProfile,
    ) -> Tuple[List[AgentSpec], List[WorkspaceSpec]]:
        """
        Plan which agents/workspaces are needed for goals.

        This is a simple planner - real implementation would be more sophisticated.
        """
        needed_agents: List[AgentSpec] = []
        needed_workspaces: List[WorkspaceSpec] = []

        for goal in goals:
            if not goal.active:
                continue

            # Simple heuristic: map capabilities to agents
            for cap in goal.required_capabilities:
                spec = self._find_agent_for_capability(cap)
                if spec and spec.name not in [a.name for a in needed_agents]:
                    needed_agents.append(spec)

        return needed_agents, needed_workspaces

    def _find_agent_for_capability(self, capability: str) -> Optional[AgentSpec]:
        """Find an agent that provides a capability."""
        # Simple mapping - could be more sophisticated
        capability_map = {
            "gpu_compute": "lightfield_renderer",
            "opengl_4.5": "lightfield_renderer",
            "network": "sync_agent",
            "search": "searcher",
            "code": "coder",
        }

        agent_name = capability_map.get(capability)
        if agent_name:
            return self._specs.get(agent_name)
        return None

    def diff_agents(
        self,
        desired: List[AgentSpec],
    ) -> Tuple[List[AgentSpec], List[AgentInstance], List[Tuple[AgentInstance, str]]]:
        """
        Diff current running agents against desired agents.

        Returns: (to_start, to_stop, to_migrate)
        """
        with self._lock:
            running_names = {a.agent_name for a in self._agents.values() if a.is_running()}
            desired_names = {s.name for s in desired}

            to_start = [s for s in desired if s.name not in running_names]
            to_stop = [a for a in self._agents.values() if a.agent_name not in desired_names and a.is_running()]
            to_migrate: List[Tuple[AgentInstance, str]] = []  # (instance, new_device)

            return to_start, to_stop, to_migrate

    def select_device(
        self,
        spec: AgentSpec,
        supply: SupplyProfile,
    ) -> Optional[str]:
        """Select best device for an agent."""
        candidates = []

        for device in supply.get_online_devices():
            # Check device type
            if spec.placement.required_device_types:
                if device.type not in spec.placement.required_device_types:
                    continue

            if spec.placement.excluded_device_types:
                if device.type in spec.placement.excluded_device_types:
                    continue

            # Check resources
            if device.memory_free_gb < spec.resources.memory_gb:
                continue

            if spec.requires_gpu():
                if not device.has_gpu():
                    continue
                if device.total_gpu_memory_free() < spec.resources.gpu_mem_gb:
                    continue

            # Score device
            score = 0.0

            # Prefer GPU-rich for GPU agents
            if spec.placement.preferred == "gpu_rich" and device.has_gpu():
                score += device.total_gpu_memory_free() * 10

            # Prefer CPU-rich for CPU agents
            if spec.placement.preferred == "cpu_rich":
                score += device.cpu_cores_available * 5

            # Prefer lower latency
            if spec.placement.preferred == "low_latency":
                if device.network.latency_ms_to_hive:
                    score -= device.network.latency_ms_to_hive

            # Prefer more free memory
            score += device.memory_free_gb

            # Penalize low battery
            if device.battery is not None and device.battery < 0.3:
                score -= 50

            candidates.append((device.id, score))

        if not candidates:
            return None

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        with self._lock:
            running = [a for a in self._agents.values() if a.is_running()]
            return {
                "total_agents": len(self._agents),
                "running_agents": len(running),
                "registered_specs": len(self._specs),
                "spawn_count": self._spawn_count,
                "stop_count": self._stop_count,
                "migration_count": self._migration_count,
            }
