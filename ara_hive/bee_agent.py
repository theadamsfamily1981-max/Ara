"""
Bee Agent - Artificial Bee Colony Worker
=========================================

Each node runs one or more BeeAgents that:
1. Register/heartbeat with the waggle board
2. Pick sites based on intensity (employed bee behavior)
3. Occasionally scout for new configurations
4. Execute tasks and update site metrics

The three bee roles:
- EMPLOYED: Exploit known good sites (weighted by intensity)
- ONLOOKER: Watch the waggle dance, follow high-intensity leads
- SCOUT: Explore abandoned sites or new configurations

In practice, each agent cycles through these behaviors probabilistically.
"""

from __future__ import annotations

import random
import time
import threading
import logging
import socket
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .waggle_board import WaggleBoard, Node, Site, Task

logger = logging.getLogger(__name__)


class BeeRole(Enum):
    """Current role of the bee."""
    EMPLOYED = "employed"  # Working a known site
    ONLOOKER = "onlooker"  # Watching dances, picking sites
    SCOUT = "scout"  # Exploring new configurations


@dataclass
class SiteState:
    """Local agent state for a site."""
    loyalty: float = 0.0  # L: commitment to this site
    q_hat: float = 0.0  # Local performance estimate
    visits: int = 0
    consecutive_failures: int = 0


@dataclass
class AgentConfig:
    """Configuration for a bee agent."""
    node_id: str
    role: str = "worker"
    task_types: List[str] = field(default_factory=list)

    # ABC algorithm parameters
    scout_probability: float = 0.1  # 10% chance to scout
    loyalty_increase: float = 0.1  # Loyalty gain on success
    loyalty_decrease: float = 0.2  # Loyalty loss on failure
    abandonment_threshold: int = 5  # Consecutive failures before abandoning
    q_hat_alpha: float = 0.3  # Smoothing factor for Q̂ updates

    # Timing
    heartbeat_interval_sec: float = 10.0
    idle_sleep_sec: float = 1.0


class BeeAgent:
    """
    A bee agent that runs on a single node.

    Implements the Artificial Bee Colony algorithm:
    1. Pick a site (employed/onlooker behavior)
    2. Claim and execute a task
    3. Update site metrics based on reward
    4. Occasionally scout for new configurations
    """

    def __init__(
        self,
        board: WaggleBoard,
        config: AgentConfig,
        task_executor: Optional[Callable[[Task], tuple[bool, float, Any]]] = None,
    ) -> None:
        self.board = board
        self.config = config
        self.task_executor = task_executor or self._default_executor

        # Local state per site
        self.site_states: Dict[int, SiteState] = {}

        # Current role
        self.role = BeeRole.ONLOOKER

        # Runtime state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_heartbeat = 0.0
        self._tasks_completed = 0
        self._tasks_failed = 0

    # =========================================================================
    # Main Loop
    # =========================================================================

    def start(self) -> None:
        """Start the agent in a background thread."""
        if self._running:
            return

        self._running = True
        self._register_node()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"BeeAgent {self.config.node_id} started")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the agent."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info(
            f"BeeAgent {self.config.node_id} stopped "
            f"(completed={self._tasks_completed}, failed={self._tasks_failed})"
        )

    def run_once(self) -> bool:
        """
        Execute one iteration of the bee loop.

        Returns True if work was done, False if idle.
        """
        # Heartbeat if needed
        if time.time() - self._last_heartbeat > self.config.heartbeat_interval_sec:
            self._heartbeat()

        # Decide role for this iteration
        self.role = self._decide_role()

        # Pick a site
        site = self._pick_site()
        if site is None:
            return False

        # Claim a task
        task = self.board.claim_task(
            task_type=site.task_type,
            node_id=self.config.node_id,
            site_id=site.id,
        )
        if task is None:
            return False

        # Update congestion
        self.board.update_site(site.id, congestion_delta=+1)

        # Execute
        start = time.time()
        try:
            success, reward, result = self.task_executor(task)
        except Exception as e:
            logger.exception(f"Task {task.id} failed with exception")
            success, reward, result = False, 0.0, None
            error = str(e)
        else:
            error = None

        duration_ms = (time.time() - start) * 1000

        # Update task
        self.board.complete_task(
            task_id=task.id,
            success=success,
            result=result if isinstance(result, dict) else {"output": result},
            reward=reward,
            error=error,
        )

        # Update site metrics
        self._update_site_metrics(site, success, reward, duration_ms)

        # Update congestion
        self.board.update_site(site.id, congestion_delta=-1)

        # Track stats
        if success:
            self._tasks_completed += 1
        else:
            self._tasks_failed += 1

        return True

    def _run_loop(self) -> None:
        """Main agent loop."""
        while self._running:
            try:
                did_work = self.run_once()
                if not did_work:
                    time.sleep(self.config.idle_sleep_sec)
            except Exception as e:
                logger.exception(f"Agent loop error: {e}")
                time.sleep(self.config.idle_sleep_sec)

    # =========================================================================
    # Node Registration
    # =========================================================================

    def _register_node(self) -> None:
        """Register this node with the waggle board."""
        from .waggle_board import Node

        hostname = socket.gethostname()
        node = Node(
            id=self.config.node_id,
            role=self.config.role,
            hostname=hostname,
            capabilities=self.config.task_types,
            last_heartbeat=time.time(),
            status="online",
        )
        self.board.register_node(node)
        logger.info(f"Registered node {self.config.node_id} ({hostname})")

        # Create sites for each task type
        for task_type in self.config.task_types:
            self.board.create_site(task_type, self.config.node_id)
            logger.info(f"Created site: {task_type}@{self.config.node_id}")

    def _heartbeat(self) -> None:
        """Send heartbeat with system metrics."""
        cpu_load = self._get_cpu_load()
        mem_used = self._get_memory_usage()
        gpu_load = self._get_gpu_load()

        self.board.heartbeat(
            node_id=self.config.node_id,
            cpu_load=cpu_load,
            mem_used_pct=mem_used,
            gpu_load=gpu_load,
        )
        self._last_heartbeat = time.time()

    def _get_cpu_load(self) -> float:
        """Get current CPU load (0-1)."""
        try:
            load1, _, _ = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            return min(load1 / cpu_count, 1.0)
        except (OSError, AttributeError):
            return 0.0

    def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            with open("/proc/meminfo") as f:
                lines = f.readlines()
            mem_total = mem_available = 0
            for line in lines:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1])
            if mem_total > 0:
                return (mem_total - mem_available) / mem_total
        except Exception:
            pass
        return 0.0

    def _get_gpu_load(self) -> float:
        """Get GPU load if available."""
        # Placeholder - implement with nvidia-smi or similar
        return 0.0

    # =========================================================================
    # Site Selection (ABC Algorithm)
    # =========================================================================

    def _decide_role(self) -> BeeRole:
        """Decide role for this iteration."""
        # Scout with some probability
        if random.random() < self.config.scout_probability:
            return BeeRole.SCOUT

        # Otherwise, employed or onlooker based on loyalty
        if self._has_loyal_site():
            return BeeRole.EMPLOYED
        return BeeRole.ONLOOKER

    def _has_loyal_site(self) -> bool:
        """Check if we have a site with high loyalty."""
        for state in self.site_states.values():
            if state.loyalty > 0.5:
                return True
        return False

    def _pick_site(self) -> Optional[Site]:
        """
        Pick a site based on current role.

        - EMPLOYED: Stick with loyal site
        - ONLOOKER: Pick from waggle board weighted by intensity
        - SCOUT: Pick abandoned site or random
        """
        if self.role == BeeRole.EMPLOYED:
            return self._pick_employed_site()
        elif self.role == BeeRole.SCOUT:
            return self._pick_scout_site()
        else:
            return self._pick_onlooker_site()

    def _pick_employed_site(self) -> Optional[Site]:
        """Pick site we're loyal to."""
        # Find highest loyalty site
        best_site_id = None
        best_loyalty = 0.0
        for site_id, state in self.site_states.items():
            if state.loyalty > best_loyalty:
                best_loyalty = state.loyalty
                best_site_id = site_id

        if best_site_id:
            # Get fresh site data from board
            sites = self._get_available_sites()
            for site in sites:
                if site.id == best_site_id:
                    return site

        # Fallback to onlooker behavior
        return self._pick_onlooker_site()

    def _pick_onlooker_site(self) -> Optional[Site]:
        """Pick site weighted by intensity (waggle dance)."""
        sites = self._get_available_sites()
        if not sites:
            return None

        # Weighted random selection by intensity
        total_intensity = sum(max(s.intensity, 0.01) for s in sites)
        r = random.random() * total_intensity

        acc = 0.0
        for site in sites:
            acc += max(site.intensity, 0.01)
            if acc >= r:
                return site

        return sites[-1]  # Fallback

    def _pick_scout_site(self) -> Optional[Site]:
        """Pick abandoned or underexplored site."""
        sites = self._get_available_sites()
        if not sites:
            return None

        # Prefer sites with low visit count or recent failures
        candidates = []
        for site in sites:
            state = self.site_states.get(site.id, SiteState())

            # Abandoned by this agent
            if state.visits > 0 and state.loyalty == 0:
                candidates.append((site, 2.0))
            # Never visited
            elif state.visits == 0:
                candidates.append((site, 1.5))
            # Low intensity (abandoned by hive)
            elif site.intensity < 0.1:
                candidates.append((site, 1.0))
            else:
                candidates.append((site, 0.5))

        if candidates:
            # Weighted selection favoring scout targets
            total = sum(w for _, w in candidates)
            r = random.random() * total
            acc = 0.0
            for site, weight in candidates:
                acc += weight
                if acc >= r:
                    return site

        return random.choice(sites) if sites else None

    def _get_available_sites(self) -> List[Site]:
        """Get all available sites for our task types."""
        all_sites = []
        for task_type in self.config.task_types:
            sites = self.board.get_sites_for_task_type(task_type)
            all_sites.extend(sites)
        return all_sites

    # =========================================================================
    # Site Metrics Update
    # =========================================================================

    def _update_site_metrics(
        self,
        site: Site,
        success: bool,
        reward: float,
        duration_ms: float,
    ) -> None:
        """Update local and global site metrics."""
        # Get or create local state
        state = self.site_states.get(site.id, SiteState())

        # Update Q̂ with exponential smoothing
        alpha = self.config.q_hat_alpha
        state.q_hat = (1 - alpha) * state.q_hat + alpha * reward
        state.visits += 1

        # Update loyalty
        if success and reward > 0:
            state.loyalty = min(state.loyalty + self.config.loyalty_increase, 1.0)
            state.consecutive_failures = 0
        else:
            state.loyalty = max(state.loyalty - self.config.loyalty_decrease, 0.0)
            state.consecutive_failures += 1

        # Check abandonment
        if state.consecutive_failures >= self.config.abandonment_threshold:
            state.loyalty = 0.0
            logger.info(f"Abandoned site {site.id} after {state.consecutive_failures} failures")

        self.site_states[site.id] = state

        # Update waggle board
        # New intensity = Q̂ (performance drives dance strength)
        new_intensity = max(state.q_hat, 0.01)

        self.board.update_site(
            site_id=site.id,
            q_hat=state.q_hat,
            intensity=new_intensity,
            success=success,
            duration_ms=duration_ms,
        )

    # =========================================================================
    # Task Execution
    # =========================================================================

    def _default_executor(self, task: Task) -> tuple[bool, float, Any]:
        """
        Default task executor (stub).

        Override with actual task execution logic.
        Returns: (success, reward, result)
        """
        # Simulate work
        time.sleep(random.uniform(0.1, 0.5))

        # Simulate success/failure
        success = random.random() > 0.1  # 90% success rate

        # Simulate reward
        if success:
            reward = random.uniform(0.5, 1.5)
            result = {"status": "completed", "task_id": task.id}
        else:
            reward = 0.0
            result = {"status": "failed", "task_id": task.id}

        return success, reward, result

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "node_id": self.config.node_id,
            "role": self.role.value,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "sites_tracked": len(self.site_states),
            "loyal_sites": sum(1 for s in self.site_states.values() if s.loyalty > 0.5),
        }


# =============================================================================
# Multi-Agent Runner
# =============================================================================

class HiveNode:
    """
    Runs multiple bee agents on a single node.

    Useful for machines with multiple cores that can handle
    parallel task execution.
    """

    def __init__(
        self,
        board: WaggleBoard,
        node_id: str,
        role: str = "worker",
        task_types: Optional[List[str]] = None,
        num_agents: int = 1,
        task_executor: Optional[Callable] = None,
    ) -> None:
        self.board = board
        self.node_id = node_id
        self.task_types = task_types or ["default"]
        self.num_agents = num_agents
        self.task_executor = task_executor

        self.agents: List[BeeAgent] = []

    def start(self) -> None:
        """Start all agents."""
        for i in range(self.num_agents):
            config = AgentConfig(
                node_id=self.node_id,
                role="worker",
                task_types=self.task_types,
            )
            agent = BeeAgent(
                board=self.board,
                config=config,
                task_executor=self.task_executor,
            )
            agent.start()
            self.agents.append(agent)

        logger.info(f"HiveNode {self.node_id} started {self.num_agents} agents")

    def stop(self) -> None:
        """Stop all agents."""
        for agent in self.agents:
            agent.stop()
        logger.info(f"HiveNode {self.node_id} stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated stats from all agents."""
        total_completed = sum(a._tasks_completed for a in self.agents)
        total_failed = sum(a._tasks_failed for a in self.agents)

        return {
            "node_id": self.node_id,
            "num_agents": len(self.agents),
            "total_completed": total_completed,
            "total_failed": total_failed,
        }
