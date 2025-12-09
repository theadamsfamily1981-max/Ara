"""
Ara Hive Node
==============

A runnable service that:
1. Loads hive configuration
2. Starts the pheromone store
3. Spawns configured agents
4. Runs the tick loop
5. Writes status for observability

Usage:
    python -m ara.hive.node --config config/hive_config.yaml

Or as a module:
    from ara.hive.node import HiveNode
    node = HiveNode()
    node.run()
"""

from __future__ import annotations

import os
import sys
import json
import time
import signal
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import yaml

from .store import PheromoneStore
from .queen import QueenHiveAdapter
from .agent import HiveAgent, GenericWorker, WorkResult
from .pheromones import PheromoneKind

logger = logging.getLogger(__name__)

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "hive_config.yaml"


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class RoleConfig:
    """Configuration for an agent role."""
    name: str
    description: str
    tools: List[str]
    action_class: str
    max_instances: int
    can_emit_priority: bool = False


@dataclass
class HiveConfig:
    """Full hive configuration."""
    node_id: str
    roles: Dict[str, RoleConfig]
    max_total_agents: int
    agent_tick_interval_ms: int
    status_file: Optional[str]
    status_interval_seconds: int
    log_level: str
    mesh_enabled: bool
    mesh_transport: str
    mesh_settings: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path) -> HiveConfig:
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Parse roles
        roles = {}
        for name, cfg in data.get("roles", {}).items():
            roles[name] = RoleConfig(
                name=name,
                description=cfg.get("description", ""),
                tools=cfg.get("tools", []),
                action_class=cfg.get("action_class", "A"),
                max_instances=cfg.get("max_instances", 1),
                can_emit_priority=cfg.get("can_emit_priority", False),
            )

        # Parse mesh
        mesh = data.get("mesh", {})

        return cls(
            node_id=mesh.get("node_id", f"node-{os.getpid()}"),
            roles=roles,
            max_total_agents=data.get("resources", {}).get("max_total_agents", 20),
            agent_tick_interval_ms=data.get("resources", {}).get("agent_tick_interval_ms", 1000),
            status_file=data.get("observability", {}).get("status_file"),
            status_interval_seconds=data.get("observability", {}).get("status_interval_seconds", 10),
            log_level=data.get("observability", {}).get("log_level", "INFO"),
            mesh_enabled=mesh.get("enabled", False),
            mesh_transport=mesh.get("transport", "local"),
            mesh_settings=mesh.get("settings", {}),
        )


# =============================================================================
# Hive Node
# =============================================================================

class HiveNode:
    """
    A runnable hive node.

    Manages:
    - PheromoneStore
    - Agent instances
    - Status reporting
    - Graceful shutdown
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[HiveConfig] = None,
    ):
        """
        Initialize hive node.

        Args:
            config_path: Path to hive_config.yaml
            config: Pre-loaded config (overrides config_path)
        """
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

        if config:
            self.config = config
        else:
            self.config = HiveConfig.from_yaml(self.config_path)

        # Core components
        self.store = PheromoneStore()
        self.queen_adapter = QueenHiveAdapter(self.store)
        self.agents: List[HiveAgent] = []

        # Runtime state
        self._running = False
        self._shutdown_event = threading.Event()
        self._tick_thread: Optional[threading.Thread] = None
        self._status_thread: Optional[threading.Thread] = None

        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config.log_level, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self):
        """Start the hive node."""
        if self._running:
            logger.warning("Hive node already running")
            return

        logger.info(f"Starting hive node: {self.config.node_id}")

        # Spawn initial agents
        self._spawn_initial_agents()

        # Set initial global mode
        self.queen_adapter.set_global_mode("NORMAL")

        # Start tick loop
        self._running = True
        self._tick_thread = threading.Thread(target=self._tick_loop, daemon=True)
        self._tick_thread.start()

        # Start status writer
        if self.config.status_file:
            self._status_thread = threading.Thread(target=self._status_loop, daemon=True)
            self._status_thread.start()

        logger.info(f"Hive node started with {len(self.agents)} agents")

    def stop(self):
        """Stop the hive node gracefully."""
        if not self._running:
            return

        logger.info("Stopping hive node...")
        self._running = False
        self._shutdown_event.set()

        if self._tick_thread:
            self._tick_thread.join(timeout=5)

        if self._status_thread:
            self._status_thread.join(timeout=2)

        # Write final status
        self._write_status()
        logger.info("Hive node stopped")

    def run(self):
        """Run the hive node (blocking)."""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

        self.start()

        # Block until shutdown
        try:
            while self._running:
                self._shutdown_event.wait(timeout=1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False
        self._shutdown_event.set()

    # =========================================================================
    # Agent Management
    # =========================================================================

    def _spawn_initial_agents(self):
        """Spawn agents based on configuration."""
        agent_count = 0

        for role_name, role_cfg in self.config.roles.items():
            # Spawn up to max_instances for each role
            # For v0.1, start with 1 each
            instances_to_spawn = min(1, role_cfg.max_instances)

            for i in range(instances_to_spawn):
                if agent_count >= self.config.max_total_agents:
                    logger.warning("Max agent limit reached")
                    return

                agent_id = f"{role_name}_{i:02d}@{self.config.node_id}"
                agent = GenericWorker(
                    agent_id=agent_id,
                    role=role_name,
                    store=self.store,
                    tools={},  # Tools would be injected here
                )
                self.agents.append(agent)
                agent_count += 1
                logger.debug(f"Spawned agent: {agent_id}")

    def spawn_agent(self, role: str) -> Optional[HiveAgent]:
        """Spawn a new agent of the given role."""
        if role not in self.config.roles:
            logger.error(f"Unknown role: {role}")
            return None

        role_cfg = self.config.roles[role]

        # Check limits
        current_count = len([a for a in self.agents if a.role == role])
        if current_count >= role_cfg.max_instances:
            logger.warning(f"Max instances for {role} reached")
            return None

        if len(self.agents) >= self.config.max_total_agents:
            logger.warning("Max total agents reached")
            return None

        agent_id = f"{role}_{current_count:02d}@{self.config.node_id}"
        agent = GenericWorker(
            agent_id=agent_id,
            role=role,
            store=self.store,
            tools={},
        )
        self.agents.append(agent)
        logger.info(f"Spawned agent: {agent_id}")
        return agent

    def remove_agent(self, agent_id: str):
        """Remove an agent by ID."""
        self.agents = [a for a in self.agents if a.id != agent_id]
        logger.info(f"Removed agent: {agent_id}")

    # =========================================================================
    # Tick Loop
    # =========================================================================

    def _tick_loop(self):
        """Main loop that ticks all agents."""
        interval = self.config.agent_tick_interval_ms / 1000.0

        while self._running:
            tick_start = time.time()

            for agent in self.agents:
                try:
                    result = agent.tick()
                    if result:
                        logger.debug(f"Agent {agent.id} work: {result.work_type}")
                except Exception as e:
                    logger.error(f"Agent {agent.id} error: {e}")

            # Sleep for remaining interval
            elapsed = time.time() - tick_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    # =========================================================================
    # Status Reporting
    # =========================================================================

    def _status_loop(self):
        """Periodically write status to file."""
        while self._running:
            self._write_status()
            time.sleep(self.config.status_interval_seconds)

    def _write_status(self):
        """Write current status to file."""
        if not self.config.status_file:
            return

        status = self.get_status()

        try:
            with open(self.config.status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to write status: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current hive status."""
        hive_status = self.queen_adapter.get_hive_status()

        return {
            "node_id": self.config.node_id,
            "timestamp": datetime.utcnow().isoformat(),
            "running": self._running,
            "agents": {
                "total": len(self.agents),
                "by_role": self._count_by_role(),
                "details": [a.status() for a in self.agents],
            },
            "pheromones": hive_status,
            "queen": {
                "current_mode": self.queen_adapter.get_current_mode(),
                "active_priorities": len(self.queen_adapter.get_active_priorities()),
                "active_alarms": len(self.queen_adapter.get_active_alarms()),
            },
        }

    def _count_by_role(self) -> Dict[str, int]:
        """Count agents by role."""
        counts = {}
        for agent in self.agents:
            counts[agent.role] = counts.get(agent.role, 0) + 1
        return counts


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Ara Hive Node")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to hive_config.yaml",
    )
    parser.add_argument(
        "--status-file",
        type=str,
        help="Override status file path",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    config = HiveConfig.from_yaml(config_path)

    # Apply overrides
    if args.status_file:
        config.status_file = args.status_file
    if args.log_level:
        config.log_level = args.log_level

    # Run node
    node = HiveNode(config=config)
    node.run()


if __name__ == "__main__":
    main()
