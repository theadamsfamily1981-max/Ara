"""
Kernel Daemon
=============

Main entry point for the Ara Soft OS Kernel.
Manages Observer, Orchestrator, Governor, and Reconciler.
"""

from __future__ import annotations

import os
import sys
import json
import signal
import threading
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Any

from ara_soft_kernel.models.supply import SupplyProfile
from ara_soft_kernel.models.demand import DemandProfile, Goal
from ara_soft_kernel.models.agent import AgentSpec
from ara_soft_kernel.observer import Observer
from ara_soft_kernel.orchestrator import Orchestrator
from ara_soft_kernel.governor import Governor, GovernanceRules
from ara_soft_kernel.reconciler import Reconciler

logger = logging.getLogger(__name__)


class KernelDaemon:
    """
    Main Ara Soft OS Kernel daemon.

    Coordinates all subsystems and provides external interface.
    """

    def __init__(
        self,
        config_dir: Optional[Path] = None,
        state_dir: Optional[Path] = None,
    ):
        # Directories
        self.config_dir = config_dir or Path.home() / ".ara" / "policy"
        self.state_dir = state_dir or Path.home() / ".ara" / "state"
        self.jobs_dir = self.state_dir / "jobs"

        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        # Load governance rules
        policy_path = self.config_dir / "ara_governance.toml"
        if policy_path.exists():
            self.rules = GovernanceRules.from_toml(policy_path)
        else:
            self.rules = GovernanceRules()
            logger.info("Using default governance rules (no policy file found)")

        # Initialize subsystems
        self.observer = Observer()
        self.orchestrator = Orchestrator()
        self.governor = Governor(self.rules)
        self.reconciler = Reconciler(
            observer=self.observer,
            orchestrator=self.orchestrator,
            governor=self.governor,
        )

        # State
        self._running = False
        self._start_time: Optional[float] = None
        self._shutdown_event = threading.Event()

    def start(self) -> None:
        """Start the kernel daemon."""
        if self._running:
            logger.warning("Daemon already running")
            return

        logger.info("Starting Ara Soft OS Kernel...")

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Restore state
        self._restore_state()

        # Start subsystems
        self.observer.start()
        self.reconciler.start()

        self._running = True
        self._start_time = time.time()

        logger.info("Ara Soft OS Kernel started")

        # Save PID
        pid_file = self.state_dir / "kernel.pid"
        pid_file.write_text(str(os.getpid()))

    def stop(self) -> None:
        """Stop the kernel daemon."""
        if not self._running:
            return

        logger.info("Stopping Ara Soft OS Kernel...")

        # Stop subsystems
        self.reconciler.stop()
        self.observer.stop()

        # Persist state
        self._persist_state()

        self._running = False

        # Remove PID file
        pid_file = self.state_dir / "kernel.pid"
        if pid_file.exists():
            pid_file.unlink()

        logger.info("Ara Soft OS Kernel stopped")

    def run(self) -> None:
        """Run the daemon until signaled to stop."""
        self.start()
        self._shutdown_event.wait()
        self.stop()

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self._shutdown_event.set()

    def _restore_state(self) -> None:
        """Restore state from disk."""
        # Restore demand
        demand_path = self.state_dir / "demand.json"
        if demand_path.exists():
            try:
                demand = DemandProfile.from_json(demand_path.read_text())
                self.observer.update_demand(demand)
                logger.info("Restored demand state")
            except Exception as e:
                logger.warning(f"Failed to restore demand: {e}")

        # Restore agent specs
        specs_path = self.state_dir / "agent_specs.json"
        if specs_path.exists():
            try:
                specs_data = json.loads(specs_path.read_text())
                for spec_data in specs_data:
                    spec = AgentSpec.from_dict(spec_data)
                    self.orchestrator.register_spec(spec)
                logger.info(f"Restored {len(specs_data)} agent specs")
            except Exception as e:
                logger.warning(f"Failed to restore agent specs: {e}")

    def _persist_state(self) -> None:
        """Persist state to disk."""
        # Persist demand
        demand_path = self.state_dir / "demand.json"
        demand = self.observer.get_demand()
        demand_path.write_text(demand.to_json())

        # Persist agent specs
        specs_path = self.state_dir / "agent_specs.json"
        specs = self.orchestrator.list_specs()
        specs_data = [s.to_dict() for s in specs]
        specs_path.write_text(json.dumps(specs_data, indent=2))

        logger.info("State persisted")

    # ─────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────

    def get_supply(self) -> SupplyProfile:
        """Get current supply profile."""
        return self.observer.get_supply()

    def get_demand(self) -> DemandProfile:
        """Get current demand profile."""
        return self.observer.get_demand()

    def set_demand(self, demand: DemandProfile) -> None:
        """Set demand profile."""
        self.observer.update_demand(demand)

    def add_goal(self, goal: Goal) -> None:
        """Add a new goal."""
        self.observer.add_goal(goal)
        logger.info(f"Added goal: {goal.id}")

    def remove_goal(self, goal_id: str) -> bool:
        """Remove a goal."""
        result = self.observer.remove_goal(goal_id)
        if result:
            logger.info(f"Removed goal: {goal_id}")
        return result

    def register_agent(self, spec: AgentSpec) -> None:
        """Register an agent specification."""
        self.orchestrator.register_spec(spec)

    def spawn_agent(self, name: str, device: Optional[str] = None) -> Optional[str]:
        """Spawn an agent by name."""
        spec = self.orchestrator.get_spec(name)
        if not spec:
            logger.error(f"Agent spec not found: {name}")
            return None

        supply = self.observer.get_supply()
        device_id = device or self.orchestrator.select_device(spec, supply)

        if not device_id:
            logger.error(f"No suitable device for {name}")
            return None

        instance = self.orchestrator.spawn_agent(spec, device_id)
        return instance.instance_id

    def stop_agent(self, instance_id: str) -> bool:
        """Stop an agent."""
        return self.orchestrator.stop_agent(instance_id)

    def list_agents(self) -> list:
        """List running agents."""
        return [a.to_dict() for a in self.orchestrator.get_running_agents()]

    def reconcile(self, dry_run: bool = False) -> dict:
        """Force a reconciliation cycle."""
        result = self.reconciler.reconcile(dry_run=dry_run)
        return {
            "timestamp": result.timestamp,
            "jobs": len(result.jobs),
            "agents_to_start": result.agents_to_start,
            "agents_to_stop": result.agents_to_stop,
            "agents_to_migrate": result.agents_to_migrate,
            "governance_rejections": result.governance_rejections,
            "cycle_time_ms": result.cycle_time_ms,
        }

    def reload_policy(self) -> bool:
        """Reload governance policy from disk."""
        return self.rules.reload()

    def get_status(self) -> Dict[str, Any]:
        """Get daemon status."""
        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            "running": self._running,
            "uptime_sec": uptime,
            "observer": self.observer.get_stats(),
            "orchestrator": self.orchestrator.get_stats(),
            "governor": self.governor.get_stats(),
            "reconciler": self.reconciler.get_stats(),
        }


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ara Soft OS Kernel")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path.home() / ".ara" / "policy",
        help="Configuration directory",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path.home() / ".ara" / "state",
        help="State directory",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Run daemon
    daemon = KernelDaemon(
        config_dir=args.config_dir,
        state_dir=args.state_dir,
    )
    daemon.run()


if __name__ == "__main__":
    main()
