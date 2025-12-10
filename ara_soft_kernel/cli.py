#!/usr/bin/env python3
"""
ara-kernel CLI
==============

Command-line interface for the Ara Soft OS Kernel.

Usage:
    ara-kernel start                    # Start daemon
    ara-kernel stop                     # Stop daemon
    ara-kernel status                   # Show status

    ara-kernel supply                   # Show supply profile
    ara-kernel demand                   # Show demand profile
    ara-kernel agents                   # List running agents
    ara-kernel workspaces               # List workspaces

    ara-kernel spawn <agent-spec.json>  # Spawn agent
    ara-kernel stop-agent <instance-id> # Stop agent

    ara-kernel policy show              # Show current policy
    ara-kernel policy reload            # Reload policy

    ara-kernel reconcile --dry-run      # Show what would change
"""

import argparse
import json
import os
import sys
import signal
import time
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ara_soft_kernel.daemon import KernelDaemon
from ara_soft_kernel.models.supply import SupplyProfile, collect_local_supply
from ara_soft_kernel.models.demand import DemandProfile, Goal
from ara_soft_kernel.models.agent import AgentSpec


def get_state_dir() -> Path:
    """Get the state directory."""
    return Path.home() / ".ara" / "state"


def get_config_dir() -> Path:
    """Get the config directory."""
    return Path.home() / ".ara" / "policy"


def is_daemon_running() -> bool:
    """Check if daemon is running."""
    pid_file = get_state_dir() / "kernel.pid"
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False


def get_daemon_pid() -> Optional[int]:
    """Get daemon PID if running."""
    pid_file = get_state_dir() / "kernel.pid"
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except ValueError:
        return None


def cmd_start(args: argparse.Namespace) -> int:
    """Start the kernel daemon."""
    if is_daemon_running():
        print("Daemon is already running")
        return 1

    if args.foreground:
        # Run in foreground
        daemon = KernelDaemon(
            config_dir=get_config_dir(),
            state_dir=get_state_dir(),
        )
        try:
            daemon.run()
        except KeyboardInterrupt:
            daemon.stop()
        return 0
    else:
        # Fork to background
        import subprocess
        proc = subprocess.Popen(
            [sys.executable, __file__, "start", "--foreground"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(1)
        if is_daemon_running():
            print(f"Daemon started (PID: {proc.pid})")
            return 0
        else:
            print("Failed to start daemon")
            return 1


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop the kernel daemon."""
    pid = get_daemon_pid()
    if pid is None:
        print("Daemon is not running")
        return 1

    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for shutdown
        for _ in range(50):
            time.sleep(0.1)
            if not is_daemon_running():
                break
        print("Daemon stopped")
        return 0
    except ProcessLookupError:
        print("Daemon is not running")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show daemon status."""
    if not is_daemon_running():
        print("Status: STOPPED")
        return 0

    pid = get_daemon_pid()
    print(f"Status: RUNNING (PID: {pid})")

    # Read stats from state files
    state_dir = get_state_dir()

    # Show supply
    try:
        supply = collect_local_supply()
        print(f"\nDevices: {len(supply.devices)}")
        for device in supply.devices:
            gpu_str = f", {len(device.gpu)} GPUs" if device.gpu else ""
            battery_str = f", Battery: {device.battery*100:.0f}%" if device.battery else ""
            print(f"  - {device.id} ({device.type}): "
                  f"CPU: {device.cpu_load*100:.0f}%, "
                  f"Mem: {device.memory_free_gb:.1f}GB free"
                  f"{gpu_str}{battery_str}")
    except Exception as e:
        print(f"\nError reading supply: {e}")

    # Show demand
    demand_path = state_dir / "demand.json"
    if demand_path.exists():
        try:
            demand = DemandProfile.from_json(demand_path.read_text())
            print(f"\nGoals: {len(demand.goals)}")
            for goal in demand.get_active_goals()[:5]:
                print(f"  - [{goal.priority:.1f}] {goal.id}: {goal.description[:50]}")
        except Exception as e:
            print(f"\nError reading demand: {e}")

    return 0


def cmd_supply(args: argparse.Namespace) -> int:
    """Show current supply profile."""
    supply = collect_local_supply()

    if args.json:
        print(supply.to_json())
    else:
        print("=== Supply Profile ===")
        print(f"Timestamp: {supply.timestamp}")
        print(f"\nDevices ({len(supply.devices)}):")
        for device in supply.devices:
            print(f"\n  {device.id} ({device.type}):")
            print(f"    CPU: {device.cpu_load*100:.1f}% ({device.cpu_cores_available} cores)")
            print(f"    Memory: {device.memory_free_gb:.1f} / {device.memory_total_gb:.1f} GB")
            if device.gpu:
                print(f"    GPUs:")
                for gpu in device.gpu:
                    print(f"      - {gpu.id}: {gpu.util*100:.0f}%, "
                          f"{gpu.vram_free_gb:.1f}/{gpu.vram_total_gb:.1f} GB VRAM")
            if device.battery is not None:
                print(f"    Battery: {device.battery*100:.0f}%")

    return 0


def cmd_demand(args: argparse.Namespace) -> int:
    """Show current demand profile."""
    demand_path = get_state_dir() / "demand.json"

    if not demand_path.exists():
        print("No demand profile found")
        return 1

    demand = DemandProfile.from_json(demand_path.read_text())

    if args.json:
        print(demand.to_json())
    else:
        print("=== Demand Profile ===")
        print(f"Timestamp: {demand.timestamp}")
        print(f"\nUser State:")
        print(f"  Task: {demand.user_state.task}")
        print(f"  Mode: {demand.user_state.mode.value}")
        print(f"  Interrupt Cost: {demand.user_state.interrupt_cost:.2f}")
        print(f"  Safety Risk: {demand.user_state.safety_risk:.2f}")
        print(f"\nGoals ({len(demand.goals)}):")
        for goal in demand.get_active_goals():
            print(f"  [{goal.priority:.1f}] {goal.id}")
            print(f"      {goal.description}")
        print(f"\nConstraints:")
        print(f"  Privacy: {demand.constraints.privacy_tier.value}")
        print(f"  Latency Tolerance: {demand.constraints.latency_tolerance_ms}ms")

    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    """List running agents."""
    # For now, read from state file
    # In full implementation, would query daemon via socket
    specs_path = get_state_dir() / "agent_specs.json"

    if not specs_path.exists():
        print("No agents registered")
        return 0

    specs = json.loads(specs_path.read_text())
    print(f"Registered agents: {len(specs)}")
    for spec in specs:
        print(f"  - {spec['name']} v{spec['version']} ({spec['kind']})")

    return 0


def cmd_spawn(args: argparse.Namespace) -> int:
    """Spawn an agent from spec file."""
    spec_path = Path(args.spec_file)
    if not spec_path.exists():
        print(f"Spec file not found: {spec_path}")
        return 1

    spec = AgentSpec.from_json(spec_path.read_text())
    print(f"Would spawn agent: {spec.name}")
    print(f"  Kind: {spec.kind.value}")
    print(f"  Resources: {spec.resources.cpu_cores} cores, "
          f"{spec.resources.memory_gb}GB RAM, "
          f"{spec.resources.gpu_mem_gb}GB VRAM")

    # In full implementation, would send to daemon
    return 0


def cmd_stop_agent(args: argparse.Namespace) -> int:
    """Stop an agent."""
    print(f"Would stop agent: {args.instance_id}")
    # In full implementation, would send to daemon
    return 0


def cmd_policy_show(args: argparse.Namespace) -> int:
    """Show current policy."""
    policy_path = get_config_dir() / "ara_governance.toml"

    if not policy_path.exists():
        print("No policy file found (using defaults)")
        print(f"Create policy at: {policy_path}")
        return 0

    print(f"=== Policy: {policy_path} ===\n")
    print(policy_path.read_text())
    return 0


def cmd_policy_reload(args: argparse.Namespace) -> int:
    """Reload policy from disk."""
    if not is_daemon_running():
        print("Daemon is not running")
        return 1

    # In full implementation, would send reload signal to daemon
    print("Policy reload requested")
    return 0


def cmd_reconcile(args: argparse.Namespace) -> int:
    """Force reconciliation cycle."""
    if args.dry_run:
        print("=== Dry Run Reconciliation ===")
        # In full implementation, would query daemon
        print("No changes needed")
    else:
        print("Reconciliation triggered")
    return 0


def cmd_add_goal(args: argparse.Namespace) -> int:
    """Add a goal."""
    demand_path = get_state_dir() / "demand.json"

    if demand_path.exists():
        demand = DemandProfile.from_json(demand_path.read_text())
    else:
        demand = DemandProfile()

    goal = Goal(
        id=args.id,
        description=args.description,
        priority=args.priority,
    )
    demand.add_goal(goal)

    demand_path.parent.mkdir(parents=True, exist_ok=True)
    demand_path.write_text(demand.to_json())

    print(f"Added goal: {goal.id}")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ara Soft OS Kernel CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # start
    p = subparsers.add_parser("start", help="Start the kernel daemon")
    p.add_argument("--foreground", "-f", action="store_true",
                   help="Run in foreground")
    p.set_defaults(func=cmd_start)

    # stop
    p = subparsers.add_parser("stop", help="Stop the kernel daemon")
    p.set_defaults(func=cmd_stop)

    # status
    p = subparsers.add_parser("status", help="Show daemon status")
    p.set_defaults(func=cmd_status)

    # supply
    p = subparsers.add_parser("supply", help="Show supply profile")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.set_defaults(func=cmd_supply)

    # demand
    p = subparsers.add_parser("demand", help="Show demand profile")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.set_defaults(func=cmd_demand)

    # agents
    p = subparsers.add_parser("agents", help="List running agents")
    p.set_defaults(func=cmd_agents)

    # spawn
    p = subparsers.add_parser("spawn", help="Spawn an agent")
    p.add_argument("spec_file", help="Agent spec JSON file")
    p.set_defaults(func=cmd_spawn)

    # stop-agent
    p = subparsers.add_parser("stop-agent", help="Stop an agent")
    p.add_argument("instance_id", help="Agent instance ID")
    p.set_defaults(func=cmd_stop_agent)

    # policy
    p = subparsers.add_parser("policy", help="Policy management")
    policy_sub = p.add_subparsers(dest="policy_command")

    pp = policy_sub.add_parser("show", help="Show current policy")
    pp.set_defaults(func=cmd_policy_show)

    pp = policy_sub.add_parser("reload", help="Reload policy from disk")
    pp.set_defaults(func=cmd_policy_reload)

    # reconcile
    p = subparsers.add_parser("reconcile", help="Force reconciliation")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would change without executing")
    p.set_defaults(func=cmd_reconcile)

    # add-goal
    p = subparsers.add_parser("add-goal", help="Add a goal")
    p.add_argument("id", help="Goal ID")
    p.add_argument("description", help="Goal description")
    p.add_argument("--priority", type=float, default=0.5,
                   help="Priority (0-1)")
    p.set_defaults(func=cmd_add_goal)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
