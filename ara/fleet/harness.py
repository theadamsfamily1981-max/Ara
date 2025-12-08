#!/usr/bin/env python3
"""
Fleet Harness - Simple Fleet Management
========================================

A minimal harness for managing your fleet of machines.

This is the "doable tonight" version:
- Load roster from YAML
- SSH to machines for health checks
- Talk to Arduinos (if connected)
- Basic status dashboard

Usage:
    python -m ara.fleet.harness status
    python -m ara.fleet.harness ping
    python -m ara.fleet.harness ssh <node_id> <command>
"""

from __future__ import annotations
import os
import sys
import time
import subprocess
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("Warning: pyyaml not installed. Install with: pip install pyyaml")


@dataclass
class FleetNode:
    """A node in the fleet."""
    id: str
    hostname: str
    role: str
    ip: str = ""
    status: str = "unknown"
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    # Runtime state
    last_ping: float = 0.0
    ping_ok: bool = False


@dataclass
class FleetHarness:
    """
    Simple fleet management harness.

    Loads roster, pings machines, runs commands.
    """
    roster_path: str = "ara/fleet/roster.yml"
    nodes: Dict[str, FleetNode] = field(default_factory=dict)
    ssh_user: str = "ara_worker"
    ssh_timeout: int = 5

    def load_roster(self) -> bool:
        """Load fleet roster from YAML."""
        if not HAS_YAML:
            print("Cannot load roster: pyyaml not installed")
            return False

        path = Path(self.roster_path)
        if not path.exists():
            print(f"Roster not found: {path}")
            return False

        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            # Parse crown jewels
            for node in data.get("crown_jewels", []):
                self._add_node(node)

            # Parse workhorses
            for node in data.get("workhorses", []):
                self._add_node(node)

            # Parse i3 infrastructure
            for node in data.get("i3_infrastructure", []):
                self._add_node(node)

            # Parse i3 interns
            for node in data.get("i3_interns", []):
                self._add_node(node)

            # Parse laptops
            for node in data.get("laptops", []):
                self._add_node(node)

            # Parse accelerators
            for node in data.get("accelerators", []):
                self._add_node(node)

            print(f"Loaded {len(self.nodes)} nodes from roster")
            return True

        except Exception as e:
            print(f"Error loading roster: {e}")
            return False

    def _add_node(self, data: Dict):
        """Add a node from roster data."""
        node_id = data.get("id", "")
        if not node_id:
            return

        ip = ""
        network = data.get("network", {})
        if isinstance(network, dict):
            ip = network.get("ip", "")

        node = FleetNode(
            id=node_id,
            hostname=data.get("hostname", node_id),
            role=data.get("role", "UNKNOWN"),
            ip=ip,
            status=data.get("status", "unknown"),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )
        self.nodes[node_id] = node

    def ping_all(self) -> Dict[str, bool]:
        """Ping all nodes with known IPs."""
        results = {}

        for node in self.nodes.values():
            if node.ip:
                ok = self.ping_node(node.id)
                results[node.id] = ok
            else:
                results[node.id] = None  # No IP to ping

        return results

    def ping_node(self, node_id: str) -> bool:
        """Ping a specific node."""
        if node_id not in self.nodes:
            return False

        node = self.nodes[node_id]
        if not node.ip:
            return False

        try:
            # Quick ping: 1 packet, 1 second timeout
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "1", node.ip],
                capture_output=True,
                timeout=5,
            )
            ok = result.returncode == 0
            node.ping_ok = ok
            node.last_ping = time.time()
            return ok

        except Exception as e:
            node.ping_ok = False
            return False

    def ssh_command(self, node_id: str, command: str) -> Optional[str]:
        """Run a command on a node via SSH."""
        if node_id not in self.nodes:
            print(f"Unknown node: {node_id}")
            return None

        node = self.nodes[node_id]
        if not node.ip:
            print(f"No IP for node: {node_id}")
            return None

        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o", f"ConnectTimeout={self.ssh_timeout}",
                    "-o", "StrictHostKeyChecking=no",
                    f"{self.ssh_user}@{node.ip}",
                    command,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return result.stdout
            else:
                print(f"SSH error: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"SSH timeout for {node_id}")
            return None
        except Exception as e:
            print(f"SSH error: {e}")
            return None

    def get_node_health(self, node_id: str) -> Dict[str, Any]:
        """Get health info for a node via SSH."""
        output = self.ssh_command(node_id, "uptime && free -h && df -h / | tail -1")
        if output:
            return {"output": output, "ok": True}
        return {"output": None, "ok": False}

    def status(self) -> Dict[str, Any]:
        """Get overall fleet status."""
        total = len(self.nodes)
        online = sum(1 for n in self.nodes.values() if n.status == "online")
        with_ip = sum(1 for n in self.nodes.values() if n.ip)

        by_role = {}
        for node in self.nodes.values():
            role = node.role
            if role not in by_role:
                by_role[role] = 0
            by_role[role] += 1

        return {
            "total_nodes": total,
            "nodes_online": online,
            "nodes_with_ip": with_ip,
            "by_role": by_role,
        }

    def print_status(self):
        """Print fleet status to console."""
        status = self.status()

        print("\n" + "=" * 60)
        print("FLEET STATUS")
        print("=" * 60)
        print(f"Total nodes: {status['total_nodes']}")
        print(f"Online: {status['nodes_online']}")
        print(f"With IP: {status['nodes_with_ip']}")
        print("\nBy role:")
        for role, count in status['by_role'].items():
            print(f"  {role}: {count}")

        print("\n" + "-" * 60)
        print("NODES")
        print("-" * 60)

        for node in sorted(self.nodes.values(), key=lambda n: n.role):
            status_str = node.status
            if node.ping_ok:
                status_str = "PING OK"

            ip_str = node.ip if node.ip else "(no IP)"
            print(f"  {node.id:20} {node.role:12} {status_str:10} {ip_str}")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fleet Harness")
    parser.add_argument("command", choices=["status", "ping", "ssh", "health"],
                        help="Command to run")
    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument("--roster", default="ara/fleet/roster.yml",
                        help="Path to roster YAML")
    parser.add_argument("--user", default="ara_worker",
                        help="SSH user")

    args = parser.parse_args()

    harness = FleetHarness(
        roster_path=args.roster,
        ssh_user=args.user,
    )

    if not harness.load_roster():
        print("Failed to load roster. Create one at ara/fleet/roster.yml")
        print("See ara/fleet/roster.yml for template.")
        sys.exit(1)

    if args.command == "status":
        harness.print_status()

    elif args.command == "ping":
        print("Pinging all nodes...")
        results = harness.ping_all()

        print("\nPing results:")
        for node_id, ok in results.items():
            if ok is None:
                status = "no IP"
            elif ok:
                status = "OK"
            else:
                status = "FAIL"
            print(f"  {node_id}: {status}")

        harness.print_status()

    elif args.command == "ssh":
        if len(args.args) < 2:
            print("Usage: harness ssh <node_id> <command>")
            sys.exit(1)

        node_id = args.args[0]
        command = " ".join(args.args[1:])

        print(f"Running on {node_id}: {command}")
        output = harness.ssh_command(node_id, command)
        if output:
            print(output)

    elif args.command == "health":
        if args.args:
            node_id = args.args[0]
            print(f"Getting health for {node_id}...")
            health = harness.get_node_health(node_id)
            if health["ok"]:
                print(health["output"])
            else:
                print("Failed to get health")
        else:
            print("Checking health of all online nodes...")
            for node in harness.nodes.values():
                if node.ip and node.status == "online":
                    print(f"\n{node.id}:")
                    health = harness.get_node_health(node.id)
                    if health["ok"]:
                        print(health["output"])
                    else:
                        print("  (unreachable)")


if __name__ == "__main__":
    main()
