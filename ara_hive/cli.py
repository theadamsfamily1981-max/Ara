#!/usr/bin/env python3
"""
Ara Hive CLI
=============

Command-line interface for testing the bee colony scheduler.

Usage:
    python -m ara_hive.cli demo           # Run demo with 2 nodes, 10 tasks
    python -m ara_hive.cli submit IMAGE   # Submit a task
    python -m ara_hive.cli stats          # Show hive stats
    python -m ara_hive.cli agent NODE_ID  # Run an agent
"""

from __future__ import annotations

import argparse
import logging
import time
import sys
from pathlib import Path

from .waggle_board import WaggleBoard, Node
from .bee_agent import BeeAgent, AgentConfig, HiveNode
from .maintenance import HiveMaintainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_DB = Path("ara_hive.sqlite")


def cmd_demo(args: argparse.Namespace) -> None:
    """Run a demo of the hive system."""
    print("=" * 60)
    print("Ara Hive Demo")
    print("=" * 60)

    # Setup
    board = WaggleBoard(args.db)
    maintainer = HiveMaintainer(board)
    maintainer.start()

    # Create nodes
    print("\n[1] Creating nodes...")
    nodes = []
    for i in range(args.nodes):
        node_id = f"demo-node-{i:02d}"
        config = AgentConfig(
            node_id=node_id,
            role="worker",
            task_types=["demo_task"],
        )
        node = HiveNode(
            board=board,
            node_id=node_id,
            task_types=["demo_task"],
            num_agents=args.agents_per_node,
        )
        node.start()
        nodes.append(node)
        print(f"  Started {node_id} with {args.agents_per_node} agents")

    # Submit tasks
    print(f"\n[2] Submitting {args.tasks} tasks...")
    for i in range(args.tasks):
        board.submit_task(
            task_type="demo_task",
            payload={"task_num": i, "message": f"Demo task {i}"},
            priority=0.5,
        )
    print(f"  Submitted {args.tasks} tasks")

    # Wait and show progress
    print(f"\n[3] Running for {args.duration}s...")
    start = time.time()
    while time.time() - start < args.duration:
        stats = board.get_stats()
        print(
            f"  Queued: {stats['tasks_queued']:3d} | "
            f"Running: {stats['tasks_running']:3d} | "
            f"Done: {stats['tasks_completed']:3d} | "
            f"Avg Intensity: {stats['avg_site_intensity']:.3f}"
        )
        time.sleep(2.0)

    # Final stats
    print("\n[4] Final Stats")
    stats = board.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    for node in nodes:
        node_stats = node.get_stats()
        print(
            f"  {node_stats['node_id']}: "
            f"completed={node_stats['total_completed']}, "
            f"failed={node_stats['total_failed']}"
        )

    # Cleanup
    print("\n[5] Stopping...")
    for node in nodes:
        node.stop()
    maintainer.stop()

    print("\nDemo complete!")


def cmd_submit(args: argparse.Namespace) -> None:
    """Submit a task to the queue."""
    board = WaggleBoard(args.db)

    task = board.submit_task(
        task_type=args.task_type,
        payload={"message": args.message} if args.message else {},
        priority=args.priority,
    )

    print(f"Submitted task {task.id} (type={task.task_type}, priority={task.priority})")


def cmd_stats(args: argparse.Namespace) -> None:
    """Show hive statistics."""
    board = WaggleBoard(args.db)
    stats = board.get_stats()

    print("Hive Statistics")
    print("=" * 40)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Node details
    print("\nNodes:")
    for node in board.list_nodes():
        status_icon = "✓" if node.status == "online" else "✗"
        print(
            f"  {status_icon} {node.id} ({node.role}) - "
            f"CPU: {node.cpu_load:.1%}, MEM: {node.mem_used_pct:.1%}"
        )

    # Site details
    print("\nTop Sites (by intensity):")
    seen_types = set()
    for task_type in ["demo_task", "image_gen", "publish", "scrape"]:
        if task_type in seen_types:
            continue
        sites = board.get_sites_for_task_type(task_type)
        if sites:
            seen_types.add(task_type)
            for site in sites[:5]:
                print(
                    f"  {site.task_type}@{site.node_id}: "
                    f"intensity={site.intensity:.3f}, "
                    f"visits={site.visit_count}, "
                    f"success_rate={site.success_rate:.1%}"
                )


def cmd_agent(args: argparse.Namespace) -> None:
    """Run a bee agent."""
    board = WaggleBoard(args.db)

    # Start maintenance (optional, may run separately)
    if args.with_maintenance:
        maintainer = HiveMaintainer(board)
        maintainer.start()

    config = AgentConfig(
        node_id=args.node_id,
        role=args.role,
        task_types=args.task_types.split(",") if args.task_types else ["demo_task"],
    )

    agent = BeeAgent(board, config)
    agent.start()

    print(f"Agent {args.node_id} running. Press Ctrl+C to stop...")

    try:
        while True:
            time.sleep(5)
            stats = agent.get_stats()
            print(
                f"[{args.node_id}] role={stats['role']}, "
                f"completed={stats['tasks_completed']}, "
                f"failed={stats['tasks_failed']}"
            )
    except KeyboardInterrupt:
        print("\nStopping...")

    agent.stop()
    if args.with_maintenance:
        maintainer.stop()


def cmd_seed(args: argparse.Namespace) -> None:
    """Seed the database with nodes and sites."""
    board = WaggleBoard(args.db)

    print("Seeding database...")

    # Create nodes
    for i in range(args.nodes):
        node = Node(
            id=f"node-{i:02d}",
            role="worker",
            hostname=f"host-{i:02d}",
            capabilities=args.task_types.split(","),
            status="online",
            last_heartbeat=time.time(),
        )
        board.register_node(node)
        print(f"  Created node: {node.id}")

        # Create sites
        for task_type in args.task_types.split(","):
            board.create_site(task_type, node.id)
            print(f"    Created site: {task_type}@{node.id}")

    print("\nSeeding complete!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ara Hive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Database path",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demo")
    demo_parser.add_argument("--nodes", type=int, default=2, help="Number of nodes")
    demo_parser.add_argument("--agents-per-node", type=int, default=2, help="Agents per node")
    demo_parser.add_argument("--tasks", type=int, default=20, help="Number of tasks")
    demo_parser.add_argument("--duration", type=float, default=30.0, help="Duration in seconds")

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a task")
    submit_parser.add_argument("task_type", help="Task type")
    submit_parser.add_argument("--message", "-m", help="Task message")
    submit_parser.add_argument("--priority", "-p", type=float, default=0.5, help="Priority")

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    # Agent command
    agent_parser = subparsers.add_parser("agent", help="Run an agent")
    agent_parser.add_argument("node_id", help="Node ID")
    agent_parser.add_argument("--role", default="worker", help="Node role")
    agent_parser.add_argument("--task-types", default="demo_task", help="Task types (comma-separated)")
    agent_parser.add_argument("--with-maintenance", action="store_true", help="Run maintenance jobs")

    # Seed command
    seed_parser = subparsers.add_parser("seed", help="Seed database")
    seed_parser.add_argument("--nodes", type=int, default=3, help="Number of nodes")
    seed_parser.add_argument("--task-types", default="demo_task,image_gen", help="Task types")

    args = parser.parse_args()

    if args.command == "demo":
        cmd_demo(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "agent":
        cmd_agent(args)
    elif args.command == "seed":
        cmd_seed(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
