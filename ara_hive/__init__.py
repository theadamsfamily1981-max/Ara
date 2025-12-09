"""
Ara Hive - Distributed Job Coordination
========================================

Implements the Artificial Bee Colony algorithm for distributed task scheduling.

Components:
- WaggleBoard: Shared state (nodes, sites, tasks)
- BeeAgent: Worker that picks sites and executes tasks
- HiveNode: Runs multiple agents on a node
- HiveMaintainer: Background jobs (evaporation, cooling)

Usage:
    from ara_hive import WaggleBoard, BeeAgent, AgentConfig, HiveMaintainer

    # Setup
    board = WaggleBoard(Path("hive.sqlite"))
    maintainer = HiveMaintainer(board)
    maintainer.start()

    # Run agent
    config = AgentConfig(node_id="worker-01", task_types=["image_gen"])
    agent = BeeAgent(board, config)
    agent.start()

    # Submit tasks
    board.submit_task("image_gen", {"prompt": "a cat"})
"""

from .waggle_board import (
    WaggleBoard,
    Node,
    Site,
    Task,
    NodeRole,
    NodeStatus,
    TaskStatus,
)

from .bee_agent import (
    BeeAgent,
    AgentConfig,
    BeeRole,
    SiteState,
    HiveNode,
)

from .maintenance import (
    EvaporationJob,
    StaleNodeDetector,
    CongestionCooler,
    HiveMaintainer,
)

__all__ = [
    # Waggle Board
    "WaggleBoard",
    "Node",
    "Site",
    "Task",
    "NodeRole",
    "NodeStatus",
    "TaskStatus",
    # Bee Agent
    "BeeAgent",
    "AgentConfig",
    "BeeRole",
    "SiteState",
    "HiveNode",
    # Maintenance
    "EvaporationJob",
    "StaleNodeDetector",
    "CongestionCooler",
    "HiveMaintainer",
]
