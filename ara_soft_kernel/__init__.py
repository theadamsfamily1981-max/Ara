"""
Ara Soft OS Kernel
==================

Ara as a soft OS layer over Linux + ALWAYS VISION + agent swarm.

Architecture:
    Observer     - Watches world, user, and resource state
    Orchestrator - Decides which agents run where
    Governor     - Enforces permissions, budgets, safety rules

The kernel continuously reconciles desired state with actual state,
emitting jobs that spawn/stop/migrate agents and create workspaces.
"""

from ara_soft_kernel.models.supply import SupplyProfile, DeviceInfo, GpuInfo
from ara_soft_kernel.models.demand import DemandProfile, UserState, Goal
from ara_soft_kernel.models.agent import AgentSpec, AgentInstance, AgentState
from ara_soft_kernel.models.workspace import WorkspaceSpec, Surface
from ara_soft_kernel.models.job import Job, JobType, JobState

from ara_soft_kernel.observer import Observer
from ara_soft_kernel.orchestrator import Orchestrator
from ara_soft_kernel.governor import Governor, GovernanceRules
from ara_soft_kernel.reconciler import Reconciler
from ara_soft_kernel.daemon import KernelDaemon

__all__ = [
    # Models
    "SupplyProfile", "DeviceInfo", "GpuInfo",
    "DemandProfile", "UserState", "Goal",
    "AgentSpec", "AgentInstance", "AgentState",
    "WorkspaceSpec", "Surface",
    "Job", "JobType", "JobState",
    # Core
    "Observer",
    "Orchestrator",
    "Governor", "GovernanceRules",
    "Reconciler",
    "KernelDaemon",
]

__version__ = "0.1.0"
