# ara/enterprise/org_chart.py
"""
The OrgChart - Corporation Croft's HR Department
=================================================

Manages the roster of machine-employees in The Fleet.
Each machine has a role, capabilities, and access contracts.

Roles:
    - INTERN: Sandbox/testing machines. High-risk experiments allowed.
    - WORKER: Production compute. Low-risk tasks only.
    - CONSULTANT: Read-only/restricted. Human endpoints, crown jewels.

Usage:
    from ara.enterprise.org_chart import OrgChart

    chart = OrgChart()
    emp = chart.get_employee_for_task(task_risk="high", needs_gpu=False)
    if emp:
        # Dispatch work to emp.hostname
        ...
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Literal
import time
import logging

log = logging.getLogger("Ara.OrgChart")


class EmployeeRole(str, Enum):
    """
    Types of machine-employees in The Fleet.
    """
    INTERN = "intern"          # Sandbox / Testing (High risk allowed)
    WORKER = "worker"          # Production compute (Low risk only)
    CONSULTANT = "consultant"  # Read-only / Restricted (No touch without explicit auth)


EmployeeStatus = Literal["offline", "online", "error", "degraded"]


@dataclass
class Employee:
    """
    A single machine in the Fleet.
    """
    id: str                        # e.g. "intern-pi01"
    hostname: str                  # e.g. "192.168.1.105" or "ara-intern.local"
    role: EmployeeRole
    capabilities: List[str] = field(default_factory=list)
    status: EmployeeStatus = "offline"
    current_task: Optional[str] = None

    # Employment contract / access-control flags.
    allow_sudo: bool = False
    allow_internet: bool = False
    allow_write_local_disk: bool = True

    # Optional metadata: OS, notes, etc.
    labels: Dict[str, str] = field(default_factory=dict)
    last_heartbeat_ts: float = 0.0

    def is_online(self, stale_after_sec: float = 120.0) -> bool:
        """
        Simple liveness check based on status + heartbeat.
        You can wire this up to actual health pings.
        """
        if self.status != "online":
            return False
        if self.last_heartbeat_ts <= 0:
            return True
        return (time.time() - self.last_heartbeat_ts) < stale_after_sec

    def has_capability(self, needle: str) -> bool:
        """
        Lightweight capability check.
        e.g. needle="gpu" or "gpu:3090".
        """
        for cap in self.capabilities:
            if needle in cap:
                return True
        return False


class OrgChart:
    """
    The HR Department.

    Manages the roster of machine employees, their roles,
    and basic dispatch selection.

    Usage:

        chart = OrgChart()
        emp = chart.get_employee_for_task(task_risk="high", needs_gpu=False)
        if emp:
            ...
    """

    def __init__(self, auto_seed: bool = True):
        self.employees: Dict[str, Employee] = {}
        if auto_seed:
            self._hire_initial_staff()

    # ------------------------------------------------------------------ #
    # HIRING / UPDATING
    # ------------------------------------------------------------------ #

    def _hire_initial_staff(self) -> None:
        """
        Seed some default roles.

        EDIT THESE to match your real network:
          - intern-pi  -> junkyard Pi / VM sandbox
          - worker-gpu -> your big GPU box
          - consultant-mac -> your Mac laptop (read-only)
        """
        log.info("OrgChart: seeding initial Fleet.")

        self.hire(
            Employee(
                id="intern-pi",
                hostname="192.168.1.105",  # e.g. Raspberry Pi / VM sandbox
                role=EmployeeRole.INTERN,
                capabilities=["python", "docker"],
                allow_internet=True,  # can download sketchy packages
                allow_sudo=False,
                labels={"tier": "sandbox"},
            )
        )

        self.hire(
            Employee(
                id="worker-gpu",
                hostname="192.168.1.200",  # your 3090 / cathedral muscle
                role=EmployeeRole.WORKER,
                capabilities=["gpu:3090", "cuda", "python"],
                allow_sudo=False,
                allow_internet=False,      # optional; keep prod constrained
                labels={"tier": "prod"},
            )
        )

        # Your old Mac laptop as a read-mostly consultant
        self.hire(
            Employee(
                id="consultant-mac",
                hostname="mac-junkyard.local",  # change to actual hostname/IP
                role=EmployeeRole.CONSULTANT,
                capabilities=["desktop", "storage:personal"],
                allow_sudo=False,
                allow_internet=False,           # treat as external human endpoint
                allow_write_local_disk=False,   # Ara should not write here by default
                labels={"tier": "crown-jewels"},
            )
        )

    def hire(self, employee: Employee) -> None:
        """
        Add or replace an employee record.
        """
        self.employees[employee.id] = employee
        log.info(
            "OrgChart: hired/updated %s (%s) @ %s caps=%s",
            employee.id,
            employee.role.value,
            employee.hostname,
            employee.capabilities,
        )

    def fire(self, employee_id: str) -> None:
        """
        Remove an employee from the roster.
        """
        if employee_id in self.employees:
            emp = self.employees.pop(employee_id)
            log.warning("OrgChart: fired %s (%s)", emp.id, emp.hostname)

    def list_employees(self) -> List[Employee]:
        return list(self.employees.values())

    def heartbeat(self, employee_id: str, status: EmployeeStatus = "online") -> None:
        """
        Update a heartbeat for a machine (called by monitor / health-check).
        """
        emp = self.employees.get(employee_id)
        if not emp:
            log.warning("OrgChart: heartbeat from unknown employee '%s'", employee_id)
            return
        emp.status = status
        emp.last_heartbeat_ts = time.time()
        log.debug("OrgChart: heartbeat from %s (%s)", emp.id, status)

    def set_status(self, employee_id: str, status: EmployeeStatus) -> None:
        emp = self.employees.get(employee_id)
        if not emp:
            return
        emp.status = status
        if status != "offline":
            emp.last_heartbeat_ts = time.time()

    # ------------------------------------------------------------------ #
    # DISPATCH LOGIC
    # ------------------------------------------------------------------ #

    def get_employee_for_task(
        self,
        task_risk: Literal["low", "medium", "high"],
        needs_gpu: bool = False,
        prefer_role: Optional[EmployeeRole] = None,
        allow_consultant: bool = False,
    ) -> Optional[Employee]:
        """
        Core dispatch policy.

        - High risk -> Intern only.
        - Needs GPU -> must have "gpu" capability.
        - Consultant machines are never chosen unless allow_consultant=True.
        - Optionally bias to a particular role (e.g. WORKER).
        """
        candidates: List[Employee] = []

        for emp in self.employees.values():
            if not emp.is_online():
                continue

            # Respect role bounds
            if task_risk == "high" and emp.role is not EmployeeRole.INTERN:
                continue

            if emp.role is EmployeeRole.CONSULTANT and not allow_consultant:
                continue

            if needs_gpu and not emp.has_capability("gpu"):
                continue

            candidates.append(emp)

        if not candidates:
            log.warning(
                "OrgChart: no available employee for task_risk=%s needs_gpu=%s",
                task_risk,
                needs_gpu,
            )
            return None

        # If a preferred role is specified, try that first.
        if prefer_role is not None:
            role_filtered = [c for c in candidates if c.role is prefer_role]
            if role_filtered:
                candidates = role_filtered

        # Simple heuristic: interns for high risk, workers otherwise
        # with a stable ordering (sorted by id).
        candidates.sort(key=lambda e: (e.role.value, e.id))

        chosen = candidates[0]
        log.info(
            "OrgChart: assigned %s (%s) for risk=%s gpu=%s",
            chosen.id,
            chosen.role.value,
            task_risk,
            needs_gpu,
        )
        return chosen

    # ------------------------------------------------------------------ #
    # EXPORT / HUD
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Dict[str, Dict]:
        """
        Return a JSON-serializable view of the Fleet
        for dashboards / shaders / soul_quantum HUD.
        """
        return {
            emp.id: {
                **asdict(emp),
                "role": emp.role.value,
            }
            for emp in self.employees.values()
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_default_org_chart: Optional[OrgChart] = None


def get_org_chart() -> OrgChart:
    """Get the default OrgChart instance."""
    global _default_org_chart
    if _default_org_chart is None:
        _default_org_chart = OrgChart()
    return _default_org_chart


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'EmployeeRole',
    'EmployeeStatus',
    'Employee',
    'OrgChart',
    'get_org_chart',
]
