# ara/enterprise/dispatcher.py
"""
The Dispatcher - Corporation Croft's Taskmaster
================================================

Sends code and commands to employees via SSH.
Keeps the Cathedral clean by delegating experiments
and heavy workloads to the appropriate machine.

This is intentionally conservative:
- No sudo by default.
- No rm -rf or destructive commands baked in.
- Contract flags (allow_sudo, allow_write_local_disk, ...) must be honored
  by higher-level call sites.

Usage:
    from ara.enterprise.org_chart import OrgChart
    from ara.enterprise.dispatcher import Dispatcher

    org = OrgChart()
    dispatcher = Dispatcher(ssh_user="ara_worker")

    emp = org.get_employee_for_task(task_risk="high", needs_gpu=False)
    if emp:
        dispatcher.run_inline(emp, "print('hello from the intern')")
"""
from __future__ import annotations

import base64
import logging
import uuid
from typing import Optional, Dict, Any, Tuple

from .org_chart import Employee

log = logging.getLogger("Ara.Dispatcher")

try:
    # Fabric 2.x / 3.x
    from fabric import Connection  # type: ignore
    _HAS_FABRIC = True
except Exception:  # pragma: no cover
    Connection = None  # lazy failure if fabric isn't installed
    _HAS_FABRIC = False


class Dispatcher:
    """
    The Taskmaster.

    Sends code and commands to employees via SSH.
    Keeps the Cathedral clean by delegating experiments
    and heavy workloads to the appropriate machine.

    This is intentionally conservative:
    - No sudo by default.
    - No rm -rf or destructive commands baked in.
    - Contract flags (allow_sudo, allow_write_local_disk, ...) must be honored
      by higher-level call sites.

    Typical usage:

        emp = org_chart.get_employee_for_task("high", needs_gpu=False)
        if emp:
            dispatcher.run_inline(emp, "print('hello from intern')")

    """

    def __init__(
        self,
        ssh_user: str = "ara_worker",
        default_python: str = "python3",
        dry_run: bool = False,
    ):
        self.ssh_user = ssh_user
        self.default_python = default_python
        self.dry_run = dry_run

        if Connection is None:
            log.warning(
                "Dispatcher: fabric not available. Remote execution will fail. "
                "Install with: pip install fabric"
            )

    @property
    def is_available(self) -> bool:
        """Check if remote execution is available (fabric installed)."""
        return _HAS_FABRIC

    # ------------------------------------------------------------------ #
    # HIGH-LEVEL APIS
    # ------------------------------------------------------------------ #

    def assign_script(
        self,
        employee: Employee,
        local_path: str,
        remote_path: str = "/tmp/ara_fleet_job.py",
        python: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Copy a local script to the employee and run it.

        Returns stdout on success, or None on failure.
        """
        py = python or self.default_python
        cmd = f"{py} {remote_path}"
        return self._run_remote(
            employee,
            cmd,
            upload_local_path=local_path,
            remote_path=remote_path,
            env=env,
        )

    def run_inline(
        self,
        employee: Employee,
        code: str,
        remote_path: str = "/tmp/ara_fleet_inline_job.py",
        python: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Create a temporary script on the remote side with the provided code
        and execute it.
        """
        py = python or self.default_python
        cmd = f"{py} {remote_path}"
        return self._run_remote(
            employee,
            cmd,
            inline_code=code,
            remote_path=remote_path,
            env=env,
        )

    def run_command(
        self,
        employee: Employee,
        command: str,
        env: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Run a raw command on the employee machine.

        Use with caution - prefer run_inline for Python tasks.
        """
        return self._run_remote(
            employee,
            command,
            env=env,
        )

    # ------------------------------------------------------------------ #
    # CORE SSH EXECUTION
    # ------------------------------------------------------------------ #

    def _run_remote(
        self,
        employee: Employee,
        command: str,
        upload_local_path: Optional[str] = None,
        inline_code: Optional[str] = None,
        remote_path: str = "/tmp/ara_fleet_job.py",
        env: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Core runner. Handles:
        - Creating SSH connection
        - Uploading code
        - Executing command
        - Returning stdout

        NOTE: This does *not* check allow_sudo / etc; that should be enforced
        by whichever layer decides what command to send.
        """
        if Connection is None:
            log.error(
                "Dispatcher: fabric.Connection not available. "
                "Install fabric or provide custom execution backend."
            )
            return None

        log.info(
            "DISPATCHER: Sending task to %s (%s) cmd=%s",
            employee.id,
            employee.role.value,
            command,
        )

        if self.dry_run:
            log.info("Dispatcher is in dry_run mode. Skipping execution.")
            return ""

        try:
            conn = Connection(
                host=employee.hostname,
                user=self.ssh_user,
                connect_timeout=10,
            )

            # Upload script if requested
            if upload_local_path is not None:
                log.debug("Dispatcher: uploading %s -> %s", upload_local_path, remote_path)
                conn.put(upload_local_path, remote=remote_path)

            if inline_code is not None:
                log.debug("Dispatcher: writing inline code to %s", remote_path)
                # Use base64 encoding for safe transfer (handles any content)
                encoded = base64.b64encode(inline_code.encode('utf-8')).decode('ascii')
                conn.run(
                    f"echo '{encoded}' | base64 -d > {remote_path}",
                    hide=True,
                )

            # Build environment prefix
            env_prefix = ""
            if env:
                exports = " ".join(
                    [f"{k}='{v}'" for k, v in env.items()]
                )
                env_prefix = f"{exports} "

            remote_cmd = f"{env_prefix}{command}"
            log.debug("Dispatcher: executing on %s: %s", employee.hostname, remote_cmd)

            result = conn.run(remote_cmd, hide=True, warn=True)  # warn=True prevents exception on non-zero
            stdout = result.stdout.strip()
            stderr = result.stderr.strip() if result.stderr else ""

            # Check exit code - non-zero means failure
            if result.exited != 0:
                log.error(
                    "DISPATCHER: Task on %s FAILED (exit=%s)\nstdout: %s\nstderr: %s",
                    employee.id,
                    result.exited,
                    stdout[:500] if stdout else "(empty)",
                    stderr[:500] if stderr else "(empty)",
                )
                employee.status = "error"
                return None

            log.info(
                "DISPATCHER: Task on %s completed (exit=%s, out_len=%d)",
                employee.id,
                result.exited,
                len(stdout),
            )

            return stdout

        except Exception as e:
            log.error("Dispatcher: failed to manage employee %s: %s", employee.id, e)
            employee.status = "error"
            return None

    # ------------------------------------------------------------------ #
    # BATCH OPERATIONS
    # ------------------------------------------------------------------ #

    def broadcast(
        self,
        employees: list[Employee],
        code: str,
        env: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Run the same code on multiple employees.

        Returns dict mapping employee_id -> stdout (or None on failure).
        """
        results = {}
        for emp in employees:
            results[emp.id] = self.run_inline(emp, code, env=env)
        return results

    def health_check(self, employee: Employee) -> bool:
        """
        Quick health check - run a simple command and verify response.
        """
        result = self.run_command(employee, "echo 'ara_fleet_ping'")
        return result is not None and "ara_fleet_ping" in result


# =============================================================================
# Convenience Functions
# =============================================================================

_default_dispatcher: Optional[Dispatcher] = None


def get_dispatcher(ssh_user: str = "ara_worker") -> Dispatcher:
    """Get the default Dispatcher instance."""
    global _default_dispatcher
    if _default_dispatcher is None:
        _default_dispatcher = Dispatcher(ssh_user=ssh_user)
    return _default_dispatcher


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'Dispatcher',
    'get_dispatcher',
]
