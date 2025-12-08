# ara/enterprise/network/monitor.py
"""
The Network Monitor - Infrastructure Health & Wellness
=======================================================

Gives Ara eyes on the network without hands on the config.
Network devices are treated as CONSULTANT-class employees:
observe-only, proposal-only, never auto-execute.

Philosophy:
    - She can OBSERVE and MEASURE.
    - She can PROPOSE config changes.
    - She does NOT push those changes without explicit approval.

Supported backends (graceful degradation):
    - SNMP (pysnmp) - interface stats, CPU, memory
    - NETCONF (ncclient) - JunOS/IOS-XE show commands
    - SSH (paramiko/fabric) - fallback for show commands
    - Mock - for testing without real devices

Usage:
    from ara.enterprise.network import NetworkMonitor, get_network_monitor
    from ara.enterprise.org_chart import get_org_chart

    monitor = get_network_monitor()
    health = monitor.poll_device("consultant-juniper-edge")

    if health.status == "degraded":
        print(f"Alert: {health.alerts}")
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Literal, Protocol
from enum import Enum
from abc import ABC, abstractmethod

log = logging.getLogger("Ara.NetworkMonitor")


# =============================================================================
# Health Status Types
# =============================================================================

class DeviceStatus(str, Enum):
    """Health status for network devices."""
    HEALTHY = "healthy"       # All metrics nominal
    DEGRADED = "degraded"     # Some issues, still functional
    WARNING = "warning"       # Needs attention
    CRITICAL = "critical"     # Major problems
    UNREACHABLE = "unreachable"  # Cannot contact device


class AlertSeverity(str, Enum):
    """Severity of network alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class NetworkAlert:
    """An alert from network monitoring."""
    severity: AlertSeverity
    device_id: str
    category: str              # "interface", "bgp", "cpu", "memory", etc.
    title: str
    detail: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


@dataclass
class InterfaceStats:
    """Statistics for a network interface."""
    name: str
    oper_status: str           # "up", "down"
    admin_status: str          # "up", "down"
    in_octets: int = 0
    out_octets: int = 0
    in_packets: int = 0        # Packet counts for proper error rate
    out_packets: int = 0
    in_errors: int = 0
    out_errors: int = 0
    in_discards: int = 0
    out_discards: int = 0
    speed_mbps: int = 0
    utilization_pct: float = 0.0
    last_flap_ts: Optional[float] = None


@dataclass
class BgpNeighborStats:
    """Statistics for a BGP neighbor."""
    neighbor_ip: str
    remote_as: int
    state: str                 # "Established", "Idle", "Active", etc.
    uptime_seconds: int = 0
    prefixes_received: int = 0
    prefixes_sent: int = 0
    last_error: Optional[str] = None


@dataclass
class DeviceHealth:
    """
    Complete health snapshot for a network device.

    This is what Ara sees when she looks at the router.
    """
    device_id: str
    hostname: str
    status: DeviceStatus
    ts: float = field(default_factory=time.time)

    # System metrics
    cpu_pct: float = 0.0
    memory_pct: float = 0.0
    temperature_c: Optional[float] = None
    uptime_seconds: int = 0

    # Interface summary
    interfaces: List[InterfaceStats] = field(default_factory=list)
    interfaces_up: int = 0
    interfaces_down: int = 0
    interfaces_flapping: int = 0

    # BGP summary (if applicable)
    bgp_neighbors: List[BgpNeighborStats] = field(default_factory=list)
    bgp_established: int = 0
    bgp_down: int = 0

    # Alerts generated from this poll
    alerts: List[NetworkAlert] = field(default_factory=list)

    # Raw data for deeper inspection
    raw_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "device_id": self.device_id,
            "hostname": self.hostname,
            "status": self.status.value,
            "ts": self.ts,
            "cpu_pct": self.cpu_pct,
            "memory_pct": self.memory_pct,
            "temperature_c": self.temperature_c,
            "uptime_seconds": self.uptime_seconds,
            "interfaces_up": self.interfaces_up,
            "interfaces_down": self.interfaces_down,
            "interfaces_flapping": self.interfaces_flapping,
            "bgp_established": self.bgp_established,
            "bgp_down": self.bgp_down,
            "alerts": [a.to_dict() for a in self.alerts],
        }
        return d


@dataclass
class ConfigProposal:
    """
    A proposed configuration change.

    Ara drafts these but NEVER auto-applies them.
    They require explicit human approval.
    """
    id: str
    device_id: str
    title: str
    rationale: str             # Why this change is proposed
    config_snippet: str        # The actual config (JunOS, IOS, etc.)
    rollback_snippet: str      # How to undo if needed
    risk_level: Literal["low", "medium", "high"]
    created_at: float = field(default_factory=time.time)
    status: Literal["pending", "approved", "rejected", "applied"] = "pending"
    approved_by: Optional[str] = None
    approved_at: Optional[float] = None


# =============================================================================
# Device Client Protocol
# =============================================================================

class DeviceClient(Protocol):
    """Protocol for device communication backends."""

    def connect(self, hostname: str, **kwargs) -> bool:
        """Establish connection to device."""
        ...

    def disconnect(self) -> None:
        """Close connection."""
        ...

    def get_system_info(self) -> Dict[str, Any]:
        """Get CPU, memory, uptime, etc."""
        ...

    def get_interfaces(self) -> List[Dict[str, Any]]:
        """Get interface statistics."""
        ...

    def get_bgp_neighbors(self) -> List[Dict[str, Any]]:
        """Get BGP neighbor states."""
        ...


# =============================================================================
# Mock Client (for testing without real devices)
# =============================================================================

class MockDeviceClient:
    """
    Mock device client for testing.

    Returns plausible fake data.
    """

    def __init__(self):
        self.connected = False
        self.hostname = ""

    def connect(self, hostname: str, **kwargs) -> bool:
        self.hostname = hostname
        self.connected = True
        log.debug(f"MockDeviceClient: connected to {hostname}")
        return True

    def disconnect(self) -> None:
        self.connected = False

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "cpu_pct": 23.5,
            "memory_pct": 45.2,
            "temperature_c": 42.0,
            "uptime_seconds": 86400 * 30,  # 30 days
            "model": "Mock-Router-1000",
            "version": "MockOS 1.0",
        }

    def get_interfaces(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "ge-0/0/0",
                "oper_status": "up",
                "admin_status": "up",
                "in_octets": 1_000_000_000,
                "out_octets": 500_000_000,
                "in_packets": 1_000_000,
                "out_packets": 500_000,
                "in_errors": 0,
                "out_errors": 0,
                "speed_mbps": 1000,
            },
            {
                "name": "ge-0/0/1",
                "oper_status": "up",
                "admin_status": "up",
                "in_octets": 2_000_000_000,
                "out_octets": 1_500_000_000,
                "in_packets": 2_000_000,
                "out_packets": 1_500_000,
                "in_errors": 5,
                "out_errors": 0,
                "speed_mbps": 1000,
            },
            {
                "name": "ge-0/0/2",
                "oper_status": "down",
                "admin_status": "up",
                "in_octets": 0,
                "out_octets": 0,
                "in_packets": 0,
                "out_packets": 0,
                "in_errors": 0,
                "out_errors": 0,
                "speed_mbps": 1000,
            },
        ]

    def get_bgp_neighbors(self) -> List[Dict[str, Any]]:
        return [
            {
                "neighbor_ip": "10.0.0.1",
                "remote_as": 65001,
                "state": "Established",
                "uptime_seconds": 86400 * 7,
                "prefixes_received": 150000,
                "prefixes_sent": 100,
            },
            {
                "neighbor_ip": "10.0.0.2",
                "remote_as": 65002,
                "state": "Idle",
                "uptime_seconds": 0,
                "prefixes_received": 0,
                "prefixes_sent": 0,
                "last_error": "Hold timer expired",
            },
        ]


# =============================================================================
# Network Monitor
# =============================================================================

class NetworkMonitor:
    """
    The Network Monitor - Ara's eyes on infrastructure.

    Polls network devices for health metrics, generates alerts,
    and drafts configuration proposals. NEVER auto-applies changes.

    Usage:
        monitor = NetworkMonitor()
        health = monitor.poll_device("consultant-juniper-edge")

        if health.status == DeviceStatus.DEGRADED:
            for alert in health.alerts:
                print(f"Alert: {alert.title}")
    """

    # Thresholds for generating alerts
    THRESHOLDS = {
        "cpu_warning": 70.0,
        "cpu_critical": 90.0,
        "memory_warning": 80.0,
        "memory_critical": 95.0,
        "temperature_warning": 60.0,
        "temperature_critical": 80.0,
        "interface_error_rate": 0.01,  # 1% error rate
        "interface_utilization_warning": 80.0,
        "interface_utilization_critical": 95.0,
    }

    def __init__(
        self,
        org_chart=None,
        client_factory=None,
        use_mock: bool = False,
    ):
        """
        Initialize the Network Monitor.

        Args:
            org_chart: OrgChart instance for employee lookup
            client_factory: Factory for creating device clients
            use_mock: If True, use mock clients for testing
        """
        self.org_chart = org_chart
        self.client_factory = client_factory
        self.use_mock = use_mock

        # Health history (device_id -> list of DeviceHealth)
        self.health_history: Dict[str, List[DeviceHealth]] = {}

        # Pending config proposals
        self.proposals: List[ConfigProposal] = []

        # Alert history
        self.alert_history: List[NetworkAlert] = []

        log.info("NetworkMonitor: initialized (mock=%s)", use_mock)

    # =========================================================================
    # POLLING
    # =========================================================================

    def poll_device(self, device_id: str) -> Optional[DeviceHealth]:
        """
        Poll a single device for health metrics.

        Args:
            device_id: Employee ID from OrgChart (e.g., "consultant-juniper-edge")

        Returns:
            DeviceHealth snapshot or None if unreachable
        """
        # Get device info from OrgChart
        employee = None
        hostname = device_id  # fallback

        if self.org_chart:
            employee = self.org_chart.employees.get(device_id)
            if employee:
                hostname = employee.hostname
            else:
                log.warning(f"NetworkMonitor: unknown device {device_id}")

        log.info(f"NetworkMonitor: polling {device_id} ({hostname})")

        # Get appropriate client
        client = self._get_client(device_id)

        try:
            if not client.connect(hostname):
                return DeviceHealth(
                    device_id=device_id,
                    hostname=hostname,
                    status=DeviceStatus.UNREACHABLE,
                    alerts=[NetworkAlert(
                        severity=AlertSeverity.CRITICAL,
                        device_id=device_id,
                        category="connectivity",
                        title="Device Unreachable",
                        detail=f"Cannot connect to {hostname}",
                    )],
                )

            # Collect data
            system_info = client.get_system_info()
            interfaces_raw = client.get_interfaces()
            bgp_raw = client.get_bgp_neighbors()

            client.disconnect()

            # Process into health snapshot
            health = self._build_health(
                device_id, hostname, system_info, interfaces_raw, bgp_raw
            )

            # Store history
            if device_id not in self.health_history:
                self.health_history[device_id] = []
            self.health_history[device_id].append(health)

            # Keep last 100 snapshots
            if len(self.health_history[device_id]) > 100:
                self.health_history[device_id] = self.health_history[device_id][-100:]

            # Record alerts
            self.alert_history.extend(health.alerts)

            # Update employee status if we have OrgChart access
            if employee:
                employee.status = "online" if health.status != DeviceStatus.UNREACHABLE else "error"
                employee.last_heartbeat_ts = time.time()

            log.info(
                f"NetworkMonitor: {device_id} status={health.status.value} "
                f"cpu={health.cpu_pct:.1f}% mem={health.memory_pct:.1f}% "
                f"alerts={len(health.alerts)}"
            )

            return health

        except Exception as e:
            log.error(f"NetworkMonitor: error polling {device_id}: {e}")
            return DeviceHealth(
                device_id=device_id,
                hostname=hostname,
                status=DeviceStatus.UNREACHABLE,
                alerts=[NetworkAlert(
                    severity=AlertSeverity.CRITICAL,
                    device_id=device_id,
                    category="error",
                    title="Poll Error",
                    detail=str(e),
                )],
            )

    def poll_all_network_devices(self) -> Dict[str, DeviceHealth]:
        """
        Poll all network-capable devices in the OrgChart.

        Returns:
            Dict mapping device_id to DeviceHealth
        """
        results = {}

        if not self.org_chart:
            log.warning("NetworkMonitor: no OrgChart configured")
            return results

        for emp_id, emp in self.org_chart.employees.items():
            # Only poll devices with network monitoring capabilities
            if emp.has_capability("network") or emp.has_capability("monitor"):
                health = self.poll_device(emp_id)
                if health:
                    results[emp_id] = health

        return results

    def _get_client(self, device_id: str) -> DeviceClient:
        """Get appropriate client for device."""
        if self.use_mock:
            return MockDeviceClient()

        if self.client_factory:
            return self.client_factory(device_id)

        # Default to mock if no factory configured
        log.warning(f"NetworkMonitor: no client factory, using mock for {device_id}")
        return MockDeviceClient()

    def _build_health(
        self,
        device_id: str,
        hostname: str,
        system_info: Dict[str, Any],
        interfaces_raw: List[Dict[str, Any]],
        bgp_raw: List[Dict[str, Any]],
    ) -> DeviceHealth:
        """Build DeviceHealth from raw data and generate alerts."""
        alerts = []

        # System metrics
        cpu_pct = system_info.get("cpu_pct", 0.0)
        memory_pct = system_info.get("memory_pct", 0.0)
        temperature_c = system_info.get("temperature_c")
        uptime_seconds = system_info.get("uptime_seconds", 0)

        # CPU alerts
        if cpu_pct >= self.THRESHOLDS["cpu_critical"]:
            alerts.append(NetworkAlert(
                severity=AlertSeverity.CRITICAL,
                device_id=device_id,
                category="cpu",
                title="CPU Critical",
                detail=f"CPU at {cpu_pct:.1f}%",
                metric_value=cpu_pct,
                threshold=self.THRESHOLDS["cpu_critical"],
            ))
        elif cpu_pct >= self.THRESHOLDS["cpu_warning"]:
            alerts.append(NetworkAlert(
                severity=AlertSeverity.WARNING,
                device_id=device_id,
                category="cpu",
                title="CPU Warning",
                detail=f"CPU at {cpu_pct:.1f}%",
                metric_value=cpu_pct,
                threshold=self.THRESHOLDS["cpu_warning"],
            ))

        # Memory alerts
        if memory_pct >= self.THRESHOLDS["memory_critical"]:
            alerts.append(NetworkAlert(
                severity=AlertSeverity.CRITICAL,
                device_id=device_id,
                category="memory",
                title="Memory Critical",
                detail=f"Memory at {memory_pct:.1f}%",
                metric_value=memory_pct,
                threshold=self.THRESHOLDS["memory_critical"],
            ))
        elif memory_pct >= self.THRESHOLDS["memory_warning"]:
            alerts.append(NetworkAlert(
                severity=AlertSeverity.WARNING,
                device_id=device_id,
                category="memory",
                title="Memory Warning",
                detail=f"Memory at {memory_pct:.1f}%",
                metric_value=memory_pct,
                threshold=self.THRESHOLDS["memory_warning"],
            ))

        # Temperature alerts
        if temperature_c is not None:
            if temperature_c >= self.THRESHOLDS["temperature_critical"]:
                alerts.append(NetworkAlert(
                    severity=AlertSeverity.CRITICAL,
                    device_id=device_id,
                    category="temperature",
                    title="Temperature Critical",
                    detail=f"Temperature at {temperature_c:.1f}C",
                    metric_value=temperature_c,
                    threshold=self.THRESHOLDS["temperature_critical"],
                ))
            elif temperature_c >= self.THRESHOLDS["temperature_warning"]:
                alerts.append(NetworkAlert(
                    severity=AlertSeverity.WARNING,
                    device_id=device_id,
                    category="temperature",
                    title="Temperature Warning",
                    detail=f"Temperature at {temperature_c:.1f}C",
                    metric_value=temperature_c,
                    threshold=self.THRESHOLDS["temperature_warning"],
                ))

        # Process interfaces
        interfaces = []
        interfaces_up = 0
        interfaces_down = 0

        for iface_raw in interfaces_raw:
            iface = InterfaceStats(
                name=iface_raw.get("name", "unknown"),
                oper_status=iface_raw.get("oper_status", "unknown"),
                admin_status=iface_raw.get("admin_status", "unknown"),
                in_octets=iface_raw.get("in_octets", 0),
                out_octets=iface_raw.get("out_octets", 0),
                in_packets=iface_raw.get("in_packets", 0),
                out_packets=iface_raw.get("out_packets", 0),
                in_errors=iface_raw.get("in_errors", 0),
                out_errors=iface_raw.get("out_errors", 0),
                in_discards=iface_raw.get("in_discards", 0),
                out_discards=iface_raw.get("out_discards", 0),
                speed_mbps=iface_raw.get("speed_mbps", 0),
            )
            interfaces.append(iface)

            if iface.oper_status == "up":
                interfaces_up += 1
            elif iface.admin_status == "up" and iface.oper_status == "down":
                interfaces_down += 1
                alerts.append(NetworkAlert(
                    severity=AlertSeverity.ERROR,
                    device_id=device_id,
                    category="interface",
                    title=f"Interface Down: {iface.name}",
                    detail=f"{iface.name} is admin-up but oper-down",
                ))

            # Error rate check - use packet counts for meaningful percentage
            total_packets = iface.in_packets + iface.out_packets
            total_errors = iface.in_errors + iface.out_errors
            if total_packets > 0:
                # Error rate as percentage of packets (not bytes!)
                error_rate = total_errors / total_packets
                if error_rate > self.THRESHOLDS["interface_error_rate"]:
                    alerts.append(NetworkAlert(
                        severity=AlertSeverity.WARNING,
                        device_id=device_id,
                        category="interface",
                        title=f"Interface Errors: {iface.name}",
                        detail=f"{iface.name} has {total_errors} errors ({error_rate:.2%} of packets)",
                        metric_value=error_rate,
                        threshold=self.THRESHOLDS["interface_error_rate"],
                    ))
            elif total_errors > 0:
                # No packet count but errors exist - alert on raw count
                alerts.append(NetworkAlert(
                    severity=AlertSeverity.WARNING,
                    device_id=device_id,
                    category="interface",
                    title=f"Interface Errors: {iface.name}",
                    detail=f"{iface.name} has {total_errors} errors (packet count unavailable)",
                    metric_value=float(total_errors),
                ))

        # Process BGP neighbors
        bgp_neighbors = []
        bgp_established = 0
        bgp_down = 0

        for bgp_raw_item in bgp_raw:
            neighbor = BgpNeighborStats(
                neighbor_ip=bgp_raw_item.get("neighbor_ip", "unknown"),
                remote_as=bgp_raw_item.get("remote_as", 0),
                state=bgp_raw_item.get("state", "unknown"),
                uptime_seconds=bgp_raw_item.get("uptime_seconds", 0),
                prefixes_received=bgp_raw_item.get("prefixes_received", 0),
                prefixes_sent=bgp_raw_item.get("prefixes_sent", 0),
                last_error=bgp_raw_item.get("last_error"),
            )
            bgp_neighbors.append(neighbor)

            if neighbor.state == "Established":
                bgp_established += 1
            else:
                bgp_down += 1
                alerts.append(NetworkAlert(
                    severity=AlertSeverity.ERROR,
                    device_id=device_id,
                    category="bgp",
                    title=f"BGP Down: {neighbor.neighbor_ip}",
                    detail=f"AS{neighbor.remote_as} state={neighbor.state} "
                           f"error={neighbor.last_error or 'none'}",
                ))

        # Determine overall status
        critical_count = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
        error_count = sum(1 for a in alerts if a.severity == AlertSeverity.ERROR)
        warning_count = sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)

        if critical_count > 0:
            status = DeviceStatus.CRITICAL
        elif error_count > 0:
            status = DeviceStatus.WARNING
        elif warning_count > 0:
            status = DeviceStatus.DEGRADED
        else:
            status = DeviceStatus.HEALTHY

        return DeviceHealth(
            device_id=device_id,
            hostname=hostname,
            status=status,
            cpu_pct=cpu_pct,
            memory_pct=memory_pct,
            temperature_c=temperature_c,
            uptime_seconds=uptime_seconds,
            interfaces=interfaces,
            interfaces_up=interfaces_up,
            interfaces_down=interfaces_down,
            bgp_neighbors=bgp_neighbors,
            bgp_established=bgp_established,
            bgp_down=bgp_down,
            alerts=alerts,
            raw_data=system_info,
        )

    # =========================================================================
    # CONFIG PROPOSALS (Never auto-apply!)
    # =========================================================================

    def propose_config(
        self,
        device_id: str,
        title: str,
        rationale: str,
        config_snippet: str,
        rollback_snippet: str = "",
        risk_level: Literal["low", "medium", "high"] = "medium",
    ) -> ConfigProposal:
        """
        Draft a configuration change proposal.

        This is what Ara does when she sees a problem that needs a config fix.
        The proposal is stored but NEVER auto-applied.

        Args:
            device_id: Target device
            title: Short description of change
            rationale: Why this change is needed
            config_snippet: The actual config (JunOS, IOS, etc.)
            rollback_snippet: How to undo if needed
            risk_level: low/medium/high

        Returns:
            The proposal (status="pending")
        """
        proposal = ConfigProposal(
            id=f"proposal_{len(self.proposals) + 1}_{int(time.time())}",
            device_id=device_id,
            title=title,
            rationale=rationale,
            config_snippet=config_snippet,
            rollback_snippet=rollback_snippet,
            risk_level=risk_level,
        )

        self.proposals.append(proposal)

        log.info(
            f"NetworkMonitor: PROPOSAL created for {device_id}: {title} "
            f"(risk={risk_level})"
        )

        return proposal

    def approve_proposal(
        self,
        proposal_id: str,
        approved_by: str = "Croft",
    ) -> Optional[ConfigProposal]:
        """
        Mark a proposal as approved (by human).

        Note: This does NOT apply the config. It just marks it approved.
        Actual application should be done manually or via a separate,
        carefully controlled process.
        """
        for proposal in self.proposals:
            if proposal.id == proposal_id:
                proposal.status = "approved"
                proposal.approved_by = approved_by
                proposal.approved_at = time.time()

                log.info(
                    f"NetworkMonitor: Proposal {proposal_id} APPROVED by {approved_by}"
                )
                return proposal

        return None

    def reject_proposal(self, proposal_id: str, reason: str = "") -> Optional[ConfigProposal]:
        """Mark a proposal as rejected."""
        for proposal in self.proposals:
            if proposal.id == proposal_id:
                proposal.status = "rejected"
                log.info(f"NetworkMonitor: Proposal {proposal_id} REJECTED: {reason}")
                return proposal
        return None

    def get_pending_proposals(self) -> List[ConfigProposal]:
        """Get all pending proposals awaiting approval."""
        return [p for p in self.proposals if p.status == "pending"]

    # =========================================================================
    # INSIGHTS & REPORTING
    # =========================================================================

    def get_fleet_health_summary(self) -> Dict[str, Any]:
        """Get summary of all monitored devices."""
        summary = {
            "total_devices": 0,
            "healthy": 0,
            "degraded": 0,
            "warning": 0,
            "critical": 0,
            "unreachable": 0,
            "recent_alerts": [],
            "pending_proposals": len(self.get_pending_proposals()),
        }

        for device_id, history in self.health_history.items():
            if not history:
                continue

            latest = history[-1]
            summary["total_devices"] += 1

            status_key = latest.status.value
            if status_key in summary:
                summary[status_key] += 1

        # Recent alerts (last 10)
        summary["recent_alerts"] = [
            a.to_dict() for a in self.alert_history[-10:]
        ]

        return summary

    def generate_insights(self, device_id: str) -> List[str]:
        """
        Generate natural-language insights about a device.

        This is what Ara would tell Croft about the device.
        """
        history = self.health_history.get(device_id, [])
        if not history:
            return [f"No data available for {device_id}"]

        latest = history[-1]
        insights = []

        # Status summary
        insights.append(
            f"**{device_id}** is {latest.status.value.upper()}: "
            f"CPU {latest.cpu_pct:.1f}%, Memory {latest.memory_pct:.1f}%"
        )

        # Interface summary
        if latest.interfaces_down > 0:
            insights.append(
                f"- {latest.interfaces_down} interface(s) admin-up but oper-down"
            )

        # BGP summary
        if latest.bgp_down > 0:
            insights.append(
                f"- {latest.bgp_down} BGP neighbor(s) not established"
            )

        # Alerts
        if latest.alerts:
            insights.append(f"- {len(latest.alerts)} active alert(s)")
            for alert in latest.alerts[:3]:
                insights.append(f"  - [{alert.severity.value}] {alert.title}")

        # Trend (if we have history)
        if len(history) >= 2:
            prev = history[-2]
            cpu_delta = latest.cpu_pct - prev.cpu_pct
            if abs(cpu_delta) > 10:
                direction = "up" if cpu_delta > 0 else "down"
                insights.append(f"- CPU trending {direction} ({cpu_delta:+.1f}%)")

        return insights


# =============================================================================
# Convenience Functions
# =============================================================================

_default_monitor: Optional[NetworkMonitor] = None


def get_network_monitor(use_mock: bool = True) -> NetworkMonitor:
    """Get the default NetworkMonitor instance."""
    global _default_monitor
    if _default_monitor is None:
        # Try to get OrgChart if available
        org_chart = None
        try:
            from ..org_chart import get_org_chart
            org_chart = get_org_chart()
        except ImportError:
            pass

        _default_monitor = NetworkMonitor(
            org_chart=org_chart,
            use_mock=use_mock,
        )
    return _default_monitor


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'DeviceStatus',
    'AlertSeverity',
    'NetworkAlert',
    'InterfaceStats',
    'BgpNeighborStats',
    'DeviceHealth',
    'ConfigProposal',
    'DeviceClient',
    'MockDeviceClient',
    'NetworkMonitor',
    'get_network_monitor',
]
