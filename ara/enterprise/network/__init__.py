"""
Network Monitoring - Ara's Eyes on Infrastructure
==================================================

This package gives Ara visibility into network devices without
giving her configuration control. Network devices are treated
as CONSULTANT-class employees: observe-only, proposal-only.

Philosophy:
    - She can OBSERVE and MEASURE.
    - She can PROPOSE config changes.
    - She does NOT push those changes without explicit approval.

Components:
    - NetworkMonitor: Polls devices, generates alerts, drafts proposals
    - DeviceHealth: Health snapshot for a device
    - ConfigProposal: Drafted config change awaiting approval

Supported backends (graceful degradation):
    - SNMP (pysnmp) - interface stats, CPU, memory
    - NETCONF (ncclient) - JunOS/IOS-XE show commands
    - SSH (paramiko/fabric) - fallback for show commands
    - Mock - for testing without real devices

Usage:
    from ara.enterprise.network import NetworkMonitor, get_network_monitor

    monitor = get_network_monitor()
    health = monitor.poll_device("consultant-juniper-edge")

    if health.status == DeviceStatus.DEGRADED:
        for alert in health.alerts:
            print(f"Alert: {alert.title}")

    # Ara proposes a fix (NEVER auto-applied)
    proposal = monitor.propose_config(
        device_id="consultant-juniper-edge",
        title="Add QoS for print traffic",
        rationale="Print jobs are starving during peak hours",
        config_snippet=\"\"\"
set class-of-service interfaces ge-0/0/1 scheduler-map print-priority
set class-of-service scheduler-maps print-priority forwarding-class best-effort scheduler be-scheduler
\"\"\",
        risk_level="low",
    )

    # Croft reviews and approves
    monitor.approve_proposal(proposal.id, approved_by="Croft")
"""

from .monitor import (
    DeviceStatus,
    AlertSeverity,
    NetworkAlert,
    InterfaceStats,
    BgpNeighborStats,
    DeviceHealth,
    ConfigProposal,
    DeviceClient,
    MockDeviceClient,
    NetworkMonitor,
    get_network_monitor,
)

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
