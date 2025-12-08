#!/usr/bin/env python3
"""
Fleet Topology Demo
====================

Demonstrates the robustness layers of the Fleet:
1. Topology - Node roles and relationships
2. Watcher - Out-of-band safety controller
3. Brainstem - Always-on orchestrator
4. Safety - Cryptographic gatekeeper
5. Power - UPS + PDU management

Run: python -m ara.fleet.demo
"""

import time
from typing import Dict, Any

from .topology import (
    FleetRole, AuthLevel, NodeCapability,
    FleetNode, FleetTopology, create_default_topology,
)
from .watcher import (
    Watcher, WatcherState, PowerAction,
    create_watcher,
)
from .brainstem import (
    Brainstem, ScheduledJob, JobState, JobPriority,
    create_brainstem,
)
from .safety import (
    CryptoGatekeeper, SafetyPolicy, ActionSeverity,
    create_safety_policy,
)
from .power import (
    PowerSpine, UPS, SmartPDU, PowerState,
    create_power_spine,
)


def demo_topology():
    """Demonstrate fleet topology."""
    print("=" * 70)
    print("Fleet Topology")
    print("=" * 70)
    print()

    topology = create_default_topology()

    print("Nodes by role:")
    print("-" * 40)
    for role in FleetRole:
        nodes = topology.get_by_role(role)
        if nodes:
            print(f"  {role.name}:")
            for node in nodes:
                caps = ", ".join(c.name for c in node.capabilities) if node.capabilities else "none"
                print(f"    {node.node_id} ({node.hostname}) - caps: {caps}")
    print()

    # Show control tree
    brainstem = topology.get_brainstem()
    if brainstem:
        print("Control tree (from Brainstem):")
        print("-" * 40)
        tree = topology.get_control_tree(brainstem.node_id)
        print_tree(tree, indent=2)
    print()

    # Summary
    summary = topology.summary()
    print("Summary:")
    print(f"  Total nodes: {summary['total_nodes']}")
    print(f"  Healthy: {summary['healthy_nodes']}")
    print(f"  Roles: {summary['roles']}")
    print()


def print_tree(tree: Dict[str, Any], indent: int = 0):
    """Print a control tree."""
    prefix = " " * indent
    print(f"{prefix}├── {tree['node']} ({tree['role']})")
    for child in tree.get("controls", []):
        print_tree(child, indent + 4)


def demo_watcher():
    """Demonstrate the Watcher (out-of-band safety)."""
    print("=" * 70)
    print("Watcher - Out-of-Band Safety Controller")
    print("=" * 70)
    print()

    watcher = create_watcher()

    # Register nodes
    nodes = ["gpu-worker-01", "fpga-worker-01", "intern-host-01"]
    for node in nodes:
        watcher.register_node(node)

    print("Monitoring nodes:", nodes)
    print()

    # Simulate health reports
    print("Scenario: Normal operation")
    print("-" * 40)
    for node in nodes:
        watcher.report_ping(node, success=True)
        watcher.report_temp(node, temp_c=65.0)

    print(f"State: {watcher.get_state().name}")
    print()

    # Simulate GPU overheating
    print("Scenario: GPU overheating")
    print("-" * 40)
    for i in range(6):
        watcher.report_temp("gpu-worker-01", temp_c=75 + i * 5)
        print(f"  Temp report {i+1}: {75 + i * 5}°C")

    print(f"State: {watcher.get_state().name}")

    events = watcher.get_recent_events(5)
    print("Recent events:")
    for event in events:
        print(f"  {event.event_type}: {event.node_id} = {event.value}")
    print()

    # Simulate ping failures
    print("Scenario: Node unresponsive")
    print("-" * 40)
    for i in range(12):
        watcher.report_ping("intern-host-01", success=False)

    print(f"State: {watcher.get_state().name}")
    node_health = watcher.get_node_health("intern-host-01")
    print(f"Ping failures: {node_health.ping_failures}")
    print()

    # Status summary
    status = watcher.get_status()
    print("Watcher status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()


def demo_brainstem():
    """Demonstrate the Brainstem (orchestrator)."""
    print("=" * 70)
    print("Brainstem - Always-On Orchestrator")
    print("=" * 70)
    print()

    brainstem = create_brainstem()

    # Register nodes
    nodes = ["gpu-worker-01", "fpga-worker-01", "intern-host-01"]
    for node in nodes:
        brainstem.register_node(node)
        brainstem.report_heartbeat(node, load=0.2, memory_gb=32.0)

    print("Registered nodes:", nodes)
    print("All nodes online:", len(brainstem.get_online_nodes()))
    print()

    # Submit jobs
    print("Submitting jobs:")
    print("-" * 40)

    jobs = [
        ScheduledJob(
            job_id="job-001",
            name="Train model",
            target_node="gpu-worker-01",
            command="python train.py",
            priority=JobPriority.HIGH,
            requires_gpu=True,
        ),
        ScheduledJob(
            job_id="job-002",
            name="Run HSF demo",
            target_node="fpga-worker-01",
            command="python -m ara.cognition.hsf.demo",
            priority=JobPriority.NORMAL,
        ),
        ScheduledJob(
            job_id="job-003",
            name="Test package",
            target_node="intern-host-01",
            command="pip install suspicious-package",
            priority=JobPriority.LOW,
        ),
    ]

    for job in jobs:
        brainstem.submit_job(job)
        print(f"  Submitted: {job.name} -> {job.target_node}")
    print()

    # Wire up dispatch (simulated)
    def dispatch(node_id: str, job: ScheduledJob) -> bool:
        print(f"  Dispatching '{job.name}' to {node_id}")
        return True

    brainstem.wire_dispatch(dispatch)

    # Tick scheduler
    print("Running scheduler tick:")
    print("-" * 40)
    dispatched = brainstem.tick()
    print(f"  Dispatched {len(dispatched)} jobs")
    print()

    # Simulate job completion
    print("Simulating job completion:")
    print("-" * 40)
    brainstem.report_job_started("job-001", "gpu-worker-01")
    brainstem.report_job_completed("job-001", exit_code=0, output="Training complete")
    print("  job-001: completed")

    brainstem.report_job_started("job-003", "intern-host-01")
    brainstem.report_job_failed("job-003", error="Package install failed - dependency conflict")
    print("  job-003: failed (will retry)")
    print()

    # Status
    status = brainstem.get_status()
    print("Brainstem status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()


def demo_safety():
    """Demonstrate the safety/crypto gatekeeper."""
    print("=" * 70)
    print("Safety - Cryptographic Gatekeeper")
    print("=" * 70)
    print()

    gatekeeper = CryptoGatekeeper()
    policy = create_safety_policy()

    # Register a key
    gatekeeper.register_key("yubikey-01")

    print("Safety policy:")
    print(f"  Always require approval: {policy.always_approve}")
    print(f"  Protected targets: {policy.protected_targets}")
    print(f"  Severity threshold: {policy.severity_threshold.name}")
    print()

    # Test policy checks
    print("Policy checks:")
    print("-" * 40)

    test_cases = [
        ("read_logs", "gpu-worker-01", ActionSeverity.LOW),
        ("restart_service", "gpu-worker-01", ActionSeverity.MEDIUM),
        ("modify_config", "intern-host-01", ActionSeverity.HIGH),
        ("delete_data", "archivist-01", ActionSeverity.CRITICAL),
        ("modify_router_config", "juniper-01", ActionSeverity.HIGH),
    ]

    for action, target, severity in test_cases:
        requires, reason = policy.requires_approval(action, target, severity)
        status = "REQUIRE APPROVAL" if requires else "allowed"
        print(f"  {action} on {target}: {status}")
        if requires:
            print(f"    -> {reason}")
    print()

    # Request approval for critical action
    print("Approval flow for Juniper config change:")
    print("-" * 40)

    request = gatekeeper.request_approval(
        action_type="modify_router_config",
        target="juniper-01",
        description="Update ACL to allow new subnet",
        severity=ActionSeverity.CRITICAL,
        requester="ara-architect",
        reason="Need to allow traffic from new GPU cluster",
        simulation_result="ACL will add: permit 10.0.5.0/24",
        diff="""+ permit ip 10.0.5.0/24 any
+ permit ip any 10.0.5.0/24""",
        risk_assessment="Low risk - additive change only",
    )

    print(f"  Request ID: {request.request_id}")
    print(f"  Pending approvals: {len(gatekeeper.get_pending_requests())}")
    print()

    # Approve
    print("Human approves with hardware key tap:")
    signed_action = gatekeeper.approve(
        request.request_id,
        approver="croft",
        key_id="yubikey-01",
    )

    if signed_action:
        print(f"  Action signed: {signed_action.action_id}")
        print(f"  Expires in: {signed_action.expires_at - time.time():.0f}s")

        # Verify
        is_valid, reason = gatekeeper.verify(signed_action.action_id)
        print(f"  Verification: {'VALID' if is_valid else 'INVALID'} ({reason})")

        # Execute
        success, msg = gatekeeper.execute(signed_action.action_id)
        print(f"  Execution: {msg}")
    print()

    # Status
    status = gatekeeper.get_status()
    print("Gatekeeper status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()


def demo_power():
    """Demonstrate power spine management."""
    print("=" * 70)
    print("Power Spine - UPS + PDU Management")
    print("=" * 70)
    print()

    spine = create_power_spine()

    print("Power spine configuration:")
    print("-" * 40)
    status = spine.get_status()
    print(f"  UPS devices: {status['ups_count']}")
    print(f"  PDU devices: {status['pdu_count']}")
    print(f"  Mapped nodes: {status['mapped_nodes']}")
    print()

    # Show PDU outlets
    for pdu_id, pdu in spine.pdu_devices.items():
        print(f"PDU: {pdu.name}")
        for outlet_id, outlet in pdu.outlets.items():
            protected = " [PROTECTED]" if outlet.is_protected else ""
            node = outlet.connected_node or "unassigned"
            print(f"  {outlet_id}: {outlet.name} -> {node}{protected}")
    print()

    # Simulate normal operation
    print("Normal operation:")
    print("-" * 40)
    ups = spine.ups_devices["ups-01"]
    ups.update_status(
        on_battery=False,
        battery_percent=100.0,
        runtime_seconds=1800,
        load_watts=800,
        input_voltage=121.5,
    )
    print(f"  UPS state: {ups.state.name}")
    print(f"  Battery: {ups.battery_percent}%")
    print(f"  Load: {ups.load_watts}W")
    print()

    # Simulate power outage
    print("Scenario: Power outage")
    print("-" * 40)

    def on_power_lost():
        print("  [CALLBACK] Power lost! Initiating graceful shutdown...")

    def on_low_battery():
        print("  [CALLBACK] Battery low! Emergency measures needed.")

    ups.wire_callbacks(
        on_power_lost=on_power_lost,
        on_low_battery=on_low_battery,
    )

    # Power goes out
    ups.update_status(
        on_battery=True,
        battery_percent=95.0,
        runtime_seconds=1500,
        load_watts=800,
    )
    print(f"  UPS state: {ups.state.name}")
    print(f"  Runtime remaining: {ups.runtime_seconds / 60:.0f} minutes")
    print()

    # Battery draining
    print("Battery draining:")
    print("-" * 40)
    for pct in [80, 50, 20, 10, 5]:
        ups.update_status(
            on_battery=True,
            battery_percent=pct,
            runtime_seconds=pct * 15,
            load_watts=800,
        )
        print(f"  Battery: {pct}% -> State: {ups.state.name}")

        if ups.should_shutdown():
            print("  [ALERT] Emergency shutdown required!")
            break
    print()

    # Power cycle a node
    print("Power cycling a node:")
    print("-" * 40)
    success = spine.power_cycle_node("intern-host-01")
    print(f"  Power cycle intern-host-01: {'success' if success else 'failed'}")
    print()

    # Try to power off protected node
    print("Attempting to power off protected node (brainstem):")
    print("-" * 40)
    # This should fail because brainstem outlet is protected
    pdu = spine.pdu_devices["pdu-01"]
    success = pdu.switch_outlet("outlet-4", False)
    print(f"  Power off brainstem: {'success' if success else 'REFUSED (protected)'}")
    print()


def demo_integrated():
    """Demonstrate integrated fleet operation."""
    print("=" * 70)
    print("Integrated Fleet Operation")
    print("=" * 70)
    print()

    # Create all components
    topology = create_default_topology()
    watcher = create_watcher()
    brainstem = create_brainstem()
    gatekeeper = CryptoGatekeeper()
    power = create_power_spine()

    print("Fleet components initialized:")
    print(f"  Nodes: {len(topology.nodes)}")
    print(f"  Watcher monitoring: {len(watcher._nodes)}")
    print(f"  Power outlets: {sum(len(p.outlets) for p in power.pdu_devices.values())}")
    print()

    # Wire up watcher to power
    def power_off(node_id: str) -> bool:
        return power.power_off_node(node_id)

    def power_on(node_id: str) -> bool:
        return power.power_on_node(node_id)

    def power_cycle(node_id: str) -> bool:
        return power.power_cycle_node(node_id)

    watcher.wire_power_control(power_off, power_on, power_cycle)

    # Register nodes with all systems
    for node_id in topology.nodes:
        if node_id not in ["brainstem-01", "watcher-01", "archivist-01"]:
            watcher.register_node(node_id)
            brainstem.register_node(node_id)
            brainstem.report_heartbeat(node_id)

    print("Scenario: GPU worker becomes unresponsive")
    print("-" * 40)

    # Simulate ping failures
    for i in range(15):
        watcher.report_ping("gpu-worker-01", success=False)

    print(f"  Ping failures: {watcher.get_node_health('gpu-worker-01').ping_failures}")
    print(f"  Watcher state: {watcher.get_state().name}")
    print()

    # Check if action was taken
    events = watcher.get_recent_events(5)
    print("Watcher events:")
    for event in events:
        if event.action_taken:
            print(f"  {event.event_type}: {event.action_taken.name}")
        else:
            print(f"  {event.event_type}: {event.details}")
    print()

    print("=" * 70)
    print("Fleet is robust:")
    print("  - Watcher monitors health, can kill power")
    print("  - Brainstem orchestrates, stays alive")
    print("  - Gatekeeper requires approval for critical actions")
    print("  - Power spine enables clean shutdowns")
    print("  - Protected nodes can't be accidentally powered off")
    print("=" * 70)


def run_demo():
    """Run all demos."""
    print()
    print("=" * 70)
    print("FLEET TOPOLOGY DEMO")
    print("The Skeleton of the Cathedral")
    print("=" * 70)
    print()

    demo_topology()
    demo_watcher()
    demo_brainstem()
    demo_safety()
    demo_power()
    demo_integrated()

    print()
    print("Demo complete.")
    print()
    print("The Fleet provides:")
    print("  - Watcher: Out-of-band kill switch")
    print("  - Archivist: Memory that survives")
    print("  - Brainstem: Orchestrator that stays online")
    print("  - Shield: Network observer without control")
    print("  - Power Spine: Survivable power events")
    print("  - Gatekeeper: Signed actions for critical ops")
    print()
    print("None compete with GPU or neuromorphic card.")
    print("They wrap the cathedral in skeleton + immune system.")


if __name__ == "__main__":
    run_demo()
