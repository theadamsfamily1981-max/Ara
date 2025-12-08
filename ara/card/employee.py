"""
Neuromorphic Card as Ara Employee
=================================

Integrates the CardRuntime with Ara's OrgChart system,
making the neuromorphic card a first-class employee
in Corporation Croft.

The card becomes "worker-neuromorph-01" with capabilities:
- corr_spike_hdc: Hyperdimensional correlation
- snn_inference: Spiking neural network inference
- hebbian_learning: On-chip Hebbian plasticity
- telemetry_filter: Event filtering and local decisions

Usage:
    from ara.card.employee import NeuromorphCardEmployee

    employee = NeuromorphCardEmployee.create()
    employee.start()  # Begin processing telemetry
"""

import time
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from queue import Queue, Empty

from ara.enterprise.org_chart import Employee, EmployeeRole, OrgChart, get_org_chart
from ara.card.runtime import CardRuntime, CardConfig, CardEvent, Decision
from ara.bridge.messages import StateHPVQuery, NewPolicy

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class NeuromorphEmployeeConfig:
    """Configuration for the neuromorph employee."""
    employee_id: str = "worker-neuromorph-01"
    hostname: str = "localhost"  # Or actual FPGA host
    hv_dim: int = 8192
    heartbeat_interval_s: float = 30.0
    max_pending_escalations: int = 100


# ============================================================================
# Neuromorph Employee
# ============================================================================

class NeuromorphCardEmployee:
    """
    The neuromorphic card as an Ara employee.

    Wraps CardRuntime and integrates with OrgChart for:
    - Automatic registration as worker-neuromorph-01
    - Heartbeat reporting
    - Task tracking
    - Escalation routing to GPU workers
    """

    def __init__(
        self,
        config: Optional[NeuromorphEmployeeConfig] = None,
        org_chart: Optional[OrgChart] = None
    ):
        self.config = config or NeuromorphEmployeeConfig()
        self.org_chart = org_chart or get_org_chart()

        # Create card runtime
        card_config = CardConfig(hv_dim=self.config.hv_dim)
        self.runtime = CardRuntime(card_config)

        # Escalation queue
        self._escalation_queue: Queue = Queue(maxsize=self.config.max_pending_escalations)

        # Wire up escalation callback
        self.runtime.set_escalation_callback(self._on_escalation)

        # Running state
        self._running = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._escalation_thread: Optional[threading.Thread] = None

        # Escalation handler (set by user)
        self._escalation_handler: Optional[Callable[[StateHPVQuery], Optional[NewPolicy]]] = None

        # Register with org chart
        self._register()

    def _register(self) -> None:
        """Register as an employee in the OrgChart."""
        employee = Employee(
            id=self.config.employee_id,
            hostname=self.config.hostname,
            role=EmployeeRole.WORKER,
            capabilities=[
                "corr_spike_hdc",      # CorrSpike-HDC kernel
                "snn_inference",       # Spiking neural network
                "hebbian_learning",    # On-chip plasticity
                "hdc_encoding",        # Hyperdimensional encoding
                "telemetry_filter",    # Event filtering
                "anomaly_detection",   # Anomaly detection
                "local_policy",        # Local policy execution
                "python",              # Python runtime
            ],
            allow_sudo=False,
            allow_internet=False,
            allow_write_local_disk=True,
            labels={
                "tier": "prod",
                "subsystem": "neuromorphic",
                "role": "subcortex",
                "hv_dim": str(self.config.hv_dim)
            },
            status="offline"
        )

        self.org_chart.hire(employee)
        logger.info(f"Registered neuromorph employee: {self.config.employee_id}")

    @classmethod
    def create(
        cls,
        employee_id: str = "worker-neuromorph-01",
        hostname: str = "localhost",
        hv_dim: int = 8192
    ) -> 'NeuromorphCardEmployee':
        """Create a new neuromorph employee."""
        config = NeuromorphEmployeeConfig(
            employee_id=employee_id,
            hostname=hostname,
            hv_dim=hv_dim
        )
        return cls(config)

    def set_escalation_handler(
        self,
        handler: Callable[[StateHPVQuery], Optional[NewPolicy]]
    ) -> None:
        """
        Set the handler for escalations.

        The handler receives STATE_HPV_QUERY and should return
        NEW_POLICY (or None if no policy update needed).

        This is typically the CortexBridge.handle_query method.
        """
        self._escalation_handler = handler

    def start(self) -> None:
        """Start the employee (heartbeat + escalation processing)."""
        if self._running:
            return

        self._running = True

        # Update status
        self.org_chart.heartbeat(self.config.employee_id, "online")

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name=f"{self.config.employee_id}-heartbeat"
        )
        self._heartbeat_thread.start()

        # Start escalation processing thread
        self._escalation_thread = threading.Thread(
            target=self._escalation_loop,
            daemon=True,
            name=f"{self.config.employee_id}-escalation"
        )
        self._escalation_thread.start()

        logger.info(f"Started neuromorph employee: {self.config.employee_id}")

    def stop(self) -> None:
        """Stop the employee."""
        self._running = False

        # Update status
        self.org_chart.set_status(self.config.employee_id, "offline")

        # Wait for threads
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)
        if self._escalation_thread:
            self._escalation_thread.join(timeout=2.0)

        logger.info(f"Stopped neuromorph employee: {self.config.employee_id}")

    def ingest(self, source: str, event_type: str, data: Dict[str, Any]) -> Decision:
        """
        Ingest telemetry event.

        Returns the decision made (ignore, local_policy, escalate, etc.)
        """
        return self.runtime.ingest_telemetry(source, event_type, data)

    def install_policy(self, policy: NewPolicy) -> None:
        """Install a policy on the card."""
        self.runtime.install_policy(policy)

    def get_stats(self) -> Dict[str, Any]:
        """Get combined stats from runtime and employee."""
        runtime_stats = self.runtime.get_stats()
        return {
            **runtime_stats,
            'employee_id': self.config.employee_id,
            'pending_escalations': self._escalation_queue.qsize(),
            'status': self.org_chart.employees.get(self.config.employee_id, Employee(id="", hostname="", role=EmployeeRole.WORKER)).status
        }

    def _on_escalation(self, query: StateHPVQuery) -> None:
        """Handle escalation from card runtime."""
        try:
            self._escalation_queue.put_nowait(query)
        except:
            logger.warning("Escalation queue full, dropping escalation")

    def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while self._running:
            try:
                self.org_chart.heartbeat(self.config.employee_id, "online")
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            time.sleep(self.config.heartbeat_interval_s)

    def _escalation_loop(self) -> None:
        """Background escalation processing loop."""
        while self._running:
            try:
                query = self._escalation_queue.get(timeout=1.0)

                if self._escalation_handler:
                    try:
                        policy = self._escalation_handler(query)
                        if policy:
                            self.runtime.install_policy(policy)
                            logger.info(f"Installed policy from escalation: {policy.policy_id}")
                    except Exception as e:
                        logger.error(f"Escalation handler error: {e}")
                else:
                    logger.warning("No escalation handler set, escalation dropped")

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Escalation loop error: {e}")


# ============================================================================
# Full Symbiosis Setup
# ============================================================================

def create_symbiosis_stack(
    hv_dim: int = 8192,
    use_mock_llm: bool = True
) -> Dict[str, Any]:
    """
    Create a complete Card <-> Cortex symbiosis stack.

    Returns dict with:
    - 'card': NeuromorphCardEmployee
    - 'bridge': CortexBridge
    - 'org_chart': OrgChart

    Usage:
        stack = create_symbiosis_stack()
        stack['card'].start()

        # Ingest telemetry
        stack['card'].ingest('router', 'flap', {'count': 5})
    """
    import numpy as np
    from ara.bridge.cortex_bridge import CortexBridge, MockLLMBackend, PolicyCompiler
    from ara.hdc.probe import HDProbe, HDProbeConfig

    # Get org chart
    org_chart = get_org_chart()

    # Create codebook
    rng = np.random.default_rng(42)
    codebook = {
        'POLICY': rng.choice([-1, 1], size=hv_dim).astype(np.int8),
        'RISK_LOW': rng.choice([-1, 1], size=hv_dim).astype(np.int8),
        'RISK_MEDIUM': rng.choice([-1, 1], size=hv_dim).astype(np.int8),
        'RISK_HIGH': rng.choice([-1, 1], size=hv_dim).astype(np.int8),
        'THROTTLE': rng.choice([-1, 1], size=hv_dim).astype(np.int8),
        'ALERT': rng.choice([-1, 1], size=hv_dim).astype(np.int8),
        'BLOCK': rng.choice([-1, 1], size=hv_dim).astype(np.int8),
        'ROUTE_FLAP': rng.choice([-1, 1], size=hv_dim).astype(np.int8),
        'CPU_SPIKE': rng.choice([-1, 1], size=hv_dim).astype(np.int8),
        'MEMORY_PRESSURE': rng.choice([-1, 1], size=hv_dim).astype(np.int8),
        'NETWORK_ERROR': rng.choice([-1, 1], size=hv_dim).astype(np.int8),
        'SECURITY': rng.choice([-1, 1], size=hv_dim).astype(np.int8),
    }

    # Create probe
    probe = HDProbe(HDProbeConfig(dim=hv_dim))
    for name, vec in codebook.items():
        if not name.startswith('RISK_') and name != 'POLICY':
            probe.add_concept(name, vec)

    # Create compiler
    compiler = PolicyCompiler(codebook)

    # Create LLM backend
    llm = MockLLMBackend() if use_mock_llm else MockLLMBackend()  # TODO: real LLM

    # Create bridge
    bridge = CortexBridge(llm, probe, compiler)

    # Create card employee
    card = NeuromorphCardEmployee.create(hv_dim=hv_dim)
    card.set_escalation_handler(bridge.handle_query)

    return {
        'card': card,
        'bridge': bridge,
        'org_chart': org_chart,
        'codebook': codebook,
        'probe': probe
    }


# ============================================================================
# Demo
# ============================================================================

def demo():
    """Demonstrate neuromorph employee integration."""
    print("=" * 60)
    print("Neuromorph Employee Demo")
    print("=" * 60)

    # Create full stack
    print("\n--- Creating Symbiosis Stack ---")
    stack = create_symbiosis_stack(hv_dim=1024, use_mock_llm=True)

    card = stack['card']
    org_chart = stack['org_chart']

    # Start the card
    print("\n--- Starting Card Employee ---")
    card.start()

    # Show org chart
    print("\n--- OrgChart Employees ---")
    for emp in org_chart.list_employees():
        caps = ', '.join(emp.capabilities[:3]) + ('...' if len(emp.capabilities) > 3 else '')
        print(f"  {emp.id}: {emp.role.value} @ {emp.hostname} [{caps}]")

    # Simulate telemetry
    print("\n--- Simulating Telemetry ---")
    import random
    random.seed(42)

    for i in range(20):
        if i % 5 == 0:
            # Anomalous
            decision = card.ingest(
                source=f"router_{i%3}",
                event_type='error',
                data={'severity': random.uniform(0.7, 1.0)}
            )
        else:
            # Normal
            decision = card.ingest(
                source=f"host_{i%5}",
                event_type='metric',
                data={'cpu': random.uniform(0.1, 0.5)}
            )

        if decision in (Decision.ESCALATE, Decision.EMERGENCY):
            print(f"  Event {i}: {decision.value}")

    # Give time for escalations to process
    time.sleep(0.5)

    # Show stats
    print(f"\n--- Card Stats ---")
    stats = card.get_stats()
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Show bridge stats
    print(f"\n--- Bridge Stats ---")
    bridge_stats = stack['bridge'].get_stats()
    for k, v in bridge_stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Stop
    card.stop()

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demo()
