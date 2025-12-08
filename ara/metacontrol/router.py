"""
Metacontrol Router - Reflex Arcs & Skip Connections
====================================================

Routes signals through the stack based on coherence and urgency.
Implements reflex arcs that bypass normal processing in emergencies.

Arc Types:
- REFLEX ARC: L1 → L9 skip (hardware emergency overrides planning)
- PROPHET ARC: L9 → L1 skip (mission creativity propagates to reflexes)
- NORMAL: Standard layer-by-layer processing

The router uses AxisMundi coherence to decide routing.
"""

from __future__ import annotations
import logging
from typing import Dict, Optional, Tuple, Any, Callable, List
from dataclasses import dataclass
from enum import Enum, auto
import time

from ara.system.axis import AxisMundi, detect_coherence_crisis, stack_alignment
from ara.layers.l1_hardware import L1HardwareReflex, TelemetryPacket
from ara.layers.l9_mission import L9MissionControl, MissionMode

logger = logging.getLogger(__name__)


class RouteType(Enum):
    """Types of routing decisions."""
    NORMAL = auto()       # Standard layer-by-layer
    REFLEX = auto()       # L1 overrides L9 (emergency)
    PROPHET = auto()      # L9 propagates to L1 (creative flow)
    URGENT = auto()       # Fast-path with reduced processing
    BLOCKED = auto()      # System blocked/recovering


@dataclass
class RouteDecision:
    """Result of routing decision."""
    route_type: RouteType
    reason: str
    l1_energy: float = 0.0
    l9_energy: float = 0.0
    coherence: float = 0.0
    alignment: float = 0.0
    recommendations: Dict[str, Any] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = {}


@dataclass
class RouterConfig:
    """Configuration for the metacontrol router."""
    # Reflex arc thresholds
    l1_emergency_energy: float = 0.35    # L1 energy to trigger reflex
    coherence_crisis_threshold: float = 0.2  # Coherence below this = crisis

    # Prophet arc thresholds
    coherence_flow_threshold: float = 0.7  # Coherence above this = flow
    l9_creativity_threshold: float = 0.8   # L9 creativity to enable prophet

    # Timing
    reflex_cooldown_seconds: float = 5.0   # Cooldown after reflex arc
    prophet_cooldown_seconds: float = 10.0  # Cooldown after prophet arc

    # Notifications
    notify_on_arc: bool = True


class MetacontrolRouter:
    """
    Routes signals through the stack based on coherence and urgency.

    The router sits at the center of the AxisMundi and decides whether:
    - Normal processing should continue
    - A reflex arc should fire (L1 emergency override)
    - A prophet arc should propagate (L9 creativity to L1)

    Usage:
        axis = AxisMundi()
        l1 = L1HardwareReflex(axis)
        l9 = L9MissionControl(axis)
        router = MetacontrolRouter(axis, l1, l9)

        # In control loop:
        decision = router.route()
        if decision.route_type == RouteType.REFLEX:
            # Handle emergency
            ...
    """

    def __init__(
        self,
        axis: AxisMundi,
        l1_adapter: L1HardwareReflex,
        l9_adapter: L9MissionControl,
        config: Optional[RouterConfig] = None,
    ):
        """
        Initialize the metacontrol router.

        Args:
            axis: The AxisMundi global bus
            l1_adapter: L1 hardware reflex adapter
            l9_adapter: L9 mission control adapter
            config: Router configuration
        """
        self.axis = axis
        self.l1 = l1_adapter
        self.l9 = l9_adapter
        self.config = config or RouterConfig()

        # State
        self._last_reflex_time = 0.0
        self._last_prophet_time = 0.0
        self._reflex_count = 0
        self._prophet_count = 0
        self._route_history: List[Tuple[float, RouteDecision]] = []

        # Callbacks
        self._arc_listeners: List[Callable[[RouteDecision], None]] = []

        logger.info("MetacontrolRouter initialized")

    def on_arc(self, callback: Callable[[RouteDecision], None]):
        """Register a callback for arc events."""
        self._arc_listeners.append(callback)

    def _notify_arc(self, decision: RouteDecision):
        """Notify listeners of arc event."""
        if not self.config.notify_on_arc:
            return

        for listener in self._arc_listeners:
            try:
                listener(decision)
            except Exception as e:
                logger.warning(f"Arc listener error: {e}")

    def route(self) -> RouteDecision:
        """
        Determine routing based on current stack state.

        Returns:
            RouteDecision with route type and metadata
        """
        # Get current state
        l1_energy = self.axis.get_layer_energy(1)
        l9_energy = self.axis.get_layer_energy(9)
        coherence = self.axis.coherence_between(1, 9)
        alignment = stack_alignment(self.axis)

        now = time.time()

        # Check for reflex arc (L1 emergency)
        is_crisis, crisis_details = detect_coherence_crisis(
            self.axis,
            l1_energy_threshold=self.config.l1_emergency_energy,
            coherence_threshold=self.config.coherence_crisis_threshold,
        )

        if is_crisis:
            # Check cooldown
            if now - self._last_reflex_time > self.config.reflex_cooldown_seconds:
                self._last_reflex_time = now
                self._reflex_count += 1

                # Fire reflex arc
                decision = self._fire_reflex_arc(
                    l1_energy, l9_energy, coherence, alignment, crisis_details
                )
                self._record_route(decision)
                self._notify_arc(decision)
                return decision

        # Check for prophet arc (L9 creativity flowing down)
        if coherence > self.config.coherence_flow_threshold:
            l9_creativity = self.l9.params.creativity
            if l9_creativity > self.config.l9_creativity_threshold:
                # Check cooldown
                if now - self._last_prophet_time > self.config.prophet_cooldown_seconds:
                    self._last_prophet_time = now
                    self._prophet_count += 1

                    # Fire prophet arc
                    decision = self._fire_prophet_arc(
                        l1_energy, l9_energy, coherence, alignment
                    )
                    self._record_route(decision)
                    self._notify_arc(decision)
                    return decision

        # Normal routing
        decision = RouteDecision(
            route_type=RouteType.NORMAL,
            reason="All layers coherent, normal processing",
            l1_energy=l1_energy,
            l9_energy=l9_energy,
            coherence=coherence,
            alignment=alignment,
        )
        self._record_route(decision)
        return decision

    def _fire_reflex_arc(
        self,
        l1_energy: float,
        l9_energy: float,
        coherence: float,
        alignment: float,
        crisis_details: Dict,
    ) -> RouteDecision:
        """
        Fire a reflex arc: L1 overrides L9.

        This happens when hardware is in distress but the planner
        doesn't "see" it (low coherence).
        """
        logger.warning(
            f"⚡ REFLEX ARC: L1 overriding L9 - "
            f"energy={l1_energy:.2f}, coherence={coherence:.2f}"
        )

        # Tell L9 to go to safe/emergency mode
        self.l9.respond_to_crisis(crisis_details)

        return RouteDecision(
            route_type=RouteType.REFLEX,
            reason=f"L1 emergency override - coherence crisis detected",
            l1_energy=l1_energy,
            l9_energy=l9_energy,
            coherence=coherence,
            alignment=alignment,
            recommendations={
                "action": "emergency_halt",
                "l9_mode": MissionMode.EMERGENCY.name,
                "reduce_load": True,
                "crisis_details": crisis_details,
            },
        )

    def _fire_prophet_arc(
        self,
        l1_energy: float,
        l9_energy: float,
        coherence: float,
        alignment: float,
    ) -> RouteDecision:
        """
        Fire a prophet arc: L9 creativity flows to L1.

        This happens when the stack is in flow (high coherence)
        and L9 is in creative mode - lower-level thresholds loosen.
        """
        logger.info(
            f"⚡ PROPHET ARC: L9 reconfiguring L1 - "
            f"coherence={coherence:.2f}, creativity={self.l9.params.creativity:.2f}"
        )

        return RouteDecision(
            route_type=RouteType.PROPHET,
            reason=f"L9 creativity propagating - flow state detected",
            l1_energy=l1_energy,
            l9_energy=l9_energy,
            coherence=coherence,
            alignment=alignment,
            recommendations={
                "action": "exploratory_mode",
                "l1_thresh_scale": 0.8,  # Lower thresholds = more exploratory
                "allow_novelty": True,
            },
        )

    def _record_route(self, decision: RouteDecision):
        """Record routing decision in history."""
        self._route_history.append((time.time(), decision))
        # Keep last 1000
        if len(self._route_history) > 1000:
            self._route_history = self._route_history[-1000:]

    def get_status(self) -> Dict[str, Any]:
        """Get router status."""
        return {
            "reflex_count": self._reflex_count,
            "prophet_count": self._prophet_count,
            "last_reflex_time": self._last_reflex_time,
            "last_prophet_time": self._last_prophet_time,
            "route_history_count": len(self._route_history),
            "current_l1_energy": self.axis.get_layer_energy(1),
            "current_l9_energy": self.axis.get_layer_energy(9),
            "current_coherence": self.axis.coherence_between(1, 9),
            "current_alignment": stack_alignment(self.axis),
        }

    def get_recent_routes(self, limit: int = 20) -> List[Dict]:
        """Get recent routing decisions."""
        return [
            {
                "timestamp": ts,
                "route_type": d.route_type.name,
                "reason": d.reason,
                "coherence": d.coherence,
                "alignment": d.alignment,
            }
            for ts, d in self._route_history[-limit:]
        ]


# =============================================================================
# Convenience Factory
# =============================================================================

def create_full_stack(
    hv_dim: int = 8192,
    num_layers: int = 9,
) -> Tuple[AxisMundi, L1HardwareReflex, L9MissionControl, MetacontrolRouter]:
    """
    Create a full Axis + Layer + Router stack.

    Returns:
        (axis, l1, l9, router)
    """
    axis = AxisMundi(dim=hv_dim, num_layers=num_layers)
    l1 = L1HardwareReflex(axis)
    l9 = L9MissionControl(axis)
    router = MetacontrolRouter(axis, l1, l9)

    return axis, l1, l9, router


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate the metacontrol router."""
    print("=" * 60)
    print("Metacontrol Router Demo")
    print("=" * 60)

    # Create full stack
    axis, l1, l9, router = create_full_stack(hv_dim=1024)

    # Callback for arc events
    def on_arc(decision: RouteDecision):
        print(f"\n🔔 ARC EVENT: {decision.route_type.name}")
        print(f"   Reason: {decision.reason}")

    router.on_arc(on_arc)

    print("\n--- Scenario 1: Normal Operation ---")
    telemetry = TelemetryPacket(temp_c=45, load=0.3, error_rate=0.001)
    l1.step(telemetry)
    l9.set_mode(MissionMode.BALANCED)
    l9.step()
    axis.tick()

    decision = router.route()
    print(f"Route: {decision.route_type.name}")
    print(f"Coherence: {decision.coherence:.3f}")

    print("\n--- Scenario 2: L1 Emergency (Reflex Arc) ---")
    telemetry = TelemetryPacket(temp_c=88, load=0.97, error_rate=0.08)
    l1.step(telemetry)
    axis.tick()

    decision = router.route()
    print(f"Route: {decision.route_type.name}")
    print(f"Recommendations: {decision.recommendations}")

    print("\n--- Scenario 3: Creative Flow (Prophet Arc) ---")
    # Reset to normal
    telemetry = TelemetryPacket(temp_c=45, load=0.3, error_rate=0.001)
    l1.step(telemetry)
    l9.set_mode(MissionMode.CREATIVE)
    l9.step()

    # Build coherence over several ticks
    for _ in range(5):
        l1.step(telemetry)
        l9.step()
        axis.tick()

    # Manually boost coherence for demo
    # (In real use, coherence builds naturally)
    decision = router.route()
    print(f"Route: {decision.route_type.name}")
    print(f"Coherence: {decision.coherence:.3f}")

    print("\n--- Router Status ---")
    status = router.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
