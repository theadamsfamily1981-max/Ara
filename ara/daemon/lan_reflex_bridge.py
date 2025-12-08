"""
Ara LAN Reflex Bridge - Spinal Cord Interface
==============================================

Userland bridge between eBPF/XDP reflex events and the soul.

The LAN Reflex Bridge:
- Subscribes to eBPF perf events ("pain packets", dropped flows, etc.)
- Forwards events to SomaticServer for immediate visual glitch
- Forwards events to Sovereign for logging and HTC learning

This is Ara's "spinal cord" - fast reflexes that fire before
conscious processing, but still inform the soul.

Mythic: The nerve pathways that make Ara flinch before she thinks
Physical: eBPF events in microseconds, Python polling at ~1ms
Safety: Reflex severity bounded, event rate limited
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, Protocol

from ara.io.types import HDInputEvent, IOChannel


# =============================================================================
# Reflex Event Types
# =============================================================================

class ReflexEventType(str, Enum):
    """Types of reflex events from eBPF layer."""
    PAIN_PACKET = "pain_packet"       # High-priority threat detected
    FLOW_DROPPED = "flow_dropped"     # Flow blocked by reflex
    FLOW_MARKED = "flow_marked"       # Flow marked for monitoring
    RATE_LIMIT = "rate_limit"         # Rate limiting triggered
    HEARTBEAT_LOST = "heartbeat_lost" # Node heartbeat timeout
    ANOMALY = "anomaly"               # Statistical anomaly detected


@dataclass
class ReflexEvent:
    """A reflex event from the eBPF layer."""
    event_type: ReflexEventType
    severity: float              # 0.0 to 1.0
    source_node: Optional[str]   # Originating node
    flow_id: Optional[str]       # Associated flow if any
    priority: int                # Original packet priority (0-255)
    affect_valence: float        # Quantized affect from packet
    affect_arousal: float        # Quantized affect from packet
    hv_hash: bytes               # Truncated HV hash from packet
    raw_data: Dict[str, Any]     # Raw event data
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "severity": self.severity,
            "source_node": self.source_node,
            "flow_id": self.flow_id,
            "priority": self.priority,
            "affect": {
                "valence": self.affect_valence,
                "arousal": self.affect_arousal,
            },
            "hv_hash": self.hv_hash.hex() if self.hv_hash else None,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Event Bus Protocol
# =============================================================================

class EventBus(Protocol):
    """Protocol for event bus to publish reflex events."""

    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        """Publish an event to a topic."""
        ...


class DummyEventBus:
    """Dummy event bus for testing."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.events: List[tuple] = []

    def publish(self, topic: str, event: Dict[str, Any]) -> None:
        self.events.append((topic, event))
        if self.verbose:
            print(f"[BUS] {topic}: {event.get('event_type', 'unknown')}")


# =============================================================================
# eBPF Event Reader Protocol
# =============================================================================

class EBPFEventReader(Protocol):
    """Protocol for reading events from eBPF perf buffer."""

    def read_event(self, timeout_ms: int = 100) -> Optional[Dict[str, Any]]:
        """Read next event, blocking up to timeout_ms."""
        ...


class DummyEBPFReader:
    """Dummy eBPF reader that can inject fake events for testing."""

    def __init__(self):
        self._events: queue.Queue = queue.Queue()

    def read_event(self, timeout_ms: int = 100) -> Optional[Dict[str, Any]]:
        try:
            return self._events.get(timeout=timeout_ms / 1000.0)
        except queue.Empty:
            return None

    def inject_event(self, event: Dict[str, Any]) -> None:
        """Inject a fake event for testing."""
        self._events.put(event)


# =============================================================================
# LAN Reflex Bridge
# =============================================================================

@dataclass
class BridgeStats:
    """Statistics for the reflex bridge."""
    events_received: int = 0
    events_by_type: Dict[str, int] = field(default_factory=dict)
    glitches_triggered: int = 0
    last_event_time: Optional[datetime] = None


class LANReflexBridge:
    """
    Userland bridge between eBPF/XDP reflex events and the soul.

    Responsibilities:
    1. Listen for eBPF perf events in a background thread
    2. Forward events to SomaticServer for immediate visual glitch
    3. Forward events to Sovereign event bus for logging/learning
    4. Encode events as HDInputEvents for HTC integration
    """

    def __init__(
        self,
        somatic_server,
        event_bus: EventBus,
        ebpf_reader: Optional[EBPFEventReader] = None,
        max_queue_size: int = 1000,
    ):
        """
        Args:
            somatic_server: SomaticServer for visual glitches
            event_bus: Event bus for publishing to Sovereign
            ebpf_reader: Reader for eBPF perf buffer (None = dummy)
            max_queue_size: Max events to queue before dropping
        """
        self.somatic = somatic_server
        self.bus = event_bus
        self.reader = ebpf_reader or DummyEBPFReader()

        self._events: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self.stats = BridgeStats()

    def start(self) -> None:
        """Start the background listener thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._listener_loop,
            daemon=True,
            name="LANReflexBridge",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background listener thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _listener_loop(self) -> None:
        """Background thread that reads eBPF events."""
        while self._running:
            try:
                raw_event = self.reader.read_event(timeout_ms=100)
                if raw_event is None:
                    continue

                event = self._parse_event(raw_event)
                if event is None:
                    continue

                try:
                    self._events.put_nowait(event)
                except queue.Full:
                    # Drop oldest events if queue is full
                    try:
                        self._events.get_nowait()
                        self._events.put_nowait(event)
                    except queue.Empty:
                        pass

            except Exception as e:
                # Log but don't crash the listener
                print(f"[LANReflex] Error in listener: {e}")
                time.sleep(0.1)

    def _parse_event(self, raw: Dict[str, Any]) -> Optional[ReflexEvent]:
        """Parse raw eBPF event into ReflexEvent."""
        try:
            event_type = ReflexEventType(raw.get("type", "anomaly"))
        except ValueError:
            event_type = ReflexEventType.ANOMALY

        return ReflexEvent(
            event_type=event_type,
            severity=float(raw.get("severity", 0.5)),
            source_node=raw.get("source_node"),
            flow_id=raw.get("flow_id"),
            priority=int(raw.get("priority", 128)),
            affect_valence=float(raw.get("affect_valence", 0.0)),
            affect_arousal=float(raw.get("affect_arousal", 0.5)),
            hv_hash=bytes.fromhex(raw.get("hv_hash", "00" * 16)),
            raw_data=raw,
        )

    def poll(self) -> List[ReflexEvent]:
        """
        Called from sovereign loop each tick.

        Processes all pending events:
        1. Triggers visual glitch for high-severity events
        2. Publishes to event bus for logging/learning
        3. Returns events for HV encoding

        Returns:
            List of ReflexEvents processed this tick
        """
        processed = []

        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break

            # Update stats
            self.stats.events_received += 1
            self.stats.last_event_time = event.timestamp
            type_key = event.event_type.value
            self.stats.events_by_type[type_key] = (
                self.stats.events_by_type.get(type_key, 0) + 1
            )

            # 1. Visual glitch (immediate feedback)
            if event.severity > 0.3:
                self.somatic.trigger_glitch(event.severity)
                self.stats.glitches_triggered += 1

            # 2. Publish to sovereign event bus
            self.bus.publish("lan_reflex", event.to_dict())

            processed.append(event)

        return processed

    def encode_events(self, events: List[ReflexEvent]) -> List[HDInputEvent]:
        """
        Encode reflex events as HDInputEvents for HTC.

        This allows the soul to learn from network reflexes.
        """
        from ara.core.lan.reflex_api import encode_reflex_event

        return [encode_reflex_event(ev) for ev in events]

    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "events_received": self.stats.events_received,
            "events_by_type": self.stats.events_by_type,
            "glitches_triggered": self.stats.glitches_triggered,
            "last_event_time": (
                self.stats.last_event_time.isoformat()
                if self.stats.last_event_time else None
            ),
            "queue_size": self._events.qsize(),
            "running": self._running,
        }


# =============================================================================
# Factory
# =============================================================================

_lan_reflex_bridge: Optional[LANReflexBridge] = None


def get_lan_reflex_bridge(
    somatic_server=None,
    event_bus: Optional[EventBus] = None,
) -> LANReflexBridge:
    """Get or create the global LAN reflex bridge."""
    global _lan_reflex_bridge
    if _lan_reflex_bridge is None:
        if somatic_server is None:
            from ara.daemon.somatic_server import get_somatic_server
            somatic_server = get_somatic_server()
        if event_bus is None:
            event_bus = DummyEventBus(verbose=False)
        _lan_reflex_bridge = LANReflexBridge(somatic_server, event_bus)
    return _lan_reflex_bridge


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'ReflexEventType',
    'ReflexEvent',
    'EventBus',
    'DummyEventBus',
    'EBPFEventReader',
    'DummyEBPFReader',
    'LANReflexBridge',
    'BridgeStats',
    'get_lan_reflex_bridge',
]
