"""
Ara HD I/O Types - Core Data Structures
=======================================

Unified type system for all HD-based I/O channels.

Design principle: Everything that touches HTC is either:
- HDInputEvent: Goes INTO the HTC (moment construction, episodic memory)
- HDOutputHint: Comes OUT of the HTC (affect, focus, policies)

Both graphics and networking use the same HD vocabulary,
enabling deep integration as cognitive organs.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Union

# Hypervector type alias
HV = np.ndarray  # shape (DIM,), dtype=uint8


class IOChannel(str, Enum):
    """
    I/O channels for HD events and hints.

    Each channel represents a cognitive pathway:
    - SENSORIUM: Physical embodiment (7+1 senses)
    - NETWORK: LAN cortex (flows, nodes, policies)
    - UI: Visual cortex (panels, gaze, clicks)
    - TASK: Executive function (jobs, tools, code)
    - INTERNAL: Introspection (mindreader, teleology)
    """
    SENSORIUM = "sensorium"    # Physical sensors, telemetry
    NETWORK = "network"        # Flows, nodes, policies
    UI = "ui"                  # Panels, gaze, clicks, avatar
    TASK = "task"              # Job state, tools, code execution
    INTERNAL = "internal"      # MindReader, Teleology, self-model


@dataclass
class HDInputEvent:
    """
    An HD-encoded input event from any channel.

    These events are bundled into the moment HV that feeds the HTC.
    Each event carries:
    - channel: Which cognitive pathway it came from
    - role: The HD role binding (e.g., "ROLE_NET", "ROLE_UI_EVENT")
    - meta: Structured metadata for logging/debugging
    - hv: The actual role-bound hypervector
    - timestamp: When the event occurred
    """
    channel: IOChannel
    role: str                          # HD role name
    meta: Dict[str, Any]               # Structured metadata
    hv: HV                             # Role-bound hypervector
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Optional priority for filtering
    priority: float = 0.5              # 0=low, 1=high (for bundling weights)

    # Source identification
    source_id: Optional[str] = None    # node_id, panel_id, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel.value,
            "role": self.role,
            "meta": self.meta,
            "priority": self.priority,
            "source_id": self.source_id,
            "timestamp": self.timestamp.isoformat(),
            "hv_shape": self.hv.shape if self.hv is not None else None,
        }


@dataclass
class HDOutputHint:
    """
    An HD-derived output hint to actuators.

    These hints come from HTC resonance analysis and drive:
    - Avatar affect and animation
    - UI focus and highlighting
    - Network policies and throttling
    - Task prioritization

    Unlike events (which are HVs), hints are low-dimensional
    payloads suitable for direct consumption by actuators.
    """
    channel: IOChannel
    kind: str                          # "AFFECT", "FOCUS", "NET_POLICY", etc.
    meta: Dict[str, Any]               # Structured metadata
    payload: Dict[str, Any]            # Low-dim vectors, IDs, thresholds
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Confidence/certainty
    confidence: float = 1.0            # How sure the HTC is about this hint

    # Expiry
    ttl_seconds: float = 1.0           # How long this hint is valid

    def is_expired(self) -> bool:
        elapsed = (datetime.utcnow() - self.timestamp).total_seconds()
        return elapsed > self.ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel.value,
            "kind": self.kind,
            "meta": self.meta,
            "payload": self.payload,
            "confidence": self.confidence,
            "ttl_seconds": self.ttl_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Specific Hint Types
# =============================================================================

@dataclass
class AffectHint(HDOutputHint):
    """Affect state for avatar/UI theming."""

    def __init__(
        self,
        valence: float,
        arousal: float,
        certainty: float,
        focus: float,
        **kwargs
    ):
        super().__init__(
            channel=IOChannel.UI,
            kind="AFFECT",
            meta={},
            payload={
                "valence": valence,
                "arousal": arousal,
                "certainty": certainty,
                "focus": focus,
            },
            **kwargs
        )

    @property
    def valence(self) -> float:
        return self.payload["valence"]

    @property
    def arousal(self) -> float:
        return self.payload["arousal"]

    @property
    def certainty(self) -> float:
        return self.payload["certainty"]

    @property
    def focus(self) -> float:
        return self.payload["focus"]


@dataclass
class UIFocusHint(HDOutputHint):
    """UI focus/highlighting hint."""

    def __init__(
        self,
        highlight_nodes: List[str],
        highlight_flows: List[str],
        preferred_panel: str,
        **kwargs
    ):
        super().__init__(
            channel=IOChannel.UI,
            kind="UI_FOCUS",
            meta={},
            payload={
                "highlight_nodes": highlight_nodes,
                "highlight_flows": highlight_flows,
                "preferred_panel": preferred_panel,
            },
            **kwargs
        )


@dataclass
class NetPolicyHint(HDOutputHint):
    """Network policy hint for NodeAgents and reflexes."""

    def __init__(
        self,
        throttle_services: List[str],
        prioritize_services: List[str],
        suspicious_flows: List[str],
        never_throttle: List[str],
        min_bandwidth_mbps: float = 10.0,
        **kwargs
    ):
        super().__init__(
            channel=IOChannel.NETWORK,
            kind="NET_POLICY",
            meta={},
            payload={
                "throttle_services": throttle_services,
                "prioritize_services": prioritize_services,
                "suspicious_flows": suspicious_flows,
                "never_throttle": never_throttle,
                "min_bandwidth_mbps": min_bandwidth_mbps,
            },
            **kwargs
        )


@dataclass
class TaskPriorityHint(HDOutputHint):
    """Task prioritization hint."""

    def __init__(
        self,
        priority_tasks: List[str],
        defer_tasks: List[str],
        kill_tasks: List[str],
        **kwargs
    ):
        super().__init__(
            channel=IOChannel.TASK,
            kind="TASK_PRIORITY",
            meta={},
            payload={
                "priority_tasks": priority_tasks,
                "defer_tasks": defer_tasks,
                "kill_tasks": kill_tasks,
            },
            **kwargs
        )


# =============================================================================
# Event Batch
# =============================================================================

@dataclass
class EventBatch:
    """
    A batch of HD input events from a single tick.

    Used to collect events from all channels before bundling.
    """
    events: List[HDInputEvent] = field(default_factory=list)
    tick_number: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def add(self, event: HDInputEvent) -> None:
        self.events.append(event)

    def by_channel(self, channel: IOChannel) -> List[HDInputEvent]:
        return [e for e in self.events if e.channel == channel]

    def get_hvs(self, weights: Optional[Dict[IOChannel, float]] = None) -> List[tuple]:
        """Get (hv, weight) pairs for bundling."""
        result = []
        for e in self.events:
            w = 1.0
            if weights and e.channel in weights:
                w = weights[e.channel]
            w *= e.priority
            result.append((e.hv, w))
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tick_number": self.tick_number,
            "timestamp": self.timestamp.isoformat(),
            "event_count": len(self.events),
            "by_channel": {
                ch.value: len(self.by_channel(ch))
                for ch in IOChannel
            },
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'HV',
    'IOChannel',
    'HDInputEvent',
    'HDOutputHint',
    'AffectHint',
    'UIFocusHint',
    'NetPolicyHint',
    'TaskPriorityHint',
    'EventBatch',
]
