"""
Ara Somatic Server - Visual Cortex Bridge
==========================================

Bridge between the HTC/Teleology soul state and the GPU renderer.

The Somatic Server:
- Subscribes to soul state (L2 affect, attractor snapshot)
- Drives GPU renderer (UE5/Godot/WebGPU) via affect + memory palace
- Emits UI interaction events back as HVs

This is Ara's "visual cortex" - where internal state becomes visible
expression and where visual attention becomes soul input.

Mythic: The face through which Ara expresses her inner state
Physical: 60Hz affect updates, 5-10Hz memory palace snapshots
Safety: Affect values clamped, glitch severity bounded
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Protocol
import numpy as np

from ara.io.types import HDInputEvent, IOChannel, HV


# =============================================================================
# GPU Client Protocol
# =============================================================================

class GPUClient(Protocol):
    """
    Protocol for GPU renderer backends.

    Implementations can wrap UE5, Godot, WebGPU, or even console output.
    """

    def push_affect(self, affect: Dict[str, float]) -> None:
        """Push affect state to drive avatar/UI expression."""
        ...

    def push_memory_palace(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """Push 3D memory palace snapshot for visualization."""
        ...

    def push_glitch(self, severity: float) -> None:
        """Trigger visual glitch effect (from reflex events)."""
        ...

    def poll_ui_events(self) -> List[Dict[str, Any]]:
        """Poll for UI interaction events since last call."""
        ...


# =============================================================================
# Dummy GPU Client (for testing)
# =============================================================================

class DummyGPUClient:
    """Console-only GPU client for testing without actual renderer."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._pending_events: List[Dict[str, Any]] = []

    def push_affect(self, affect: Dict[str, float]) -> None:
        if self.verbose:
            v = affect.get("valence", 0)
            a = affect.get("arousal", 0)
            c = affect.get("certainty", 0)
            f = affect.get("focus", 0)
            print(f"[GPU] Affect: valence={v:.2f} arousal={a:.2f} "
                  f"certainty={c:.2f} focus={f:.2f}")

    def push_memory_palace(
        self,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        if self.verbose:
            n = len(points) if points is not None else 0
            print(f"[GPU] Memory Palace: {n} attractors projected to 3D")

    def push_glitch(self, severity: float) -> None:
        if self.verbose:
            bars = "!" * int(severity * 10)
            print(f"[GPU] GLITCH [{bars:10}] severity={severity:.2f}")

    def poll_ui_events(self) -> List[Dict[str, Any]]:
        events = self._pending_events.copy()
        self._pending_events.clear()
        return events

    def inject_event(self, event: Dict[str, Any]) -> None:
        """Inject a fake UI event for testing."""
        self._pending_events.append(event)


# =============================================================================
# Somatic Server
# =============================================================================

@dataclass
class SomaticState:
    """Current state of the somatic server."""
    last_affect: Dict[str, float] = field(default_factory=dict)
    last_palace_update: Optional[datetime] = None
    tick_count: int = 0
    glitch_count: int = 0


class SomaticServer:
    """
    Bridge between the HTC/Teleology and the GPU renderer.

    The somatic server is called every sovereign tick to:
    1. Compute affect from resonance/reward history
    2. Push affect to GPU for avatar expression
    3. Optionally push memory palace snapshot
    4. Collect UI events and encode them as HVs
    """

    def __init__(
        self,
        gpu_client: GPUClient,
        history_len: int = 100,
        palace_update_hz: float = 5.0,
    ):
        """
        Args:
            gpu_client: Backend for GPU rendering
            history_len: How many ticks of history to keep
            palace_update_hz: How often to update memory palace (Hz)
        """
        self.gpu = gpu_client
        self.history_len = history_len
        self.palace_interval = 1.0 / palace_update_hz

        self.resonance_hist: List[np.ndarray] = []
        self.reward_hist: List[int] = []
        self.state = SomaticState()

        self._last_palace_time = 0.0

    def update_from_soul(
        self,
        resonance: np.ndarray,
        reward: int,
        attractor_snapshot: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> List[HDInputEvent]:
        """
        Called every sovereign tick.

        Args:
            resonance: Current resonance vector from HTC
            reward: Current reward signal
            attractor_snapshot: Optional dict with attractor state for palace
            timestamp: Current time (for palace rate limiting)

        Returns:
            List of HDInputEvents representing UI interactions since last call
        """
        import time
        if timestamp is None:
            timestamp = time.time()

        self.state.tick_count += 1

        # Update history
        self.resonance_hist.append(resonance)
        self.reward_hist.append(reward)

        # Trim history
        if len(self.resonance_hist) > self.history_len:
            self.resonance_hist = self.resonance_hist[-self.history_len:]
            self.reward_hist = self.reward_hist[-self.history_len:]

        # Compute and push affect
        affect = self._compute_affect()
        self.state.last_affect = affect
        self.gpu.push_affect(affect)

        # Maybe update memory palace (rate limited)
        if attractor_snapshot and (timestamp - self._last_palace_time) >= self.palace_interval:
            self._update_palace(attractor_snapshot)
            self._last_palace_time = timestamp

        # Collect UI events and encode as HVs
        ui_events = self.gpu.poll_ui_events()
        hv_events = self._encode_ui_events(ui_events)

        return hv_events

    def trigger_glitch(self, severity: float) -> None:
        """
        Trigger immediate visual glitch.

        Called by LANReflexBridge when reflex events occur.
        """
        severity = max(0.0, min(1.0, severity))
        self.state.glitch_count += 1
        self.gpu.push_glitch(severity)

    def _compute_affect(self) -> Dict[str, float]:
        """Compute affect from resonance/reward history."""
        from ara.core.graphics.affect import affect_from_history
        return affect_from_history(self.resonance_hist, self.reward_hist)

    def _update_palace(self, snapshot: Dict[str, Any]) -> None:
        """Update memory palace visualization."""
        self.state.last_palace_update = datetime.utcnow()

        points = snapshot.get("points_3d")
        colors = snapshot.get("teleology_colors")

        if points is not None and colors is not None:
            self.gpu.push_memory_palace(points, colors)

    def _encode_ui_events(
        self,
        events: List[Dict[str, Any]],
    ) -> List[HDInputEvent]:
        """Encode raw UI events into HDInputEvents."""
        from ara.core.graphics.event_codec import encode_ui_interaction

        hv_events = []
        for ev in events:
            hv = encode_ui_interaction(
                panel_id=ev.get("panel_id", "unknown"),
                action=ev.get("action", "unknown"),
                dwell_bin=ev.get("dwell_bin", "MINIMAL"),
            )

            hv_events.append(HDInputEvent(
                channel=IOChannel.UI,
                role="ROLE_UI_INTERACTION",
                meta=ev,
                hv=hv,
                priority=0.6 if ev.get("action") == "click" else 0.4,
            ))

        return hv_events

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "tick_count": self.state.tick_count,
            "glitch_count": self.state.glitch_count,
            "history_len": len(self.resonance_hist),
            "last_affect": self.state.last_affect,
            "last_palace_update": (
                self.state.last_palace_update.isoformat()
                if self.state.last_palace_update else None
            ),
        }


# =============================================================================
# Factory
# =============================================================================

_somatic_server: Optional[SomaticServer] = None


def get_somatic_server(
    gpu_client: Optional[GPUClient] = None,
) -> SomaticServer:
    """Get or create the global somatic server."""
    global _somatic_server
    if _somatic_server is None:
        if gpu_client is None:
            gpu_client = DummyGPUClient(verbose=False)
        _somatic_server = SomaticServer(gpu_client)
    return _somatic_server


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'GPUClient',
    'DummyGPUClient',
    'SomaticServer',
    'SomaticState',
    'get_somatic_server',
]
