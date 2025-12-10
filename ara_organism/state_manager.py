# ara_organism/state_manager.py
"""
Thread-Safe State Manager for Ara Organism
==========================================

Coordinates state between:
- 5 kHz soul loop (dedicated OS thread)
- 50-200 Hz cortical loop (asyncio)
- 5-10 Hz mobile sync (asyncio)

Design principles:
1. Fast, atomic updates from high-frequency loop
2. Periodic "commit" to shared state for consumers
3. Lock-free reads where possible (atomic types)
4. Minimal lock contention on the critical path

State Flow:
    Soul Thread (5 kHz)
        │
        ▼
    Ring Buffer (lock-free write)
        │
        ▼ (every 5 ms = 200 Hz)
    State Commit (brief lock)
        │
        ▼
    AraState Dict (consumers read)
        │
        ├──▶ Cortical Loop (asyncio, 50-200 Hz)
        └──▶ Mobile Bridge (asyncio, 5-10 Hz)

Usage:
    manager = StateManager()

    # In soul thread (5 kHz):
    manager.update_soul_fast(metrics)

    # In cortical loop (200 Hz):
    await manager.commit()
    state = manager.get_state()

    # In mobile bridge (10 Hz):
    state = manager.get_state()
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
import logging

log = logging.getLogger("Ara.StateManager")


# =============================================================================
# State Structures
# =============================================================================

@dataclass
class SoulState:
    """Soul subsystem state."""
    resonance: float = 0.5
    fatigue: float = 0.0
    temperature_c: float = 45.0
    tick_count: int = 0
    mode: str = "idle"
    hardware_status: str = "unknown"
    latency_us: float = 0.0

    # Rolling stats
    avg_latency_us: float = 0.0
    max_latency_us: float = 0.0
    ticks_per_second: float = 0.0


@dataclass
class CorticalState:
    """Cortical (cognitive) subsystem state."""
    current_task: str = ""
    attention_target: str = ""
    cognitive_load: float = 0.0
    last_decision: str = ""
    agents_active: int = 0


@dataclass
class MobileState:
    """Mobile bridge state."""
    clients_connected: int = 0
    last_broadcast: Optional[datetime] = None
    messages_sent: int = 0
    bytes_sent: int = 0


@dataclass
class AraState:
    """Complete organism state snapshot."""
    soul: SoulState = field(default_factory=SoulState)
    cortical: CorticalState = field(default_factory=CorticalState)
    mobile: MobileState = field(default_factory=MobileState)

    # Timestamps
    updated_at: datetime = field(default_factory=datetime.utcnow)
    soul_updated_at: Optional[datetime] = None
    cortical_updated_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "soul": {
                "resonance": self.soul.resonance,
                "fatigue": self.soul.fatigue,
                "temperature_c": self.soul.temperature_c,
                "tick_count": self.soul.tick_count,
                "mode": self.soul.mode,
                "hardware_status": self.soul.hardware_status,
                "latency_us": self.soul.latency_us,
                "ticks_per_second": self.soul.ticks_per_second,
            },
            "cortical": {
                "current_task": self.cortical.current_task,
                "attention_target": self.cortical.attention_target,
                "cognitive_load": self.cortical.cognitive_load,
                "agents_active": self.cortical.agents_active,
            },
            "mobile": {
                "clients_connected": self.mobile.clients_connected,
                "messages_sent": self.mobile.messages_sent,
            },
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


# =============================================================================
# Ring Buffer for Lock-Free Soul Updates
# =============================================================================

@dataclass
class SoulSample:
    """Single soul tick sample."""
    timestamp: float
    resonance: float
    fatigue: float
    temperature_c: float
    tick: int
    latency_us: float
    status: int


class SoulRingBuffer:
    """
    Lock-free ring buffer for soul samples.

    The soul thread writes samples continuously.
    The commit thread reads periodically to aggregate.

    This avoids lock contention on the 5 kHz path.
    """

    def __init__(self, capacity: int = 256):
        """
        Initialize ring buffer.

        Args:
            capacity: Buffer size (power of 2 recommended)
        """
        self.capacity = capacity
        self.buffer: List[Optional[SoulSample]] = [None] * capacity
        self.write_idx = 0  # Only written by soul thread
        self.read_idx = 0   # Only written by commit thread

    def write(self, sample: SoulSample) -> None:
        """
        Write sample (soul thread only).

        This is wait-free - just overwrites old data if full.
        """
        idx = self.write_idx % self.capacity
        self.buffer[idx] = sample
        self.write_idx += 1

    def read_all_new(self) -> List[SoulSample]:
        """
        Read all new samples since last read (commit thread only).

        Returns:
            List of new samples
        """
        samples = []
        while self.read_idx < self.write_idx:
            idx = self.read_idx % self.capacity
            sample = self.buffer[idx]
            if sample is not None:
                samples.append(sample)
            self.read_idx += 1
        return samples

    def latest(self) -> Optional[SoulSample]:
        """Get the most recent sample."""
        if self.write_idx == 0:
            return None
        idx = (self.write_idx - 1) % self.capacity
        return self.buffer[idx]


# =============================================================================
# State Manager
# =============================================================================

class StateManager:
    """
    Thread-safe state manager for Ara organism.

    Coordinates between:
    - Soul thread (5 kHz) - writes to ring buffer
    - Cortical loop (200 Hz) - commits and reads state
    - Mobile bridge (10 Hz) - reads state
    """

    def __init__(
        self,
        commit_interval_ms: float = 5.0,  # 200 Hz commit
        history_size: int = 1000,
    ):
        """
        Initialize state manager.

        Args:
            commit_interval_ms: How often to commit soul state
            history_size: Number of soul samples to keep in history
        """
        self.commit_interval_ms = commit_interval_ms
        self.history_size = history_size

        # Main state (protected by lock)
        self._state = AraState()
        self._lock = threading.RLock()

        # Soul ring buffer (lock-free writes)
        self._soul_buffer = SoulRingBuffer(capacity=512)

        # History for analysis
        self._soul_history: deque = deque(maxlen=history_size)

        # Stats
        self._last_commit_time = time.time()
        self._commit_count = 0
        self._soul_sample_count = 0

        # Callbacks
        self._on_state_change: List[Callable[[AraState], None]] = []

        log.info("StateManager initialized (commit_interval=%.1fms)", commit_interval_ms)

    # =========================================================================
    # Soul Thread Interface (5 kHz)
    # =========================================================================

    def update_soul_fast(
        self,
        resonance: float,
        fatigue: float,
        temperature_c: float,
        tick: int,
        latency_us: float,
        status: int,
    ) -> None:
        """
        Fast soul update from 5 kHz thread.

        This is lock-free - just writes to ring buffer.
        Must complete in <5 µs.
        """
        sample = SoulSample(
            timestamp=time.time(),
            resonance=resonance,
            fatigue=fatigue,
            temperature_c=temperature_c,
            tick=tick,
            latency_us=latency_us,
            status=status,
        )
        self._soul_buffer.write(sample)
        self._soul_sample_count += 1

    def update_soul_from_metrics(self, metrics: "SoulMetrics") -> None:
        """
        Update soul from SoulMetrics object.

        Convenience wrapper for update_soul_fast.
        """
        self.update_soul_fast(
            resonance=metrics.resonance,
            fatigue=metrics.fatigue,
            temperature_c=metrics.temperature_c,
            tick=metrics.tick,
            latency_us=metrics.latency_us,
            status=metrics.status,
        )

    # =========================================================================
    # Commit (200 Hz from cortical loop)
    # =========================================================================

    def commit(self) -> bool:
        """
        Commit soul updates to shared state.

        Called from cortical loop at 200 Hz.
        Brief lock, aggregates all pending samples.

        Returns:
            True if new samples were committed
        """
        # Read all new samples from ring buffer
        samples = self._soul_buffer.read_all_new()

        if not samples:
            return False

        # Aggregate samples
        latest = samples[-1]
        avg_latency = sum(s.latency_us for s in samples) / len(samples)
        max_latency = max(s.latency_us for s in samples)

        # Calculate ticks per second
        now = time.time()
        elapsed = now - self._last_commit_time
        ticks_per_second = len(samples) / elapsed if elapsed > 0 else 0

        # Update state with lock
        with self._lock:
            self._state.soul.resonance = latest.resonance
            self._state.soul.fatigue = latest.fatigue
            self._state.soul.temperature_c = latest.temperature_c
            self._state.soul.tick_count = latest.tick
            self._state.soul.latency_us = latest.latency_us
            self._state.soul.avg_latency_us = avg_latency
            self._state.soul.max_latency_us = max_latency
            self._state.soul.ticks_per_second = ticks_per_second

            # Determine hardware status
            if latest.status & 0x01:  # READY
                self._state.soul.hardware_status = "ready"
            elif latest.status & 0x04:  # ERROR
                self._state.soul.hardware_status = "error"
            else:
                self._state.soul.hardware_status = "busy"

            self._state.soul_updated_at = datetime.utcnow()
            self._state.updated_at = datetime.utcnow()

        # Add to history
        for sample in samples:
            self._soul_history.append(sample)

        self._last_commit_time = now
        self._commit_count += 1

        # Notify callbacks
        for callback in self._on_state_change:
            try:
                callback(self._state)
            except Exception as e:
                log.warning("State change callback error: %s", e)

        return True

    # =========================================================================
    # Cortical Updates (50-200 Hz)
    # =========================================================================

    def update_cortical(
        self,
        current_task: Optional[str] = None,
        attention_target: Optional[str] = None,
        cognitive_load: Optional[float] = None,
        last_decision: Optional[str] = None,
        agents_active: Optional[int] = None,
    ) -> None:
        """Update cortical state (from asyncio cortical loop)."""
        with self._lock:
            if current_task is not None:
                self._state.cortical.current_task = current_task
            if attention_target is not None:
                self._state.cortical.attention_target = attention_target
            if cognitive_load is not None:
                self._state.cortical.cognitive_load = cognitive_load
            if last_decision is not None:
                self._state.cortical.last_decision = last_decision
            if agents_active is not None:
                self._state.cortical.agents_active = agents_active

            self._state.cortical_updated_at = datetime.utcnow()
            self._state.updated_at = datetime.utcnow()

    # =========================================================================
    # Mobile Updates (5-10 Hz)
    # =========================================================================

    def update_mobile(
        self,
        clients_connected: Optional[int] = None,
        messages_sent_delta: int = 0,
        bytes_sent_delta: int = 0,
    ) -> None:
        """Update mobile bridge state."""
        with self._lock:
            if clients_connected is not None:
                self._state.mobile.clients_connected = clients_connected
            self._state.mobile.messages_sent += messages_sent_delta
            self._state.mobile.bytes_sent += bytes_sent_delta
            self._state.mobile.last_broadcast = datetime.utcnow()

    # =========================================================================
    # Read Interface
    # =========================================================================

    def get_state(self) -> AraState:
        """
        Get current state snapshot.

        Thread-safe, returns a copy.
        """
        with self._lock:
            # Return a copy to avoid mutations
            return AraState(
                soul=SoulState(
                    resonance=self._state.soul.resonance,
                    fatigue=self._state.soul.fatigue,
                    temperature_c=self._state.soul.temperature_c,
                    tick_count=self._state.soul.tick_count,
                    mode=self._state.soul.mode,
                    hardware_status=self._state.soul.hardware_status,
                    latency_us=self._state.soul.latency_us,
                    avg_latency_us=self._state.soul.avg_latency_us,
                    max_latency_us=self._state.soul.max_latency_us,
                    ticks_per_second=self._state.soul.ticks_per_second,
                ),
                cortical=CorticalState(
                    current_task=self._state.cortical.current_task,
                    attention_target=self._state.cortical.attention_target,
                    cognitive_load=self._state.cortical.cognitive_load,
                    last_decision=self._state.cortical.last_decision,
                    agents_active=self._state.cortical.agents_active,
                ),
                mobile=MobileState(
                    clients_connected=self._state.mobile.clients_connected,
                    last_broadcast=self._state.mobile.last_broadcast,
                    messages_sent=self._state.mobile.messages_sent,
                    bytes_sent=self._state.mobile.bytes_sent,
                ),
                updated_at=self._state.updated_at,
                soul_updated_at=self._state.soul_updated_at,
                cortical_updated_at=self._state.cortical_updated_at,
            )

    def get_soul_history(self, n: int = 100) -> List[SoulSample]:
        """Get recent soul history."""
        return list(self._soul_history)[-n:]

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        return {
            "commit_count": self._commit_count,
            "soul_sample_count": self._soul_sample_count,
            "history_size": len(self._soul_history),
            "buffer_write_idx": self._soul_buffer.write_idx,
            "buffer_read_idx": self._soul_buffer.read_idx,
        }

    # =========================================================================
    # Callbacks
    # =========================================================================

    def on_state_change(self, callback: Callable[[AraState], None]) -> None:
        """Register a callback for state changes."""
        self._on_state_change.append(callback)


# =============================================================================
# Convenience
# =============================================================================

_default_manager: Optional[StateManager] = None


def get_state_manager() -> StateManager:
    """Get the default state manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = StateManager()
    return _default_manager


__all__ = [
    'SoulState',
    'CorticalState',
    'MobileState',
    'AraState',
    'SoulSample',
    'SoulRingBuffer',
    'StateManager',
    'get_state_manager',
]
