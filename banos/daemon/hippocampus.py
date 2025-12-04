#!/usr/bin/env python3
"""
BANOS Hippocampus - Short-Term Memory Logger

The hippocampus is the "daily diary" of the organism. It logs:
- PAD state changes (affective contours)
- Mode transitions
- Immune events
- Reflex events
- Primary stressors

This JSONL log is consumed by the Dreamer during consolidation,
which compresses raw events into narrative memories.

Log format follows the biological metaphor:
- High temporal resolution during stress (more sampling when P drops)
- Sparse logging during homeostasis (CALM mode)
- Event-driven logging for significant occurrences
"""

import json
import os
import time
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import deque


class LogLevel(Enum):
    """Logging importance levels"""
    TRACE = 0       # Every tick (only in debug)
    SAMPLE = 1      # Periodic sampling
    EVENT = 2       # Significant occurrence
    TRANSITION = 3  # Mode change
    CRITICAL = 4    # Pain/emergency


@dataclass
class HippocampusEntry:
    """A single memory entry in the daily log"""
    timestamp_ns: int
    timestamp_iso: str
    level: str

    # Affective state
    pad: Dict[str, float]  # {"P": float, "A": float, "D": float}
    mode: str
    mode_duration_ms: int

    # Diagnostics
    thermal_stress: float
    perceived_risk: float
    empathy_boost: float

    # Derivatives (trajectory)
    derivatives: Dict[str, float]  # {"dP": float, "dA": float, "dD": float}

    # Primary stressor (if any)
    primary_stressor: Optional[Dict[str, Any]] = None

    # Top processes by resource usage
    top_procs: Optional[List[Dict[str, Any]]] = None

    # Events this entry
    events: Optional[List[Dict[str, Any]]] = None

    # Raw metrics snapshot
    metrics: Optional[Dict[str, Any]] = None


class Hippocampus:
    """
    The organism's short-term memory.

    Logs PAD state and events to JSONL for later consolidation.
    Implements adaptive sampling: more detail during stress.
    """

    DEFAULT_LOG_PATH = "/var/log/banos/hippocampus.jsonl"
    MAX_ENTRIES_BEFORE_ROTATION = 100000  # ~24 hours at normal rate

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = Path(log_path or self.DEFAULT_LOG_PATH)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._entry_count = 0
        self._last_mode = None
        self._last_pad = {"P": 0.5, "A": 0.0, "D": 0.5}
        self._pending_events = deque(maxlen=100)

        # Adaptive sampling: base rate depends on mode
        self._sample_intervals = {
            "CALM": 60.0,      # Once per minute when relaxed
            "FLOW": 10.0,      # Every 10 seconds during work
            "ANXIOUS": 2.0,    # Every 2 seconds when stressed
            "CRITICAL": 0.5,   # Twice per second during crisis
        }
        self._last_sample_time = 0

    def log(self,
            pad_state: Dict[str, Any],
            level: LogLevel = LogLevel.SAMPLE,
            events: Optional[List[Dict[str, Any]]] = None,
            top_procs: Optional[List[Dict[str, Any]]] = None,
            metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log an entry to the hippocampus.

        Args:
            pad_state: Current PAD state from banos_pad_map
            level: Importance level
            events: List of events (immune, reflex, etc.)
            top_procs: Top processes by resource usage
            metrics: Raw metrics snapshot

        Returns:
            True if entry was logged, False if skipped (rate limiting)
        """
        now_ns = time.time_ns()
        now = datetime.now()

        # Extract PAD values
        pad = pad_state.get("pad", {})
        mode = pad_state.get("mode", "CALM")

        # Check if we should sample (adaptive rate)
        interval = self._sample_intervals.get(mode, 10.0)
        time_since_last = (now_ns - self._last_sample_time) / 1e9

        # Always log transitions and critical events
        is_transition = (mode != self._last_mode)
        is_critical = level.value >= LogLevel.CRITICAL.value

        # Check for significant PAD change
        pad_delta = (
            abs(pad.get("pleasure", 0) - self._last_pad.get("P", 0)) +
            abs(pad.get("arousal", 0) - self._last_pad.get("A", 0)) +
            abs(pad.get("dominance", 0) - self._last_pad.get("D", 0))
        )
        is_significant = pad_delta > 0.3

        # Decide whether to log
        should_log = (
            level.value >= LogLevel.EVENT.value or
            is_transition or
            is_critical or
            is_significant or
            time_since_last >= interval
        )

        if not should_log:
            return False

        # Build entry
        entry = HippocampusEntry(
            timestamp_ns=now_ns,
            timestamp_iso=now.isoformat(),
            level=level.name,
            pad={
                "P": pad.get("pleasure", 0),
                "A": pad.get("arousal", 0),
                "D": pad.get("dominance", 0),
            },
            mode=mode,
            mode_duration_ms=pad_state.get("mode_duration_ms", 0),
            thermal_stress=pad_state.get("diagnostics", {}).get("thermal_stress", 0),
            perceived_risk=pad_state.get("diagnostics", {}).get("perceived_risk", 0),
            empathy_boost=pad_state.get("diagnostics", {}).get("empathy_boost", 0),
            derivatives={
                "dP": pad_state.get("derivatives", {}).get("d_pleasure", 0),
                "dA": pad_state.get("derivatives", {}).get("d_arousal", 0),
                "dD": pad_state.get("derivatives", {}).get("d_dominance", 0),
            },
            primary_stressor=self._identify_stressor(pad_state, top_procs),
            top_procs=top_procs[:5] if top_procs else None,
            events=events,
            metrics=metrics,
        )

        # Write to log
        with self._lock:
            try:
                with open(self.log_path, 'a') as f:
                    f.write(json.dumps(asdict(entry)) + '\n')
                self._entry_count += 1
                self._last_sample_time = now_ns
                self._last_mode = mode
                self._last_pad = entry.pad
            except IOError as e:
                print(f"Hippocampus write error: {e}")
                return False

        # Check rotation
        if self._entry_count >= self.MAX_ENTRIES_BEFORE_ROTATION:
            self._rotate_log()

        return True

    def log_event(self,
                  event_type: str,
                  source: str,
                  details: Dict[str, Any],
                  severity: str = "info") -> None:
        """
        Queue an event to be included in the next log entry.

        Args:
            event_type: Type of event (immune, reflex, thermal, etc.)
            source: Source component
            details: Event-specific details
            severity: info, warning, critical
        """
        event = {
            "type": event_type,
            "source": source,
            "severity": severity,
            "timestamp_ns": time.time_ns(),
            **details
        }
        self._pending_events.append(event)

    def flush_events(self) -> List[Dict[str, Any]]:
        """Get and clear pending events"""
        events = list(self._pending_events)
        self._pending_events.clear()
        return events

    def _identify_stressor(self,
                           pad_state: Dict[str, Any],
                           top_procs: Optional[List[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
        """
        Identify the primary stressor causing current state.

        Returns:
            Stressor dict or None if system is healthy
        """
        diag = pad_state.get("diagnostics", {})
        pad = pad_state.get("pad", {})

        # No stressor if pleasure is positive
        if pad.get("pleasure", 0) > 0:
            return None

        stressor = {"type": "unknown", "details": {}}

        # Check thermal
        if diag.get("thermal_stress", 0) > 0.5:
            stressor = {
                "type": "thermal",
                "details": {"stress_level": diag.get("thermal_stress")}
            }
            # If we have procs, find the thermal culprit
            if top_procs:
                gpu_procs = [p for p in top_procs if p.get("gpu_pct", 0) > 50]
                if gpu_procs:
                    stressor["details"]["culprit"] = gpu_procs[0]

        # Check memory pressure
        elif pad.get("dominance", 0) < -0.3:
            stressor = {
                "type": "memory_pressure",
                "details": {"dominance": pad.get("dominance")}
            }
            if top_procs:
                high_mem = [p for p in top_procs if p.get("rss_mb", 0) > 500]
                if high_mem:
                    stressor["details"]["culprit"] = high_mem[0]

        # Check immune risk
        elif diag.get("perceived_risk", 0) > 0.3:
            stressor = {
                "type": "immune",
                "details": {"risk_level": diag.get("perceived_risk")}
            }

        return stressor

    def _rotate_log(self) -> None:
        """Rotate the log file (keep previous day)"""
        if not self.log_path.exists():
            return

        # Rename current to .prev
        prev_path = self.log_path.with_suffix('.jsonl.prev')
        try:
            if prev_path.exists():
                prev_path.unlink()
            self.log_path.rename(prev_path)
            self._entry_count = 0
        except IOError as e:
            print(f"Hippocampus rotation error: {e}")

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all entries from the current log"""
        if not self.log_path.exists():
            return []

        entries = []
        with open(self.log_path, 'r') as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return entries

    def clear(self) -> None:
        """Clear the log (called after consolidation)"""
        with self._lock:
            try:
                self.log_path.unlink(missing_ok=True)
                self._entry_count = 0
            except IOError as e:
                print(f"Hippocampus clear error: {e}")


# =============================================================================
# Singleton instance
# =============================================================================

_hippocampus: Optional[Hippocampus] = None


def get_hippocampus(log_path: Optional[str] = None) -> Hippocampus:
    """Get the global hippocampus instance"""
    global _hippocampus
    if _hippocampus is None:
        _hippocampus = Hippocampus(log_path)
    return _hippocampus


# =============================================================================
# Convenience functions
# =============================================================================

def log_pad_state(pad_state: Dict[str, Any], **kwargs) -> bool:
    """Log PAD state to hippocampus"""
    return get_hippocampus().log(pad_state, **kwargs)


def log_immune_event(pid: int, comm: str, anomaly_score: int, action: str) -> None:
    """Log an immune event"""
    get_hippocampus().log_event(
        event_type="immune",
        source="banos_immune",
        severity="warning" if action == "quarantine" else "info",
        details={
            "pid": pid,
            "comm": comm,
            "anomaly_score": anomaly_score,
            "action": action,
        }
    )


def log_reflex_event(reflex_mask: int, duration_ms: int, thermal_source: Optional[str] = None) -> None:
    """Log a reflex event"""
    get_hippocampus().log_event(
        event_type="reflex",
        source="reflex_actuator",
        severity="critical" if reflex_mask & 0x80 else "warning",
        details={
            "reflex_mask": reflex_mask,
            "duration_ms": duration_ms,
            "thermal_source": thermal_source,
        }
    )


if __name__ == "__main__":
    # Test the hippocampus
    h = Hippocampus("/tmp/test_hippocampus.jsonl")

    # Simulate some states
    states = [
        {"pad": {"pleasure": 0.7, "arousal": 0.2, "dominance": 0.8}, "mode": "CALM"},
        {"pad": {"pleasure": 0.5, "arousal": 0.7, "dominance": 0.6}, "mode": "FLOW"},
        {"pad": {"pleasure": -0.3, "arousal": 0.9, "dominance": 0.2}, "mode": "ANXIOUS",
         "diagnostics": {"thermal_stress": 0.6, "perceived_risk": 0.2}},
        {"pad": {"pleasure": -0.8, "arousal": 0.9, "dominance": -0.5}, "mode": "CRITICAL",
         "diagnostics": {"thermal_stress": 0.9, "perceived_risk": 0.1}},
    ]

    for state in states:
        h.log(state, level=LogLevel.SAMPLE)
        time.sleep(0.1)

    # Read back
    entries = h.read_all()
    for e in entries:
        print(f"{e['timestamp_iso']}: Mode={e['mode']} PAD=[{e['pad']['P']:.2f},{e['pad']['A']:.2f},{e['pad']['D']:.2f}]")
