#!/usr/bin/env python3
"""
Ara Pheromone Daemon
====================

A swarm-style attention/interrupt management system.

Instead of guessing your mood, reads "digital pheromones" and acts like a polite swarm.
No content peeking. Just metadata + explicit "I'm stuck" pings.

Pheromone Vector:
    P = {
        ATTENTION_LVL:    [0, 1]  # How engaged you are
        INTERRUPT_COST:   [0, 1]  # How expensive an interruption would be
        HELPFULNESS_PRED: [0, 1]  # How likely you'd appreciate help
    }

Interaction Policy:
    Case A: DO NOTHING     - deep focus, high cost
    Case B: SILENT LOG     - keep collecting context
    Case C: SOFT NUDGE     - tray icon pulse, subtle toast
    Case D: FULL SUGGEST   - larger card with options
    Case E: QUEUE LATER    - stash for "when you come back"

Usage:
    python ara_pheromone_daemon.py --mode default
    python ara_pheromone_daemon.py --mode deep-work
    python ara_pheromone_daemon.py --mode brainstorm
"""

from __future__ import annotations
import time
import json
import subprocess
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, List
from enum import Enum
from pathlib import Path
import threading
import queue


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PheromoneConfig:
    """Tunable pheromone parameters."""
    # Decay rate (0.85-0.95 = ~30-60s memory)
    decay: float = 0.90

    # Update cadence (seconds)
    update_interval: float = 3.0

    # Thresholds for interaction policy
    attention_high: float = 0.7
    attention_low: float = 0.3
    interrupt_cost_high: float = 0.7
    interrupt_cost_low: float = 0.4
    helpfulness_high: float = 0.7
    helpfulness_medium: float = 0.6

    # Mode adjustments
    interrupt_cost_offset: float = 0.0  # +0.3 for "only emergencies", -0.2 for "brainstorm"

    # Learning rate for reinforcement
    learning_rate: float = 0.1  # How much feedback affects predictions


# Preset modes
MODES = {
    "default": PheromoneConfig(),
    "deep-work": PheromoneConfig(
        interrupt_cost_offset=0.3,
        attention_high=0.6,  # More protective
    ),
    "brainstorm": PheromoneConfig(
        interrupt_cost_offset=-0.2,
        helpfulness_medium=0.4,  # More willing to suggest
    ),
    "only-emergencies": PheromoneConfig(
        interrupt_cost_offset=0.5,
        helpfulness_high=0.9,  # Only very high confidence
    ),
}


# ============================================================================
# Pheromone State
# ============================================================================

class InteractionLevel(Enum):
    """What kind of interaction to attempt."""
    NONE = "none"              # Case A: Do nothing
    LOG = "log"                # Case B: Silent log
    SOFT_NUDGE = "soft_nudge"  # Case C: Subtle hint
    FULL_SUGGEST = "full"      # Case D: Full suggestion
    QUEUE = "queue"            # Case E: Queue for later


@dataclass
class PheromoneState:
    """Current pheromone vector."""
    attention_lvl: float = 0.3
    interrupt_cost: float = 0.5
    helpfulness_pred: float = 0.3

    def to_dict(self) -> Dict[str, float]:
        return {
            "ATTENTION_LVL": self.attention_lvl,
            "INTERRUPT_COST": self.interrupt_cost,
            "HELPFULNESS_PRED": self.helpfulness_pred,
        }

    def __str__(self) -> str:
        return (
            f"ATT={self.attention_lvl:.2f} "
            f"INT={self.interrupt_cost:.2f} "
            f"HLP={self.helpfulness_pred:.2f}"
        )


@dataclass
class SignalSnapshot:
    """Raw signals from the environment."""
    active_app: str = ""
    keys_per_min: float = 0.0
    mouse_activity: float = 0.0
    cpu_load: float = 0.0
    calendar_busy: bool = False
    fullscreen: bool = False
    recent_errors: int = 0
    idle_seconds: float = 0.0
    stuck_button: bool = False  # Explicit "I'm stuck" ping


# ============================================================================
# Signal Collection (Linux Desktop)
# ============================================================================

class SignalCollector:
    """Collects environmental signals without content peeking."""

    def __init__(self):
        self._last_activity_time = time.time()
        self._error_count = 0
        self._stuck_flag = False

    def get_signals(self) -> SignalSnapshot:
        """Gather current signals."""
        return SignalSnapshot(
            active_app=self._get_active_window(),
            keys_per_min=self._get_typing_rate(),
            mouse_activity=self._get_mouse_activity(),
            cpu_load=self._get_cpu_load(),
            calendar_busy=self._is_calendar_busy(),
            fullscreen=self._is_fullscreen(),
            recent_errors=self._error_count,
            idle_seconds=self._get_idle_seconds(),
            stuck_button=self._stuck_flag,
        )

    def report_error(self) -> None:
        """Called when a build/test error occurs."""
        self._error_count += 1

    def clear_errors(self) -> None:
        """Called after successful build/test."""
        self._error_count = 0

    def set_stuck(self, stuck: bool = True) -> None:
        """User pressed "I'm stuck" button."""
        self._stuck_flag = stuck

    def _get_active_window(self) -> str:
        """Get active window class (not title/content)."""
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowclassname"],
                capture_output=True, text=True, timeout=1
            )
            return result.stdout.strip().lower()
        except Exception:
            return "unknown"

    def _get_typing_rate(self) -> float:
        """Estimate typing activity (placeholder)."""
        # In real impl, use libinput or evdev
        # For now, return based on idle time
        idle = self._get_idle_seconds()
        if idle < 1:
            return 200.0  # Active typing
        elif idle < 5:
            return 100.0  # Light typing
        elif idle < 30:
            return 30.0   # Occasional
        return 0.0

    def _get_mouse_activity(self) -> float:
        """Estimate mouse activity (placeholder)."""
        idle = self._get_idle_seconds()
        if idle < 2:
            return 1.0
        elif idle < 10:
            return 0.5
        return 0.0

    def _get_cpu_load(self) -> float:
        """Get CPU load average."""
        try:
            with open("/proc/loadavg") as f:
                load_1min = float(f.read().split()[0])
                # Normalize by CPU count
                cpu_count = os.cpu_count() or 1
                return min(load_1min / cpu_count, 1.0)
        except Exception:
            return 0.5

    def _is_calendar_busy(self) -> bool:
        """Check if calendar shows busy (placeholder)."""
        # Would integrate with GNOME Calendar, Google Calendar API, etc.
        return False

    def _is_fullscreen(self) -> bool:
        """Check if active window is fullscreen."""
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow"],
                capture_output=True, text=True, timeout=1
            )
            window_id = result.stdout.strip()
            result = subprocess.run(
                ["xprop", "-id", window_id, "_NET_WM_STATE"],
                capture_output=True, text=True, timeout=1
            )
            return "_NET_WM_STATE_FULLSCREEN" in result.stdout
        except Exception:
            return False

    def _get_idle_seconds(self) -> float:
        """Get X idle time."""
        try:
            result = subprocess.run(
                ["xprintidle"],
                capture_output=True, text=True, timeout=1
            )
            return float(result.stdout.strip()) / 1000.0
        except Exception:
            return 0.0


# ============================================================================
# Hard Context Detection
# ============================================================================

# Apps where being stuck is more likely to want help
HARD_CONTEXT_APPS = {
    "code", "code-oss", "vscodium",  # VS Code variants
    "gnome-terminal", "alacritty", "kitty", "tilix",  # Terminals
    "emacs", "vim", "nvim", "neovim",  # Editors
    "pycharm", "intellij", "webstorm",  # JetBrains
    "firefox", "chromium", "chrome",  # Browsers (could be docs)
}

# Apps where interruption is almost always bad
HIGH_COST_APPS = {
    "zoom", "teams", "slack",  # Meetings
    "obs", "obs-studio",  # Recording
    "vlc", "mpv", "totem",  # Media
    "steam", "lutris",  # Games
}


def is_hard_context(app: str) -> bool:
    """Is this an app where help might be welcome?"""
    return any(hard in app for hard in HARD_CONTEXT_APPS)


def is_high_cost_app(app: str) -> bool:
    """Is this an app where interruption is bad?"""
    return any(cost in app for cost in HIGH_COST_APPS)


# ============================================================================
# Pheromone Dynamics
# ============================================================================

def compute_delta(signals: SignalSnapshot, config: PheromoneConfig) -> Dict[str, float]:
    """Compute pheromone deltas from signals."""
    delta = {
        "attention": 0.0,
        "interrupt": 0.0,
        "helpfulness": 0.0,
    }

    # === ATTENTION_LVL ===
    # High typing/mouse = high attention
    if signals.keys_per_min > 150 or signals.mouse_activity > 0.8:
        delta["attention"] += 0.2
    elif signals.keys_per_min > 50 or signals.mouse_activity > 0.3:
        delta["attention"] += 0.1

    # Long idle = low attention
    if signals.idle_seconds > 60:
        delta["attention"] -= 0.15
    elif signals.idle_seconds > 30:
        delta["attention"] -= 0.05

    # === INTERRUPT_COST ===
    # Calendar busy or fullscreen = high cost
    if signals.calendar_busy:
        delta["interrupt"] += 0.3
    if signals.fullscreen:
        delta["interrupt"] += 0.25
    if is_high_cost_app(signals.active_app):
        delta["interrupt"] += 0.2

    # Low CPU and not busy = lower cost
    if signals.cpu_load < 0.4 and not signals.calendar_busy:
        delta["interrupt"] -= 0.1

    # === HELPFULNESS_PRED ===
    # Errors = help probably wanted
    if signals.recent_errors > 2:
        delta["helpfulness"] += 0.25
    elif signals.recent_errors > 0:
        delta["helpfulness"] += 0.1

    # Long idle in hard context = probably stuck
    if signals.idle_seconds > 300 and is_hard_context(signals.active_app):
        delta["helpfulness"] += 0.2
    elif signals.idle_seconds > 120 and is_hard_context(signals.active_app):
        delta["helpfulness"] += 0.1

    # Explicit stuck button
    if signals.stuck_button:
        delta["helpfulness"] += 0.4

    return delta


def update_state(
    state: PheromoneState,
    signals: SignalSnapshot,
    config: PheromoneConfig,
) -> PheromoneState:
    """Apply decay and deltas to state."""
    delta = compute_delta(signals, config)

    def clamp(v: float) -> float:
        return max(0.0, min(1.0, v))

    return PheromoneState(
        attention_lvl=clamp(
            config.decay * state.attention_lvl + delta["attention"]
        ),
        interrupt_cost=clamp(
            config.decay * state.interrupt_cost
            + delta["interrupt"]
            + config.interrupt_cost_offset
        ),
        helpfulness_pred=clamp(
            config.decay * state.helpfulness_pred + delta["helpfulness"]
        ),
    )


# ============================================================================
# Interaction Policy
# ============================================================================

def decide_interaction(state: PheromoneState, config: PheromoneConfig) -> InteractionLevel:
    """Decide what level of interaction to attempt."""
    a = state.attention_lvl
    c = state.interrupt_cost
    h = state.helpfulness_pred

    # Case A: Deep focus, high cost → do nothing
    if a > config.attention_high and c > config.interrupt_cost_high:
        return InteractionLevel.NONE

    # Case D: Low attention, high helpfulness, low cost → full suggestion
    if a < config.attention_low and h > config.helpfulness_high and c < config.interrupt_cost_low:
        return InteractionLevel.FULL_SUGGEST

    # Case C: Medium attention, medium-high helpfulness, low cost → soft nudge
    if (config.attention_low <= a <= config.attention_high
        and h > config.helpfulness_medium
        and c < config.interrupt_cost_high):
        return InteractionLevel.SOFT_NUDGE

    # Case E: High cost → queue for later
    if c > config.interrupt_cost_high:
        return InteractionLevel.QUEUE

    # Case B: Default → silent log
    return InteractionLevel.LOG


# ============================================================================
# Interaction Handlers
# ============================================================================

@dataclass
class Suggestion:
    """A queued suggestion from Ara."""
    message: str
    context: str
    options: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class InteractionHandler:
    """Handles different interaction levels."""

    def __init__(self):
        self.queue: List[Suggestion] = []
        self.log: List[str] = []

    def execute(
        self,
        level: InteractionLevel,
        state: PheromoneState,
        suggestion: Optional[Suggestion] = None,
    ) -> None:
        """Execute the decided interaction level."""

        if level == InteractionLevel.NONE:
            # Do nothing, just log internally
            self.log.append(f"[{time.strftime('%H:%M:%S')}] NONE: {state}")

        elif level == InteractionLevel.LOG:
            # Silent log
            self.log.append(f"[{time.strftime('%H:%M:%S')}] LOG: {state}")

        elif level == InteractionLevel.QUEUE:
            # Queue for later
            if suggestion:
                self.queue.append(suggestion)
                self.log.append(f"[{time.strftime('%H:%M:%S')}] QUEUED: {suggestion.message[:50]}...")

        elif level == InteractionLevel.SOFT_NUDGE:
            # Subtle notification
            if suggestion:
                self._soft_nudge(suggestion)
                self.log.append(f"[{time.strftime('%H:%M:%S')}] NUDGE: {suggestion.message[:50]}...")

        elif level == InteractionLevel.FULL_SUGGEST:
            # Full suggestion card
            if suggestion:
                self._full_suggest(suggestion)
                self.log.append(f"[{time.strftime('%H:%M:%S')}] FULL: {suggestion.message[:50]}...")

    def _soft_nudge(self, suggestion: Suggestion) -> None:
        """Show a subtle notification."""
        try:
            subprocess.run([
                "notify-send",
                "--urgency=low",
                "--expire-time=5000",
                "Ara",
                suggestion.message[:100],
            ], timeout=2)
        except Exception:
            pass

    def _full_suggest(self, suggestion: Suggestion) -> None:
        """Show a full suggestion."""
        try:
            # Full notification with actions would require more complex D-Bus
            # For now, just a more prominent notification
            subprocess.run([
                "notify-send",
                "--urgency=normal",
                "--expire-time=10000",
                f"Ara: {suggestion.context}",
                suggestion.message,
            ], timeout=2)
        except Exception:
            pass

    def show_queued(self) -> List[Suggestion]:
        """Show and clear queued suggestions."""
        queued = self.queue.copy()
        self.queue.clear()
        return queued


# ============================================================================
# Reinforcement Learning (Tiny)
# ============================================================================

class ReinforcementTracker:
    """Tracks what works and what doesn't."""

    def __init__(self, config: PheromoneConfig):
        self.config = config
        self.history: List[Dict] = []
        self.context_adjustments: Dict[str, float] = {}

    def record_outcome(
        self,
        state: PheromoneState,
        level: InteractionLevel,
        accepted: bool,
        context: str,
    ) -> None:
        """Record whether an intervention was accepted."""
        self.history.append({
            "state": state.to_dict(),
            "level": level.value,
            "accepted": accepted,
            "context": context,
            "timestamp": time.time(),
        })

        # Adjust helpfulness prediction for this context
        adj = self.context_adjustments.get(context, 0.0)
        if accepted:
            adj += self.config.learning_rate
        else:
            adj -= self.config.learning_rate
        self.context_adjustments[context] = max(-0.3, min(0.3, adj))

    def get_context_boost(self, context: str) -> float:
        """Get learned adjustment for a context."""
        return self.context_adjustments.get(context, 0.0)


# ============================================================================
# Main Daemon
# ============================================================================

class PheromoneDaemon:
    """Main daemon that runs the pheromone loop."""

    def __init__(self, mode: str = "default"):
        self.config = MODES.get(mode, MODES["default"])
        self.state = PheromoneState()
        self.collector = SignalCollector()
        self.handler = InteractionHandler()
        self.reinforcement = ReinforcementTracker(self.config)
        self.running = False
        self._suggestion_queue: queue.Queue = queue.Queue()

    def start(self) -> None:
        """Start the daemon loop."""
        self.running = True
        print(f"Ara Pheromone Daemon starting (mode: {self.config})")
        print("Press Ctrl+C to stop")

        try:
            while self.running:
                self._tick()
                time.sleep(self.config.update_interval)
        except KeyboardInterrupt:
            print("\nDaemon stopped")

    def stop(self) -> None:
        """Stop the daemon."""
        self.running = False

    def _tick(self) -> None:
        """One iteration of the pheromone loop."""
        # Collect signals
        signals = self.collector.get_signals()

        # Update state
        self.state = update_state(self.state, signals, self.config)

        # Decide interaction level
        level = decide_interaction(self.state, self.config)

        # Check if we have a suggestion to deliver
        suggestion = None
        try:
            suggestion = self._suggestion_queue.get_nowait()
        except queue.Empty:
            pass

        # Execute interaction
        if suggestion or level in (InteractionLevel.LOG, InteractionLevel.NONE):
            self.handler.execute(level, self.state, suggestion)

        # Debug output
        print(f"\r{self.state} → {level.value:12s}", end="", flush=True)

    def queue_suggestion(self, message: str, context: str = "", options: List[str] = None) -> None:
        """Queue a suggestion from Ara Brain."""
        self._suggestion_queue.put(Suggestion(
            message=message,
            context=context,
            options=options or [],
        ))

    def report_error(self) -> None:
        """Report a build/test error."""
        self.collector.report_error()

    def clear_errors(self) -> None:
        """Clear error count."""
        self.collector.clear_errors()

    def set_stuck(self, stuck: bool = True) -> None:
        """User pressed "I'm stuck"."""
        self.collector.set_stuck(stuck)

    def set_mode(self, mode: str) -> None:
        """Change operating mode."""
        if mode in MODES:
            self.config = MODES[mode]
            print(f"\nMode changed to: {mode}")

    def get_state(self) -> Dict:
        """Get current state for external queries."""
        return {
            "pheromones": self.state.to_dict(),
            "level": decide_interaction(self.state, self.config).value,
            "queued_suggestions": len(self.handler.queue),
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ara Pheromone Daemon")
    parser.add_argument(
        "--mode",
        choices=list(MODES.keys()),
        default="default",
        help="Operating mode",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    args = parser.parse_args()

    daemon = PheromoneDaemon(mode=args.mode)
    daemon.start()


if __name__ == "__main__":
    main()
