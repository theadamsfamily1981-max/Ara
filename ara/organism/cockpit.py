#!/usr/bin/env python3
"""
CorrSpike-HDC Subcortex Cockpit
================================

GTK4/Adwaita dashboard for monitoring the organism's emotional state.

Displays:
- Current emotion + strength
- VAD coordinates (valence, arousal, dominance)
- Sparsity + homeostasis deviation
- Context tags
- Eternal memory events (store/recall/dream)

Connects to emotion_bridge via WebSocket.

Usage:
    python cockpit.py  # GTK4 window
    python cockpit.py --terminal  # Terminal-only mode
"""

from __future__ import annotations
import asyncio
import json
import threading
import logging
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
from collections import deque

logger = logging.getLogger(__name__)

# Optional GTK4 imports
try:
    import gi
    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    from gi.repository import Gtk, Adw, GLib
    HAS_GTK = True
except (ImportError, ValueError):
    HAS_GTK = False
    Gtk = None
    Adw = None
    GLib = None

# Optional WebSocket
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


WS_URL = "ws://127.0.0.1:8765"
MAX_MEMORY_EVENTS = 100


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class EmotionDisplay:
    """Displayable emotion state."""
    emotion: str = "—"
    strength: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    sparsity: float = 0.0
    homeo_dev: float = 0.0
    tags: List[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class MemoryEventDisplay:
    """Displayable memory event."""
    event_type: str
    index: int
    emotion: str = ""
    similarity: float = 0.0
    strength: float = 0.0
    timestamp: str = ""


# ============================================================================
# WebSocket Listener
# ============================================================================

async def ws_listener(
    url: str,
    on_emotion: Callable[[dict], None],
    on_memory: Callable[[dict], None],
    on_hpv: Callable[[dict], None],
) -> None:
    """
    Listen to WebSocket and dispatch to callbacks.

    Args:
        url: WebSocket URL
        on_emotion: Callback for emotion updates
        on_memory: Callback for memory events
        on_hpv: Callback for HPV events
    """
    if not HAS_WEBSOCKETS:
        logger.error("websockets not installed")
        return

    retry_delay = 1.0

    while True:
        try:
            async with websockets.connect(url) as ws:
                logger.info(f"Connected to {url}")
                retry_delay = 1.0

                async for message in ws:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")
                        payload = data.get("data", {})

                        if msg_type == "emotion":
                            on_emotion(payload)
                        elif msg_type == "memory":
                            on_memory(payload)
                        elif msg_type == "hpv":
                            on_hpv(payload)

                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON: {message[:100]}")

        except Exception as e:
            logger.warning(f"WebSocket error: {e}, retrying in {retry_delay}s")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 30.0)


def start_ws_thread(
    on_emotion: Callable[[dict], None],
    on_memory: Callable[[dict], None],
    on_hpv: Callable[[dict], None],
    url: str = WS_URL,
) -> threading.Thread:
    """Start WebSocket listener in a background thread."""

    def run():
        asyncio.run(ws_listener(url, on_emotion, on_memory, on_hpv))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


# ============================================================================
# GTK4 Cockpit Application
# ============================================================================

if HAS_GTK:

    class CorrSpikeCockpit(Adw.Application):
        """GTK4/Adwaita cockpit application."""

        def __init__(self):
            super().__init__(application_id="org.ara.CorrSpikeCockpit")
            self.window = None
            self.labels: Dict[str, Gtk.Label] = {}
            self.memory_list: Optional[Gtk.ListBox] = None
            self.memory_events: deque = deque(maxlen=MAX_MEMORY_EVENTS)

        def do_activate(self):
            if self.window is None:
                self._build_ui()

            self.window.present()

            # Start WebSocket listener
            start_ws_thread(
                on_emotion=lambda d: GLib.idle_add(self._update_emotion, d),
                on_memory=lambda d: GLib.idle_add(self._add_memory_event, d),
                on_hpv=lambda d: GLib.idle_add(self._update_hpv, d),
            )

        def _build_ui(self):
            """Build the GTK UI."""
            self.window = Adw.ApplicationWindow(application=self)
            self.window.set_title("CorrSpike-HDC Subcortex Cockpit")
            self.window.set_default_size(600, 500)

            # Main vertical box
            main_box = Gtk.Box(
                orientation=Gtk.Orientation.VERTICAL,
                spacing=12,
                margin_top=16,
                margin_bottom=16,
                margin_start=16,
                margin_end=16,
            )
            self.window.set_content(main_box)

            # Header
            header = Gtk.Label(label="Subcortex Emotional State")
            header.add_css_class("title-2")
            main_box.append(header)

            # Emotion display
            emo_frame = Gtk.Frame(label="Current Emotion")
            emo_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            emo_box.set_margin_top(8)
            emo_box.set_margin_bottom(8)
            emo_box.set_margin_start(8)
            emo_box.set_margin_end(8)
            emo_frame.set_child(emo_box)
            main_box.append(emo_frame)

            self.labels["emotion"] = Gtk.Label(label="Emotion: —")
            self.labels["emotion"].set_xalign(0.0)
            self.labels["emotion"].add_css_class("title-3")
            emo_box.append(self.labels["emotion"])

            self.labels["vad"] = Gtk.Label(
                label="V: —  A: —  D: —"
            )
            self.labels["vad"].set_xalign(0.0)
            emo_box.append(self.labels["vad"])

            self.labels["meta"] = Gtk.Label(
                label="Sparsity: —  HomeoDev: —"
            )
            self.labels["meta"].set_xalign(0.0)
            emo_box.append(self.labels["meta"])

            self.labels["tags"] = Gtk.Label(label="Tags: —")
            self.labels["tags"].set_xalign(0.0)
            emo_box.append(self.labels["tags"])

            self.labels["timestamp"] = Gtk.Label(label="Last: —")
            self.labels["timestamp"].set_xalign(0.0)
            self.labels["timestamp"].add_css_class("dim-label")
            emo_box.append(self.labels["timestamp"])

            # Memory events
            mem_frame = Gtk.Frame(label="Eternal Memory Events")
            mem_scroll = Gtk.ScrolledWindow()
            mem_scroll.set_min_content_height(150)
            mem_scroll.set_vexpand(True)
            mem_frame.set_child(mem_scroll)
            main_box.append(mem_frame)

            self.memory_list = Gtk.ListBox()
            self.memory_list.set_selection_mode(Gtk.SelectionMode.NONE)
            mem_scroll.set_child(self.memory_list)

            # Status bar
            self.labels["status"] = Gtk.Label(label="Waiting for connection...")
            self.labels["status"].add_css_class("dim-label")
            self.labels["status"].set_xalign(0.0)
            main_box.append(self.labels["status"])

        def _update_emotion(self, data: dict) -> None:
            """Update emotion display."""
            emotion = data.get("emotion", "—")
            strength = data.get("strength", 0.0)
            valence = data.get("valence", 0.0)
            arousal = data.get("arousal", 0.0)
            dominance = data.get("dominance", 0.0)
            sparsity = data.get("sparsity", 0.0)
            homeo_dev = data.get("homeo_dev", 0.0)
            tags = data.get("tags", [])

            self.labels["emotion"].set_text(
                f"Emotion: {emotion} ({strength:.0%})"
            )
            self.labels["vad"].set_text(
                f"V: {valence:+.2f}  A: {arousal:+.2f}  D: {dominance:+.2f}"
            )
            self.labels["meta"].set_text(
                f"Sparsity: {sparsity:.2f}  HomeoDev: {homeo_dev:.3f}"
            )
            self.labels["tags"].set_text(
                f"Tags: {', '.join(tags) if tags else '—'}"
            )
            self.labels["timestamp"].set_text(
                f"Last: {datetime.now().strftime('%H:%M:%S')}"
            )
            self.labels["status"].set_text("Connected")

        def _add_memory_event(self, data: dict) -> None:
            """Add a memory event to the list."""
            event_type = data.get("event_type", "?")
            index = data.get("index", -1)
            emotion = data.get("emotion", "")
            similarity = data.get("similarity", 0.0)
            strength = data.get("strength", 0.0)

            # Create row
            row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            row.set_margin_top(4)
            row.set_margin_bottom(4)

            # Type badge
            type_colors = {
                "store": "suggested-action",
                "recall": "warning",
                "dream": "accent",
            }
            type_label = Gtk.Label(label=event_type.upper())
            type_label.add_css_class(type_colors.get(event_type, ""))
            type_label.set_size_request(60, -1)
            row.append(type_label)

            # Details
            if event_type == "store":
                detail = f"idx={index} {emotion} str={strength:.2f}"
            else:
                detail = f"idx={index} sim={similarity:.2f} str={strength:.2f}"

            detail_label = Gtk.Label(label=detail)
            detail_label.set_xalign(0.0)
            detail_label.set_hexpand(True)
            row.append(detail_label)

            # Timestamp
            ts_label = Gtk.Label(label=datetime.now().strftime("%H:%M:%S"))
            ts_label.add_css_class("dim-label")
            row.append(ts_label)

            # Add to list (prepend = newest first)
            self.memory_list.prepend(row)

            # Trim old entries
            while True:
                children = list(self.memory_list)
                if len(children) <= MAX_MEMORY_EVENTS:
                    break
                self.memory_list.remove(children[-1])

        def _update_hpv(self, data: dict) -> None:
            """Update HPV display (optional)."""
            pass  # Could add HPV panel


    def run_gtk_cockpit():
        """Run the GTK cockpit."""
        app = CorrSpikeCockpit()
        app.run(None)


# ============================================================================
# Terminal Cockpit (fallback)
# ============================================================================

class TerminalCockpit:
    """Simple terminal-based cockpit."""

    def __init__(self):
        self.last_emotion: Optional[dict] = None
        self.memory_events: deque = deque(maxlen=10)

    def update_emotion(self, data: dict) -> None:
        """Update emotion display."""
        self.last_emotion = data
        self._render()

    def add_memory_event(self, data: dict) -> None:
        """Add memory event."""
        data["timestamp"] = datetime.now().strftime("%H:%M:%S")
        self.memory_events.appendleft(data)
        self._render()

    def update_hpv(self, data: dict) -> None:
        """Update HPV (ignored in terminal)."""
        pass

    def _render(self) -> None:
        """Render to terminal."""
        # Clear screen
        print("\033[H\033[J", end="")

        print("=" * 60)
        print("CorrSpike-HDC Subcortex Cockpit (Terminal)")
        print("=" * 60)

        if self.last_emotion:
            e = self.last_emotion
            print(f"\nEmotion: {e.get('emotion', '—')} ({e.get('strength', 0):.0%})")
            print(f"  V: {e.get('valence', 0):+.2f}  "
                  f"A: {e.get('arousal', 0):+.2f}  "
                  f"D: {e.get('dominance', 0):+.2f}")
            print(f"  Sparsity: {e.get('sparsity', 0):.2f}  "
                  f"HomeoDev: {e.get('homeo_dev', 0):.3f}")
            tags = e.get('tags', [])
            print(f"  Tags: {', '.join(tags) if tags else '—'}")

        if self.memory_events:
            print("\nRecent Memory Events:")
            for evt in list(self.memory_events)[:5]:
                et = evt.get("event_type", "?")
                idx = evt.get("index", -1)
                ts = evt.get("timestamp", "")
                print(f"  [{ts}] {et.upper():6s} idx={idx}")

        print("\n" + "=" * 60)
        print("Press Ctrl+C to exit")

    def run(self) -> None:
        """Run terminal cockpit."""
        print("Starting terminal cockpit...")
        print("Connecting to WebSocket...")

        start_ws_thread(
            on_emotion=self.update_emotion,
            on_memory=self.add_memory_event,
            on_hpv=self.update_hpv,
        )

        try:
            while True:
                import time
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nExiting...")


def run_terminal_cockpit():
    """Run terminal-based cockpit."""
    cockpit = TerminalCockpit()
    cockpit.run()


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="CorrSpike-HDC Cockpit")
    parser.add_argument("--terminal", action="store_true",
                        help="Use terminal mode instead of GTK")
    parser.add_argument("--url", default="ws://127.0.0.1:8765",
                        help="WebSocket URL")

    args = parser.parse_args()

    # Update module-level WS_URL for use by other functions
    global WS_URL
    WS_URL = args.url

    if args.terminal or not HAS_GTK:
        if not HAS_GTK and not args.terminal:
            print("GTK4 not available, using terminal mode")
        run_terminal_cockpit()
    else:
        run_gtk_cockpit()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
