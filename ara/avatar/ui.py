"""
Avatar UI for Ara

Visual representation of Ara's emotional state.
Uses GTK4 + libadwaita for a modern Linux look.

Install:
    sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0 gir1.2-adw-1

Fallback: Simple terminal UI if GTK not available.
"""

import logging
import threading
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("ara.avatar.ui")

# Try to import GTK
GTK_AVAILABLE = False
Gtk = None
Adw = None
GLib = None
Gdk = None
Pango = None

try:
    import gi
    gi.require_version('Gtk', '4.0')
    gi.require_version('Adw', '1')
    from gi.repository import Gtk, Adw, GLib, Gdk, Pango
    GTK_AVAILABLE = True
    logger.info("GTK4 + libadwaita available")
except (ImportError, ValueError) as e:
    logger.warning(f"GTK not available: {e}. Using terminal UI.")


@dataclass
class AvatarState:
    """Current state of the avatar."""
    listening: bool = False
    thinking: bool = False
    speaking: bool = False
    user_text: str = ""
    ara_text: str = ""
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    risk_level: str = "LOW"
    kitten_steps: int = 0
    kitten_spike_rate: float = 0.0


class AvatarWindow:
    """
    Ara Avatar Window.

    Shows:
    - Ara's "face" (colored circle/emoji based on mood)
    - Status indicator (listening/thinking/speaking)
    - PAD emotional state visualization
    - Recent conversation text
    - Kitten SNN stats
    """

    def __init__(self):
        self.state = AvatarState()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._window = None
        self._app = None

    def start(self):
        """Start the avatar UI in a background thread."""
        if GTK_AVAILABLE:
            self._thread = threading.Thread(target=self._run_gtk, daemon=True)
            self._thread.start()
            time.sleep(0.5)  # Give GTK time to initialize
        else:
            logger.info("Using terminal UI (GTK not available)")
            self._running = True

    def stop(self):
        """Stop the avatar UI."""
        self._running = False
        if GTK_AVAILABLE and self._app:
            GLib.idle_add(self._app.quit)

    def _run_gtk(self):
        """Run GTK main loop."""
        self._app = Adw.Application(application_id="org.ara.avatar")
        self._app.connect("activate", self._on_activate)
        self._running = True
        self._app.run(None)

    def _on_activate(self, app):
        """Build the GTK window."""
        # Create window
        self._window = Adw.ApplicationWindow(application=app)
        self._window.set_title("Ara")
        self._window.set_default_size(400, 600)

        # Main box
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.set_margin_top(20)
        main_box.set_margin_bottom(20)
        main_box.set_margin_start(20)
        main_box.set_margin_end(20)

        # Header
        header = Adw.HeaderBar()
        self._window.set_titlebar(header)

        # Face area (drawing area for mood visualization)
        self._face_area = Gtk.DrawingArea()
        self._face_area.set_size_request(200, 200)
        self._face_area.set_halign(Gtk.Align.CENTER)
        self._face_area.set_draw_func(self._draw_face)
        main_box.append(self._face_area)

        # Status label
        self._status_label = Gtk.Label(label="Ready")
        self._status_label.add_css_class("title-2")
        main_box.append(self._status_label)

        # PAD display
        pad_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        pad_box.set_halign(Gtk.Align.CENTER)

        self._valence_bar = self._create_pad_bar("V", "Valence")
        self._arousal_bar = self._create_pad_bar("A", "Arousal")
        self._dominance_bar = self._create_pad_bar("D", "Dominance")

        pad_box.append(self._valence_bar)
        pad_box.append(self._arousal_bar)
        pad_box.append(self._dominance_bar)
        main_box.append(pad_box)

        # Kitten stats
        self._kitten_label = Gtk.Label(label="Kitten: -")
        self._kitten_label.add_css_class("caption")
        main_box.append(self._kitten_label)

        # Separator
        main_box.append(Gtk.Separator())

        # Conversation area
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        self._conversation_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        scroll.set_child(self._conversation_box)
        main_box.append(scroll)

        # Set content
        self._window.set_content(main_box)
        self._window.present()

        # Start update timer
        GLib.timeout_add(100, self._update_ui)

    def _create_pad_bar(self, letter: str, tooltip: str):
        """Create a PAD indicator bar."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        label = Gtk.Label(label=letter)
        label.add_css_class("heading")
        box.append(label)

        bar = Gtk.LevelBar()
        bar.set_min_value(-1.0)
        bar.set_max_value(1.0)
        bar.set_value(0.0)
        bar.set_size_request(60, 20)
        bar.set_tooltip_text(tooltip)
        box.append(bar)

        box._bar = bar
        return box

    def _draw_face(self, area, cr, width, height):
        """Draw Ara's face based on emotional state."""
        # Center coordinates
        cx, cy = width / 2, height / 2
        radius = min(width, height) / 2 - 10

        # Color based on valence (red-negative to green-positive)
        v = self.state.valence
        if v > 0:
            r, g, b = 0.3, 0.5 + v * 0.5, 0.3
        elif v < 0:
            r, g, b = 0.5 - v * 0.5, 0.3, 0.3
        else:
            r, g, b = 0.4, 0.4, 0.5

        # Brightness based on arousal
        a = self.state.arousal
        brightness = 0.5 + a * 0.5
        r, g, b = r * brightness, g * brightness, b * brightness

        # Draw face circle
        cr.set_source_rgb(r, g, b)
        cr.arc(cx, cy, radius, 0, 2 * 3.14159)
        cr.fill()

        # Draw status indicator
        if self.state.listening:
            # Pulsing ring for listening
            cr.set_source_rgb(0.2, 0.6, 0.9)
            cr.set_line_width(4)
            cr.arc(cx, cy, radius + 8, 0, 2 * 3.14159)
            cr.stroke()
        elif self.state.thinking:
            # Dots for thinking
            cr.set_source_rgb(0.9, 0.7, 0.2)
            for i in range(3):
                angle = (time.time() * 2 + i * 2.1) % (2 * 3.14159)
                dx = cx + radius * 0.5 * (i - 1) * 0.4
                cr.arc(dx, cy + 30, 5, 0, 2 * 3.14159)
                cr.fill()
        elif self.state.speaking:
            # Waves for speaking
            cr.set_source_rgb(0.2, 0.8, 0.4)
            cr.set_line_width(3)
            for i in range(3):
                wave_radius = radius + 15 + i * 10
                cr.arc(cx, cy, wave_radius, -0.5, 0.5)
                cr.stroke()

        # Draw eyes
        cr.set_source_rgb(1, 1, 1)
        eye_y = cy - radius * 0.2
        eye_x_offset = radius * 0.3

        # Eye size based on arousal
        eye_size = 8 + self.state.arousal * 4

        cr.arc(cx - eye_x_offset, eye_y, eye_size, 0, 2 * 3.14159)
        cr.fill()
        cr.arc(cx + eye_x_offset, eye_y, eye_size, 0, 2 * 3.14159)
        cr.fill()

        # Pupils
        cr.set_source_rgb(0.1, 0.1, 0.1)
        pupil_size = eye_size * 0.5
        cr.arc(cx - eye_x_offset, eye_y, pupil_size, 0, 2 * 3.14159)
        cr.fill()
        cr.arc(cx + eye_x_offset, eye_y, pupil_size, 0, 2 * 3.14159)
        cr.fill()

        # Mouth based on valence
        mouth_y = cy + radius * 0.3
        cr.set_source_rgb(0.2, 0.2, 0.2)
        cr.set_line_width(3)

        if self.state.valence > 0.2:
            # Smile
            cr.arc(cx, mouth_y + 10, 20, 0.2, 2.94)
        elif self.state.valence < -0.2:
            # Frown
            cr.arc(cx, mouth_y + 30, 20, 3.34, 6.08)
        else:
            # Neutral
            cr.move_to(cx - 15, mouth_y)
            cr.line_to(cx + 15, mouth_y)

        cr.stroke()

    def _update_ui(self) -> bool:
        """Update UI elements periodically."""
        if not self._running:
            return False

        # Update status label
        if self.state.listening:
            status = "Listening..."
        elif self.state.thinking:
            status = "Thinking..."
        elif self.state.speaking:
            status = "Speaking..."
        else:
            status = "Ready"
        self._status_label.set_label(status)

        # Update PAD bars
        self._valence_bar._bar.set_value(self.state.valence)
        self._arousal_bar._bar.set_value(self.state.arousal)
        self._dominance_bar._bar.set_value(self.state.dominance)

        # Update kitten stats
        self._kitten_label.set_label(
            f"Kitten: {self.state.kitten_steps} steps | "
            f"{self.state.kitten_spike_rate:.1%} spike rate"
        )

        # Redraw face
        self._face_area.queue_draw()

        return True

    # ============================================================
    # Public API
    # ============================================================

    def set_listening(self, listening: bool):
        """Set listening state."""
        self.state.listening = listening
        self.state.thinking = False
        self.state.speaking = False

    def set_thinking(self, thinking: bool):
        """Set thinking state."""
        self.state.listening = False
        self.state.thinking = thinking
        self.state.speaking = False

    def set_speaking(self, speaking: bool):
        """Set speaking state."""
        self.state.listening = False
        self.state.thinking = False
        self.state.speaking = speaking

    def update_state(self, pad: Dict[str, float], clv: Dict[str, Any]):
        """Update emotional state from Ara Core response."""
        self.state.valence = pad.get("valence", 0.0)
        self.state.arousal = pad.get("arousal", 0.0)
        self.state.dominance = pad.get("dominance", 0.0)
        self.state.risk_level = clv.get("risk_level", "LOW")

    def update_kitten(self, steps: int, spike_rate: float):
        """Update Kitten SNN stats."""
        self.state.kitten_steps = steps
        self.state.kitten_spike_rate = spike_rate

    def show_user_text(self, text: str):
        """Show user's utterance."""
        self.state.user_text = text
        if GTK_AVAILABLE and self._conversation_box:
            GLib.idle_add(self._add_message, "You", text, False)

    def show_ara_text(self, text: str):
        """Show Ara's response."""
        self.state.ara_text = text
        if GTK_AVAILABLE and self._conversation_box:
            GLib.idle_add(self._add_message, "Ara", text, True)

    def _add_message(self, sender: str, text: str, is_ara: bool):
        """Add a message to the conversation view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        box.set_halign(Gtk.Align.END if is_ara else Gtk.Align.START)

        label = Gtk.Label(label=f"{sender}:")
        label.add_css_class("caption")
        label.set_halign(Gtk.Align.START)
        box.append(label)

        msg = Gtk.Label(label=text)
        msg.set_wrap(True)
        msg.set_wrap_mode(Pango.WrapMode.WORD_CHAR)
        msg.set_max_width_chars(40)
        msg.set_halign(Gtk.Align.START)

        if is_ara:
            msg.add_css_class("accent")

        box.append(msg)
        self._conversation_box.append(box)


# ============================================================
# Terminal UI Fallback
# ============================================================

class TerminalAvatarUI:
    """Simple terminal-based avatar UI."""

    def __init__(self):
        self.state = AvatarState()
        self._running = False

    def start(self):
        self._running = True
        print("\n" + "=" * 50)
        print("  Ara Avatar (Terminal Mode)")
        print("=" * 50 + "\n")

    def stop(self):
        self._running = False

    def set_listening(self, listening: bool):
        if listening:
            print("  [🎤 Listening...]")

    def set_thinking(self, thinking: bool):
        if thinking:
            print("  [💭 Thinking...]")

    def set_speaking(self, speaking: bool):
        if speaking:
            print("  [🔊 Speaking...]")

    def update_state(self, pad: Dict[str, float], clv: Dict[str, Any]):
        v = pad.get("valence", 0)
        a = pad.get("arousal", 0)
        d = pad.get("dominance", 0)
        risk = clv.get("risk_level", "LOW")
        print(f"  PAD: V={v:+.2f} A={a:.2f} D={d:.2f} | Risk: {risk}")

    def update_kitten(self, steps: int, spike_rate: float):
        print(f"  Kitten: {steps} steps | {spike_rate:.1%} spike rate")

    def show_user_text(self, text: str):
        print(f"\n  You: {text}")

    def show_ara_text(self, text: str):
        print(f"\n  Ara: {text}")


# ============================================================
# Factory
# ============================================================

def create_avatar_ui() -> AvatarWindow:
    """Create the appropriate avatar UI."""
    if GTK_AVAILABLE:
        return AvatarWindow()
    else:
        return TerminalAvatarUI()


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print(f"GTK available: {GTK_AVAILABLE}")

    ui = create_avatar_ui()
    ui.start()

    # Simulate some states
    time.sleep(1)
    ui.set_listening(True)
    time.sleep(2)
    ui.show_user_text("Hello Ara!")
    ui.set_listening(False)
    ui.set_thinking(True)
    time.sleep(2)
    ui.update_state(
        pad={"valence": 0.5, "arousal": 0.6, "dominance": 0.4},
        clv={"risk_level": "LOW"}
    )
    ui.set_thinking(False)
    ui.set_speaking(True)
    ui.show_ara_text("Hello! I'm doing great, thanks for asking!")
    time.sleep(3)
    ui.set_speaking(False)

    # Keep running if GTK
    if GTK_AVAILABLE:
        input("Press Enter to exit...")
    ui.stop()
