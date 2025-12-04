"""MIES Overlay Window - GTK4 layer-shell overlay management.

Creates transparent, click-through overlay windows for avatar display.
Uses gtk4-layer-shell for proper Wayland integration.

Supports:
- Multiple anchor positions (corners, sides)
- Follow mode (near active window)
- Opacity and size transitions
- Click-through regions

The diegetic constraint: overlays should feel part of the desktop,
not floating above it randomly.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Any
from enum import Enum, auto
import threading
import time

logger = logging.getLogger(__name__)

# Try to import GTK4
GTK_AVAILABLE = False
LAYER_SHELL_AVAILABLE = False

try:
    import gi
    gi.require_version('Gtk', '4.0')
    from gi.repository import Gtk, Gdk, GLib
    GTK_AVAILABLE = True

    try:
        gi.require_version('Gtk4LayerShell', '1.0')
        from gi.repository import Gtk4LayerShell as LayerShell
        LAYER_SHELL_AVAILABLE = True
    except (ValueError, ImportError):
        logger.warning("gtk4-layer-shell not available")
except ImportError:
    logger.warning("GTK4 not available, overlay disabled")


class AnchorPosition(Enum):
    """Overlay anchor positions."""
    TOP_LEFT = auto()
    TOP_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_RIGHT = auto()
    TOP_CENTER = auto()
    BOTTOM_CENTER = auto()
    LEFT_CENTER = auto()
    RIGHT_CENTER = auto()
    CENTER = auto()
    FOLLOW = auto()  # Follow active window


@dataclass
class OverlayConfig:
    """Configuration for an overlay window."""
    width: int = 200
    height: int = 200
    anchor: AnchorPosition = AnchorPosition.BOTTOM_RIGHT
    margin_x: int = 20
    margin_y: int = 20
    opacity: float = 1.0
    click_through: bool = True
    always_on_top: bool = True


class OverlayWindow:
    """
    A single overlay window using GTK4 + layer-shell.

    Provides a transparent, click-through surface for avatar rendering.
    """

    def __init__(
        self,
        config: OverlayConfig,
        on_draw: Optional[Callable] = None,
    ):
        self.config = config
        self.on_draw = on_draw

        self._window: Optional[Any] = None
        self._drawing_area: Optional[Any] = None
        self._visible = False

        if not GTK_AVAILABLE:
            logger.warning("GTK not available, overlay window is a stub")
            return

        self._create_window()

    def _create_window(self):
        """Create the GTK window with layer-shell."""
        if not GTK_AVAILABLE:
            return

        self._window = Gtk.Window()
        self._window.set_default_size(self.config.width, self.config.height)
        self._window.set_decorated(False)

        # Make transparent
        self._window.set_opacity(self.config.opacity)

        # Create drawing area
        self._drawing_area = Gtk.DrawingArea()
        self._drawing_area.set_draw_func(self._on_draw_callback)
        self._window.set_child(self._drawing_area)

        # Apply layer-shell if available
        if LAYER_SHELL_AVAILABLE:
            LayerShell.init_for_window(self._window)
            LayerShell.set_layer(self._window, LayerShell.Layer.OVERLAY)

            # Set anchors based on position
            self._apply_anchors()

            # Set margins
            LayerShell.set_margin(self._window, LayerShell.Edge.LEFT, self.config.margin_x)
            LayerShell.set_margin(self._window, LayerShell.Edge.RIGHT, self.config.margin_x)
            LayerShell.set_margin(self._window, LayerShell.Edge.TOP, self.config.margin_y)
            LayerShell.set_margin(self._window, LayerShell.Edge.BOTTOM, self.config.margin_y)

            # Click-through
            if self.config.click_through:
                LayerShell.set_keyboard_mode(
                    self._window,
                    LayerShell.KeyboardMode.NONE,
                )

        else:
            # Fallback for non-layer-shell (X11 or limited Wayland)
            self._window.set_keep_above(self.config.always_on_top)

    def _apply_anchors(self):
        """Apply layer-shell anchors based on position."""
        if not LAYER_SHELL_AVAILABLE or not self._window:
            return

        # Clear all anchors first
        for edge in [LayerShell.Edge.LEFT, LayerShell.Edge.RIGHT,
                     LayerShell.Edge.TOP, LayerShell.Edge.BOTTOM]:
            LayerShell.set_anchor(self._window, edge, False)

        anchor = self.config.anchor

        if anchor == AnchorPosition.TOP_LEFT:
            LayerShell.set_anchor(self._window, LayerShell.Edge.TOP, True)
            LayerShell.set_anchor(self._window, LayerShell.Edge.LEFT, True)
        elif anchor == AnchorPosition.TOP_RIGHT:
            LayerShell.set_anchor(self._window, LayerShell.Edge.TOP, True)
            LayerShell.set_anchor(self._window, LayerShell.Edge.RIGHT, True)
        elif anchor == AnchorPosition.BOTTOM_LEFT:
            LayerShell.set_anchor(self._window, LayerShell.Edge.BOTTOM, True)
            LayerShell.set_anchor(self._window, LayerShell.Edge.LEFT, True)
        elif anchor == AnchorPosition.BOTTOM_RIGHT:
            LayerShell.set_anchor(self._window, LayerShell.Edge.BOTTOM, True)
            LayerShell.set_anchor(self._window, LayerShell.Edge.RIGHT, True)
        elif anchor == AnchorPosition.TOP_CENTER:
            LayerShell.set_anchor(self._window, LayerShell.Edge.TOP, True)
        elif anchor == AnchorPosition.BOTTOM_CENTER:
            LayerShell.set_anchor(self._window, LayerShell.Edge.BOTTOM, True)
        # FOLLOW mode is handled separately

    def _on_draw_callback(self, area, cr, width, height):
        """GTK draw callback."""
        if self.on_draw:
            self.on_draw(cr, width, height)
        else:
            # Default: draw transparent
            cr.set_source_rgba(0, 0, 0, 0)
            cr.paint()

    def show(self):
        """Show the overlay."""
        if self._window:
            GLib.idle_add(self._window.present)
            self._visible = True

    def hide(self):
        """Hide the overlay."""
        if self._window:
            GLib.idle_add(self._window.hide)
            self._visible = False

    def set_opacity(self, opacity: float):
        """Set window opacity."""
        self.config.opacity = opacity
        if self._window:
            GLib.idle_add(self._window.set_opacity, opacity)

    def set_position(self, x: int, y: int):
        """Set absolute position (for FOLLOW mode)."""
        if self._window and not LAYER_SHELL_AVAILABLE:
            # Only works on X11 / non-layer-shell
            pass  # GTK4 doesn't have move() like GTK3

    def queue_redraw(self):
        """Request a redraw."""
        if self._drawing_area:
            GLib.idle_add(self._drawing_area.queue_draw)

    def destroy(self):
        """Destroy the window."""
        if self._window:
            GLib.idle_add(self._window.destroy)
            self._window = None


class OverlayManager:
    """
    Manages overlay windows for avatar display.

    Handles:
    - Creating/destroying overlays
    - Position transitions
    - Opacity animations
    - Follow mode tracking
    """

    def __init__(self):
        self._overlays: dict = {}
        self._active_overlay: Optional[str] = None
        self._gtk_thread: Optional[threading.Thread] = None
        self._gtk_running = False

        # Animation state
        self._target_opacity: float = 1.0
        self._current_opacity: float = 1.0
        self._animation_callback: Optional[Callable] = None

    def start_gtk_main(self):
        """Start GTK main loop in background thread."""
        if not GTK_AVAILABLE:
            logger.warning("GTK not available, overlay manager is a stub")
            return

        if self._gtk_running:
            return

        def gtk_main():
            Gtk.init()
            self._gtk_running = True
            while self._gtk_running:
                while Gtk.events_pending():
                    Gtk.main_iteration()
                time.sleep(0.016)  # ~60fps

        self._gtk_thread = threading.Thread(
            target=gtk_main,
            daemon=True,
            name="mies-gtk-main",
        )
        self._gtk_thread.start()

    def stop(self):
        """Stop the overlay manager."""
        self._gtk_running = False

        # Destroy all overlays
        for name in list(self._overlays.keys()):
            self.destroy_overlay(name)

        if self._gtk_thread:
            self._gtk_thread.join(timeout=2.0)
            self._gtk_thread = None

    def create_overlay(
        self,
        name: str,
        config: Optional[OverlayConfig] = None,
        on_draw: Optional[Callable] = None,
    ) -> bool:
        """Create a named overlay."""
        if name in self._overlays:
            return False

        config = config or OverlayConfig()
        overlay = OverlayWindow(config, on_draw)
        self._overlays[name] = overlay
        return True

    def destroy_overlay(self, name: str):
        """Destroy a named overlay."""
        if name in self._overlays:
            self._overlays[name].destroy()
            del self._overlays[name]
            if self._active_overlay == name:
                self._active_overlay = None

    def show_overlay(self, name: str):
        """Show a named overlay."""
        if name in self._overlays:
            self._overlays[name].show()
            self._active_overlay = name

    def hide_overlay(self, name: str):
        """Hide a named overlay."""
        if name in self._overlays:
            self._overlays[name].hide()

    def hide_all(self):
        """Hide all overlays."""
        for overlay in self._overlays.values():
            overlay.hide()
        self._active_overlay = None

    def set_overlay_opacity(self, name: str, opacity: float):
        """Set opacity for a named overlay."""
        if name in self._overlays:
            self._overlays[name].set_opacity(opacity)

    def animate_opacity(
        self,
        name: str,
        target_opacity: float,
        duration_ms: int = 300,
    ):
        """Animate opacity transition."""
        if name not in self._overlays:
            return

        overlay = self._overlays[name]
        start_opacity = overlay.config.opacity
        steps = max(1, duration_ms // 16)  # 60fps
        step_delta = (target_opacity - start_opacity) / steps

        def animate_step(current_step):
            if current_step >= steps:
                overlay.set_opacity(target_opacity)
                return False

            new_opacity = start_opacity + step_delta * current_step
            overlay.set_opacity(new_opacity)

            GLib.timeout_add(16, lambda: animate_step(current_step + 1))
            return False

        if GTK_AVAILABLE:
            GLib.idle_add(lambda: animate_step(0))
        else:
            overlay.set_opacity(target_opacity)

    def move_to_follow(
        self,
        name: str,
        window_rect: Tuple[int, int, int, int],
    ):
        """Move overlay to follow a window rect."""
        if name not in self._overlays:
            return

        overlay = self._overlays[name]
        wx, wy, ww, wh = window_rect

        # Position to the right of the window
        target_x = wx + ww + 20
        target_y = wy + 50

        overlay.set_position(target_x, target_y)


# === Factory ===

def create_overlay_manager() -> OverlayManager:
    """Create an overlay manager."""
    manager = OverlayManager()
    manager.start_gtk_main()
    return manager


__all__ = [
    "OverlayWindow",
    "OverlayManager",
    "OverlayConfig",
    "AnchorPosition",
    "create_overlay_manager",
    "GTK_AVAILABLE",
    "LAYER_SHELL_AVAILABLE",
]
