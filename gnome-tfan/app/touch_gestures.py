#!/usr/bin/env python3
"""
Touch Gesture Recognizers for Cockpit HUD

Provides touch gesture recognition for the touchscreen cockpit:
- Swipe (up/down/left/right)
- Pinch (zoom in/out)
- Long press (context menu)
- Two-finger rotate

Uses GTK4 gesture controllers.
"""

import gi
gi.require_version('Gtk', '4.0')
from gi.repository import Gtk, Gdk, GLib
from enum import Enum
import math
import logging

logger = logging.getLogger(__name__)


class SwipeDirection(Enum):
    """Swipe direction enumeration."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    NONE = "none"


class GestureHandler:
    """
    Unified gesture handler for touchscreen interactions.

    Attaches multiple gesture controllers to a widget and
    emits callbacks for recognized gestures.
    """

    def __init__(self, widget: Gtk.Widget):
        """
        Initialize gesture handler.

        Args:
            widget: GTK widget to attach gestures to
        """
        self.widget = widget
        self.controllers = []

        # Callbacks
        self.on_swipe = None
        self.on_pinch = None
        self.on_rotate = None
        self.on_long_press = None
        self.on_tap = None

        # Gesture state
        self.pinch_scale = 1.0
        self.rotation_angle = 0.0

        self._setup_gestures()

    def _setup_gestures(self):
        """Set up all gesture controllers."""
        # Swipe gesture
        swipe = Gtk.GestureSwipe()
        swipe.connect("swipe", self._on_swipe)
        self.widget.add_controller(swipe)
        self.controllers.append(swipe)

        # Zoom (pinch) gesture
        zoom = Gtk.GestureZoom()
        zoom.connect("scale-changed", self._on_zoom_scale_changed)
        zoom.connect("begin", self._on_zoom_begin)
        zoom.connect("end", self._on_zoom_end)
        self.widget.add_controller(zoom)
        self.controllers.append(zoom)

        # Rotate gesture
        rotate = Gtk.GestureRotate()
        rotate.connect("angle-changed", self._on_rotate_angle_changed)
        rotate.connect("begin", self._on_rotate_begin)
        rotate.connect("end", self._on_rotate_end)
        self.widget.add_controller(rotate)
        self.controllers.append(rotate)

        # Long press gesture
        long_press = Gtk.GestureLongPress()
        long_press.set_delay_factor(1.0)  # Standard delay
        long_press.connect("pressed", self._on_long_press)
        self.widget.add_controller(long_press)
        self.controllers.append(long_press)

        # Click/tap gesture
        click = Gtk.GestureClick()
        click.connect("pressed", self._on_click_pressed)
        click.connect("released", self._on_click_released)
        self.widget.add_controller(click)
        self.controllers.append(click)

        logger.info("[Gestures] Controllers attached to widget")

    def _on_swipe(self, gesture, velocity_x, velocity_y):
        """Handle swipe gesture."""
        # Determine direction from velocity
        direction = self._get_swipe_direction(velocity_x, velocity_y)

        if direction != SwipeDirection.NONE:
            logger.debug(f"[Gestures] Swipe detected: {direction.value}")

            if self.on_swipe:
                self.on_swipe(direction, velocity_x, velocity_y)

    def _get_swipe_direction(self, vx, vy) -> SwipeDirection:
        """
        Determine swipe direction from velocity.

        Args:
            vx: X velocity
            vy: Y velocity

        Returns:
            SwipeDirection: Detected direction
        """
        min_velocity = 100  # Minimum velocity threshold

        abs_vx = abs(vx)
        abs_vy = abs(vy)

        if abs_vx < min_velocity and abs_vy < min_velocity:
            return SwipeDirection.NONE

        if abs_vx > abs_vy:
            # Horizontal swipe
            return SwipeDirection.RIGHT if vx > 0 else SwipeDirection.LEFT
        else:
            # Vertical swipe
            return SwipeDirection.DOWN if vy > 0 else SwipeDirection.UP

    def _on_zoom_begin(self, gesture, sequence):
        """Handle zoom gesture start."""
        self.pinch_scale = 1.0
        logger.debug("[Gestures] Zoom begin")

    def _on_zoom_scale_changed(self, gesture, scale):
        """Handle zoom scale change."""
        self.pinch_scale = scale

        if self.on_pinch:
            self.on_pinch(scale, "changed")

    def _on_zoom_end(self, gesture, sequence):
        """Handle zoom gesture end."""
        logger.debug(f"[Gestures] Zoom end: scale={self.pinch_scale:.2f}")

        if self.on_pinch:
            self.on_pinch(self.pinch_scale, "end")

    def _on_rotate_begin(self, gesture, sequence):
        """Handle rotate gesture start."""
        self.rotation_angle = 0.0
        logger.debug("[Gestures] Rotate begin")

    def _on_rotate_angle_changed(self, gesture, angle, angle_delta):
        """Handle rotation angle change."""
        self.rotation_angle = math.degrees(angle)

        if self.on_rotate:
            self.on_rotate(self.rotation_angle, angle_delta)

    def _on_rotate_end(self, gesture, sequence):
        """Handle rotate gesture end."""
        logger.debug(f"[Gestures] Rotate end: angle={self.rotation_angle:.1f}°")

        if self.on_rotate:
            self.on_rotate(self.rotation_angle, 0)

    def _on_long_press(self, gesture, x, y):
        """Handle long press gesture."""
        logger.debug(f"[Gestures] Long press at ({x:.0f}, {y:.0f})")

        if self.on_long_press:
            self.on_long_press(x, y)

    def _on_click_pressed(self, gesture, n_press, x, y):
        """Handle click/tap press."""
        if self.on_tap:
            self.on_tap(x, y, "pressed", n_press)

    def _on_click_released(self, gesture, n_press, x, y):
        """Handle click/tap release."""
        if self.on_tap:
            self.on_tap(x, y, "released", n_press)


class ScrollableGestureArea(Gtk.ScrolledWindow):
    """
    Scrollable area with touch gesture support.

    Provides natural touch scrolling with kinetic effects.
    """

    def __init__(self):
        """Initialize scrollable gesture area."""
        super().__init__()

        # Enable kinetic scrolling
        self.set_kinetic_scrolling(True)

        # Touch-friendly scroll policy
        self.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)

        # Capture scroll events
        self.set_propagate_natural_height(True)
        self.set_propagate_natural_width(True)

        # Add swipe gesture for page navigation
        self.gesture_handler = None
        self.on_page_swipe = None

    def set_swipe_callback(self, callback):
        """
        Set callback for swipe gestures.

        Args:
            callback: Function(direction: SwipeDirection)
        """
        if not self.gesture_handler:
            self.gesture_handler = GestureHandler(self)

        def handle_swipe(direction, vx, vy):
            # Only trigger page swipe for fast horizontal swipes
            if direction in [SwipeDirection.LEFT, SwipeDirection.RIGHT]:
                if abs(vx) > 500:  # Fast swipe threshold
                    callback(direction)

        self.gesture_handler.on_swipe = handle_swipe


class RippleEffect:
    """
    Touch ripple feedback effect.

    Creates expanding circle animation on touch.
    """

    def __init__(self, widget: Gtk.Widget):
        """
        Initialize ripple effect.

        Args:
            widget: Widget to add ripple effect to
        """
        self.widget = widget
        self.ripple_overlay = None
        self._setup_ripple()

    def _setup_ripple(self):
        """Set up ripple effect CSS and controller."""
        # Add click controller for ripple trigger
        click = Gtk.GestureClick()
        click.connect("pressed", self._trigger_ripple)
        self.widget.add_controller(click)

        # Add ripple CSS class
        self.widget.add_css_class("ripple-effect")

    def _trigger_ripple(self, gesture, n_press, x, y):
        """
        Trigger ripple animation at touch point.

        Args:
            x, y: Touch coordinates
        """
        # Create ripple element using CSS custom properties
        # The actual animation is handled by CSS
        allocation = self.widget.get_allocation()

        # Calculate ripple size (diameter should cover widget)
        size = max(allocation.width, allocation.height) * 2

        # Set CSS custom properties for ripple position
        # This requires the widget to have CSS that uses these properties
        self.widget.add_css_class("ripple-active")

        # Remove class after animation
        GLib.timeout_add(400, lambda: self.widget.remove_css_class("ripple-active"))


def get_ripple_css() -> str:
    """
    Get CSS for ripple effect.

    Returns:
        str: CSS string
    """
    return """
    /* Ripple effect container */
    .ripple-effect {
        position: relative;
        overflow: hidden;
    }

    /* Ripple animation */
    .ripple-effect::after {
        content: "";
        position: absolute;
        top: 50%;
        left: 50%;
        width: 5px;
        height: 5px;
        background: radial-gradient(
            circle,
            rgba(0, 212, 255, 0.6) 0%,
            rgba(0, 212, 255, 0) 70%
        );
        border-radius: 50%;
        transform: translate(-50%, -50%) scale(0);
        opacity: 0;
        pointer-events: none;
    }

    .ripple-effect.ripple-active::after {
        animation: ripple-expand 0.4s ease-out forwards;
    }

    @keyframes ripple-expand {
        0% {
            transform: translate(-50%, -50%) scale(0);
            opacity: 1;
        }
        100% {
            transform: translate(-50%, -50%) scale(100);
            opacity: 0;
        }
    }

    /* Touch feedback for buttons */
    .touch-button {
        transition: transform 0.1s ease, background 0.15s ease;
    }

    .touch-button:active {
        transform: scale(0.95);
    }

    /* Pressed state feedback */
    .touch-pressed {
        transform: scale(0.97);
        filter: brightness(1.1);
    }
    """


def setup_touch_feedback(widget: Gtk.Widget, press_class: str = "touch-pressed"):
    """
    Add basic touch feedback to a widget.

    Args:
        widget: Widget to add feedback to
        press_class: CSS class to add on press
    """
    click = Gtk.GestureClick()

    def on_press(gesture, n_press, x, y):
        widget.add_css_class(press_class)

    def on_release(gesture, n_press, x, y):
        widget.remove_css_class(press_class)

    click.connect("pressed", on_press)
    click.connect("released", on_release)
    widget.add_controller(click)


# Example usage / testing
if __name__ == '__main__':
    import sys

    app = Gtk.Application(application_id='com.quanta.tfan.gesture-test')

    def on_activate(app):
        window = Gtk.ApplicationWindow(application=app)
        window.set_default_size(800, 600)
        window.set_title("Touch Gesture Test")

        # Apply ripple CSS
        css_provider = Gtk.CssProvider()
        css_provider.load_from_string(get_ripple_css())
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # Main container
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=20)
        box.set_margin_top(20)
        box.set_margin_bottom(20)
        box.set_margin_start(20)
        box.set_margin_end(20)
        window.set_child(box)

        # Title
        title = Gtk.Label(label="Touch Gesture Test")
        title.add_css_class("title-1")
        box.append(title)

        # Gesture test area
        gesture_area = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        gesture_area.set_hexpand(True)
        gesture_area.set_vexpand(True)
        gesture_area.add_css_class("card")
        gesture_area.add_css_class("ripple-effect")

        # Status label
        status = Gtk.Label(label="Perform gestures here")
        status.set_vexpand(True)
        status.set_valign(Gtk.Align.CENTER)
        gesture_area.append(status)

        box.append(gesture_area)

        # Attach gesture handler
        handler = GestureHandler(gesture_area)

        def on_swipe(direction, vx, vy):
            status.set_text(f"Swipe: {direction.value}\nVelocity: ({vx:.0f}, {vy:.0f})")

        def on_pinch(scale, state):
            status.set_text(f"Pinch: {scale:.2f}x\nState: {state}")

        def on_rotate(angle, delta):
            status.set_text(f"Rotate: {angle:.1f}°")

        def on_long_press(x, y):
            status.set_text(f"Long Press at ({x:.0f}, {y:.0f})")

        def on_tap(x, y, state, n):
            if state == "pressed":
                gesture_area.add_css_class("ripple-active")
                GLib.timeout_add(400, lambda: gesture_area.remove_css_class("ripple-active"))

        handler.on_swipe = on_swipe
        handler.on_pinch = on_pinch
        handler.on_rotate = on_rotate
        handler.on_long_press = on_long_press
        handler.on_tap = on_tap

        # Test buttons with ripple
        button_box = Gtk.Box(spacing=12)
        button_box.set_halign(Gtk.Align.CENTER)

        for label in ["Button 1", "Button 2", "Button 3"]:
            btn = Gtk.Button(label=label)
            btn.add_css_class("ripple-effect")
            btn.add_css_class("touch-button")
            RippleEffect(btn)
            button_box.append(btn)

        box.append(button_box)

        window.present()

    app.connect('activate', on_activate)
    app.run(None)
