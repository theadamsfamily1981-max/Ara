#!/usr/bin/env python3
"""
Video Background for Cockpit HUD

GStreamer-based video background player for the touchscreen cockpit.
Plays looping hologram/robot animations with transparency effects.

Features:
- Loop playback
- Opacity control
- Blur/filter effects
- Pause when topology active
- Hardware-accelerated decoding
"""

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
from gi.repository import Gtk, Gst, GstVideo, GLib, Gio
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VideoBackground:
    """
    GStreamer video background for cockpit HUD.

    Plays a looping video with configurable opacity and filters.
    """

    def __init__(self, video_path: str = None):
        """
        Initialize video background player.

        Args:
            video_path: Path to video file (webm/mp4)
        """
        # Initialize GStreamer
        Gst.init(None)

        self.video_path = video_path
        self.player = None
        self.sink = None
        self.picture = None
        self.is_playing = False
        self.opacity = 0.2  # Default 20% opacity
        self.paused = False

        # Default video location
        if not self.video_path:
            default_paths = [
                Path(__file__).parent.parent / 'assets' / 'hologram_loop.webm',
                Path(__file__).parent.parent / 'assets' / 'hologram_loop.mp4',
                Path(__file__).parent.parent / 'assets' / 'robot_loop.webm',
                Path(__file__).parent.parent / 'assets' / 'background.webm',
            ]
            for path in default_paths:
                if path.exists():
                    self.video_path = str(path)
                    break

        self._setup_player()

    def _setup_player(self):
        """Set up GStreamer playbin pipeline."""
        if not self.video_path or not Path(self.video_path).exists():
            logger.warning(f"[Video] No video file found at: {self.video_path}")
            self._create_placeholder()
            return

        try:
            # Create playbin
            self.player = Gst.ElementFactory.make("playbin", "video-player")
            if not self.player:
                logger.error("[Video] Failed to create playbin")
                self._create_placeholder()
                return

            # Set video file
            self.player.set_property("uri", f"file://{self.video_path}")

            # Mute audio
            self.player.set_property("mute", True)

            # Create GTK4 paintable sink
            self.sink = Gst.ElementFactory.make("gtk4paintablesink", "video-sink")
            if not self.sink:
                # Fallback to autovideosink
                logger.warning("[Video] gtk4paintablesink not available, using fallback")
                self._create_placeholder()
                return

            self.player.set_property("video-sink", self.sink)

            # Create picture widget
            self.picture = Gtk.Picture()
            paintable = self.sink.get_property("paintable")
            if paintable:
                self.picture.set_paintable(paintable)

            # Apply initial opacity
            self.picture.set_opacity(self.opacity)

            # Add CSS class for styling
            self.picture.add_css_class("video-background")

            # Set up bus for messages
            bus = self.player.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self._on_bus_message)

            logger.info(f"[Video] Player initialized with: {self.video_path}")

        except Exception as e:
            logger.error(f"[Video] Error setting up player: {e}")
            self._create_placeholder()

    def _create_placeholder(self):
        """Create placeholder widget when video not available."""
        self.picture = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.picture.add_css_class("video-placeholder")
        self.picture.set_opacity(0)  # Invisible placeholder

    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages."""
        msg_type = message.type

        if msg_type == Gst.MessageType.EOS:
            # End of stream - loop video
            self._loop_video()

        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"[Video] GStreamer error: {err.message}")
            logger.debug(f"[Video] Debug: {debug}")
            self.stop()

        elif msg_type == Gst.MessageType.STATE_CHANGED:
            if message.src == self.player:
                old, new, pending = message.parse_state_changed()
                if new == Gst.State.PLAYING:
                    self.is_playing = True
                elif new == Gst.State.PAUSED:
                    self.is_playing = False

        return True

    def _loop_video(self):
        """Loop video back to start."""
        if self.player and not self.paused:
            # Seek to beginning
            self.player.seek_simple(
                Gst.Format.TIME,
                Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT,
                0
            )
            logger.debug("[Video] Looped back to start")

    def get_widget(self) -> Gtk.Widget:
        """
        Get the video widget to add to container.

        Returns:
            Gtk.Widget: Video picture widget
        """
        return self.picture

    def play(self):
        """Start video playback."""
        if self.player:
            self.player.set_state(Gst.State.PLAYING)
            self.paused = False
            logger.info("[Video] Playback started")

    def pause(self):
        """Pause video playback."""
        if self.player:
            self.player.set_state(Gst.State.PAUSED)
            self.paused = True
            logger.info("[Video] Playback paused")

    def stop(self):
        """Stop video playback."""
        if self.player:
            self.player.set_state(Gst.State.NULL)
            self.is_playing = False
            logger.info("[Video] Playback stopped")

    def set_opacity(self, opacity: float):
        """
        Set video opacity.

        Args:
            opacity: Opacity value (0.0 - 1.0)
        """
        self.opacity = max(0.0, min(1.0, opacity))
        if self.picture:
            self.picture.set_opacity(self.opacity)

    def fade_in(self, duration_ms: int = 1000):
        """Fade video in over duration."""
        if not self.picture:
            return

        steps = 20
        interval = duration_ms // steps
        current_opacity = 0.0
        step_size = self.opacity / steps

        def fade_step():
            nonlocal current_opacity
            current_opacity += step_size
            if current_opacity >= self.opacity:
                self.picture.set_opacity(self.opacity)
                return False
            self.picture.set_opacity(current_opacity)
            return True

        self.picture.set_opacity(0)
        GLib.timeout_add(interval, fade_step)

    def fade_out(self, duration_ms: int = 500):
        """Fade video out over duration."""
        if not self.picture:
            return

        steps = 20
        interval = duration_ms // steps
        current_opacity = self.opacity
        step_size = self.opacity / steps

        def fade_step():
            nonlocal current_opacity
            current_opacity -= step_size
            if current_opacity <= 0:
                self.picture.set_opacity(0)
                return False
            self.picture.set_opacity(current_opacity)
            return True

        GLib.timeout_add(interval, fade_step)

    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if self.player:
            self.player.set_state(Gst.State.NULL)
            self.player = None
        logger.info("[Video] Cleaned up")


class AnimatedBackground:
    """
    CSS-animated fallback background when video is not available.

    Creates animated gradient/particle effects using pure CSS.
    """

    def __init__(self):
        """Initialize animated background."""
        self.widget = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.widget.set_hexpand(True)
        self.widget.set_vexpand(True)
        self.widget.add_css_class("animated-background")

        # Add CSS for animations
        self._add_animation_css()

    def _add_animation_css(self):
        """Add CSS animations for background effects."""
        css = """
        .animated-background {
            background: linear-gradient(
                135deg,
                rgba(10, 14, 26, 0.95) 0%,
                rgba(20, 30, 60, 0.95) 50%,
                rgba(10, 14, 26, 0.95) 100%
            );
            background-size: 400% 400%;
            animation: gradient-shift 15s ease infinite;
        }

        @keyframes gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Particle-like dots overlay */
        .animated-background::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-image:
                radial-gradient(2px 2px at 20px 30px, rgba(0, 212, 255, 0.3), transparent),
                radial-gradient(2px 2px at 40px 70px, rgba(102, 126, 234, 0.3), transparent),
                radial-gradient(2px 2px at 90px 40px, rgba(0, 212, 255, 0.2), transparent),
                radial-gradient(2px 2px at 130px 80px, rgba(118, 75, 162, 0.3), transparent),
                radial-gradient(2px 2px at 160px 30px, rgba(0, 212, 255, 0.2), transparent);
            background-repeat: repeat;
            background-size: 200px 100px;
            animation: particle-drift 20s linear infinite;
            pointer-events: none;
        }

        @keyframes particle-drift {
            0% { transform: translateY(0); }
            100% { transform: translateY(-100px); }
        }
        """

        css_provider = Gtk.CssProvider()
        css_provider.load_from_string(css)

        display = self.widget.get_display()
        if display:
            Gtk.StyleContext.add_provider_for_display(
                display,
                css_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )

    def get_widget(self) -> Gtk.Widget:
        """Get the animated background widget."""
        return self.widget


def create_background(video_path: str = None) -> tuple:
    """
    Create appropriate background based on available resources.

    Args:
        video_path: Optional path to video file

    Returns:
        tuple: (widget, background_object)
    """
    # Try video first
    if video_path and Path(video_path).exists():
        bg = VideoBackground(video_path)
        if bg.player:
            return bg.get_widget(), bg

    # Check default video locations
    bg = VideoBackground()
    if bg.player:
        return bg.get_widget(), bg

    # Fall back to animated CSS background
    logger.info("[Video] Using animated CSS background")
    bg = AnimatedBackground()
    return bg.get_widget(), bg


# Example usage / testing
if __name__ == '__main__':
    import sys

    # Create test window
    app = Gtk.Application(application_id='com.quanta.tfan.video-test')

    def on_activate(app):
        window = Gtk.ApplicationWindow(application=app)
        window.set_default_size(800, 600)
        window.set_title("Video Background Test")

        # Create overlay for layering
        overlay = Gtk.Overlay()
        window.set_child(overlay)

        # Add video background
        video_path = sys.argv[1] if len(sys.argv) > 1 else None
        widget, bg = create_background(video_path)
        overlay.set_child(widget)

        # Add content on top
        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content.set_valign(Gtk.Align.CENTER)
        content.set_halign(Gtk.Align.CENTER)

        label = Gtk.Label(label="Video Background Test")
        label.add_css_class("title-1")
        content.append(label)

        # Controls
        controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        controls.set_halign(Gtk.Align.CENTER)
        controls.set_margin_top(20)

        if hasattr(bg, 'play'):
            play_btn = Gtk.Button(label="Play")
            play_btn.connect("clicked", lambda b: bg.play())
            controls.append(play_btn)

            pause_btn = Gtk.Button(label="Pause")
            pause_btn.connect("clicked", lambda b: bg.pause())
            controls.append(pause_btn)

            # Opacity slider
            opacity_scale = Gtk.Scale.new_with_range(
                Gtk.Orientation.HORIZONTAL, 0, 1, 0.05
            )
            opacity_scale.set_value(0.2)
            opacity_scale.set_size_request(150, -1)
            opacity_scale.connect("value-changed", lambda s: bg.set_opacity(s.get_value()))
            controls.append(opacity_scale)

        content.append(controls)
        overlay.add_overlay(content)

        # Start playback
        if hasattr(bg, 'play'):
            bg.play()
            bg.fade_in(1500)

        window.present()

        # Cleanup on close
        window.connect("close-request", lambda w: bg.cleanup() if hasattr(bg, 'cleanup') else None)

    app.connect('activate', on_activate)
    app.run(None)
