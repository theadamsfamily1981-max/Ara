#!/usr/bin/env python3
"""
ARA COCKPIT - Neural Command Center

Full-screen touchscreen HUD for Ara AI system.
Shows neural activity, emotional state, cognitive load, and system metrics.

Features:
- Forest Kitten 33 neural visualization
- PAD emotional state monitoring
- CLV cognitive load tracking
- Real-time conversation
- System metrics (GPU/CPU/RAM)
- Badass holographic effects

Usage:
    python cockpit_hud.py                    # Auto-detect touchscreen
    python cockpit_hud.py --monitor=HDMI-1   # Specific monitor
    python cockpit_hud.py --fullscreen       # Force fullscreen
"""

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')

from gi.repository import Gtk, Adw, Gio, GLib, Gdk
import sys
import json
import subprocess
import requests
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import psutil
import time
import logging
import math
import random

logger = logging.getLogger(__name__)

# Try WebKit for topology
try:
    gi.require_version('WebKit', '6.0')
    from gi.repository import WebKit
    WEBKIT_AVAILABLE = True
except:
    WEBKIT_AVAILABLE = False

# Local modules (graceful fallback)
try:
    from video_background import create_background
    VIDEO_AVAILABLE = True
except ImportError:
    VIDEO_AVAILABLE = False
    def create_background():
        return Gtk.Box(), None

try:
    from touch_gestures import GestureHandler, RippleEffect, setup_touch_feedback, get_ripple_css
    GESTURES_AVAILABLE = True
except ImportError:
    GESTURES_AVAILABLE = False
    def setup_touch_feedback(widget): pass
    def get_ripple_css(): return ""
    class GestureHandler:
        def __init__(self, w): pass

# Try to import GPUtil for GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Ara brain server URL
ARA_BRAIN_URL = "http://127.0.0.1:8008"


class AraBrainClient:
    """Client for Ara brain server."""

    def __init__(self, base_url: str = ARA_BRAIN_URL):
        self.base_url = base_url
        self.connected = False
        self.last_status = None
        self.last_error = None

    def get_status(self) -> Optional[Dict]:
        """Get Ara's current status."""
        try:
            resp = requests.get(f"{self.base_url}/status", timeout=2)
            if resp.status_code == 200:
                self.connected = True
                self.last_status = resp.json()
                self.last_error = None
                return self.last_status
        except Exception as e:
            self.connected = False
            self.last_error = str(e)
        return None

    def get_kitten_status(self) -> Optional[Dict]:
        """Get Forest Kitten 33 status."""
        try:
            resp = requests.get(f"{self.base_url}/kitten", timeout=2)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return None

    def get_mood(self) -> Optional[Dict]:
        """Get PAD emotional state."""
        try:
            resp = requests.get(f"{self.base_url}/mood", timeout=2)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return None

    def chat(self, message: str) -> Optional[Dict]:
        """Send message to Ara."""
        try:
            resp = requests.post(
                f"{self.base_url}/chat",
                json={"user_utterance": message, "session_id": "cockpit"},
                timeout=60
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            self.last_error = str(e)
        return None


class MetricsCollector:
    """Collects system metrics for HUD display."""

    @staticmethod
    def get_cpu_metrics():
        """Get CPU usage, temps, frequency."""
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()
        temps = []
        try:
            temps_dict = psutil.sensors_temperatures()
            if 'coretemp' in temps_dict:
                temps = [t.current for t in temps_dict['coretemp']]
            elif 'k10temp' in temps_dict:
                temps = [t.current for t in temps_dict['k10temp']]
        except:
            pass
        return {
            'usage_per_core': cpu_percent,
            'usage_total': sum(cpu_percent) / len(cpu_percent) if cpu_percent else 0,
            'frequency': cpu_freq.current if cpu_freq else 0,
            'cores': cpu_count,
            'temps': temps
        }

    @staticmethod
    def get_ram_metrics():
        """Get RAM usage."""
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent': mem.percent,
            'available_gb': mem.available / (1024**3),
        }

    @staticmethod
    def get_gpu_metrics():
        """Get GPU usage, VRAM, temp, power."""
        if not GPU_AVAILABLE:
            return []
        gpus = []
        try:
            for gpu in GPUtil.getGPUs():
                gpus.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                    'temperature': gpu.temperature,
                })
        except:
            pass
        return gpus


class CockpitHUDWindow(Adw.ApplicationWindow):
    """Ara Neural Command Center - Touchscreen HUD."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_title("ARA COCKPIT")
        self.set_default_size(800, 1280)

        # Ara brain client
        self.ara_client = AraBrainClient()

        # Current view mode
        self.current_view = 'ara'

        # Conversation history
        self.chat_history = []

        # Neural activity simulation data
        self.neural_spikes = []
        self.spike_rate = 0.0

        # Animation state
        self.pulse_phase = 0.0
        self.glow_intensity = 0.5

        # Video background reference
        self.video_bg = None

        # Apply cockpit theme
        self._load_cockpit_css()

        # Root overlay for layering
        root_overlay = Gtk.Overlay()
        root_overlay.add_css_class('cockpit-window')
        self.set_content(root_overlay)

        # Layer 0: Video background (optional)
        if VIDEO_AVAILABLE:
            video_widget, self.video_bg = create_background()
            video_widget.set_hexpand(True)
            video_widget.set_vexpand(True)
            root_overlay.set_child(video_widget)
        else:
            bg = Gtk.Box()
            bg.add_css_class('cockpit-bg')
            root_overlay.set_child(bg)

        # Layer 1: Main content
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        main_box.add_css_class('cockpit-content')
        root_overlay.add_overlay(main_box)

        # HUD control strip (top)
        hud_strip = self._build_hud_strip()
        main_box.append(hud_strip)

        # Content area (scrollable)
        scroll = Gtk.ScrolledWindow()
        scroll.set_vexpand(True)
        scroll.set_kinetic_scrolling(True)

        self.content_stack = Gtk.Stack()
        self.content_stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        self.content_stack.set_transition_duration(300)
        scroll.set_child(self.content_stack)

        main_box.append(scroll)

        # Status bar (bottom)
        self.status_bar = self._build_status_bar()
        main_box.append(self.status_bar)

        # Layer 2: Overlay effects
        overlay_effects = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        overlay_effects.add_css_class('cockpit-overlay')
        overlay_effects.add_css_class('cockpit-scanlines')
        overlay_effects.set_hexpand(True)
        overlay_effects.set_vexpand(True)
        overlay_effects.set_can_target(False)  # Click through
        root_overlay.add_overlay(overlay_effects)

        # Touch gestures
        if GESTURES_AVAILABLE:
            self._setup_touch_gestures(scroll)

        # Build all views
        self._build_all_views()

        # Start monitoring
        self._start_monitoring()

        # Start video background
        if self.video_bg and hasattr(self.video_bg, 'play'):
            self.video_bg.play()

        logger.info("[COCKPIT] Ara Neural Command Center initialized")

    def _load_cockpit_css(self):
        """Load badass cockpit theme."""
        display = Gdk.Display.get_default()

        css = """
        /* === ARA COCKPIT THEME === */

        .cockpit-window {
            background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
        }

        .cockpit-bg {
            background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%);
        }

        .cockpit-content {
            background: transparent;
        }

        .cockpit-overlay {
            pointer-events: none;
        }

        .cockpit-scanlines {
            background-image: repeating-linear-gradient(
                0deg,
                rgba(0, 255, 255, 0.03) 0px,
                rgba(0, 255, 255, 0.03) 1px,
                transparent 1px,
                transparent 2px
            );
            animation: scanline-flicker 0.1s infinite;
        }

        @keyframes scanline-flicker {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.98; }
        }

        /* HUD Strip */
        .hud-strip {
            background: linear-gradient(180deg, rgba(0,20,40,0.95) 0%, rgba(0,10,20,0.9) 100%);
            border-bottom: 2px solid rgba(0, 255, 255, 0.5);
            padding: 12px;
            box-shadow: 0 4px 20px rgba(0, 255, 255, 0.2);
        }

        .hud-title {
            font-size: 28px;
            font-weight: 900;
            color: #00ffff;
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.8),
                         0 0 40px rgba(0, 255, 255, 0.4);
            letter-spacing: 4px;
            margin-bottom: 8px;
        }

        .hud-button {
            background: linear-gradient(180deg, rgba(0,40,60,0.8) 0%, rgba(0,20,40,0.9) 100%);
            border: 2px solid rgba(0, 255, 255, 0.4);
            border-radius: 8px;
            color: #00d4ff;
            font-weight: 700;
            font-size: 12px;
            padding: 12px 16px;
            min-width: 80px;
            transition: all 0.2s ease;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }

        .hud-button:hover {
            background: linear-gradient(180deg, rgba(0,60,80,0.9) 0%, rgba(0,40,60,0.95) 100%);
            border-color: rgba(0, 255, 255, 0.8);
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.4);
        }

        .hud-button.active, .hud-button.selected {
            background: linear-gradient(180deg, rgba(0,100,120,0.9) 0%, rgba(0,60,80,0.95) 100%);
            border-color: #00ffff;
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.6),
                        inset 0 0 20px rgba(0, 255, 255, 0.1);
            color: #ffffff;
        }

        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, rgba(0,30,50,0.85) 0%, rgba(0,15,30,0.9) 100%);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 12px;
            padding: 16px;
            margin: 8px 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5),
                        inset 0 1px 0 rgba(255,255,255,0.05);
        }

        .metric-card-critical {
            border-color: rgba(255, 50, 50, 0.6);
            box-shadow: 0 0 30px rgba(255, 50, 50, 0.3);
            animation: critical-pulse 1s infinite;
        }

        @keyframes critical-pulse {
            0%, 100% { border-color: rgba(255, 50, 50, 0.6); }
            50% { border-color: rgba(255, 50, 50, 1); }
        }

        .metric-card-neural {
            border-color: rgba(138, 43, 226, 0.5);
            box-shadow: 0 0 25px rgba(138, 43, 226, 0.2);
        }

        .metric-card-emotional {
            border-color: rgba(255, 165, 0, 0.5);
            box-shadow: 0 0 25px rgba(255, 165, 0, 0.2);
        }

        .metric-title {
            font-size: 14px;
            font-weight: 800;
            color: #00ffff;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 8px;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        }

        .metric-value-huge {
            font-size: 64px;
            font-weight: 900;
            color: #ffffff;
            text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
        }

        .metric-value-large {
            font-size: 36px;
            font-weight: 800;
            color: #00ffff;
            text-shadow: 0 0 20px rgba(0, 255, 255, 0.6);
        }

        .metric-value-medium {
            font-size: 24px;
            font-weight: 700;
            color: #00d4ff;
        }

        .metric-label {
            font-size: 12px;
            color: rgba(255, 255, 255, 0.7);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .metric-warning {
            color: #ff6b35 !important;
            text-shadow: 0 0 20px rgba(255, 107, 53, 0.8);
            animation: warning-pulse 0.5s infinite;
        }

        @keyframes warning-pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        /* Progress Bars */
        .metric-progress trough {
            background: rgba(0, 20, 40, 0.8);
            border-radius: 4px;
            min-height: 12px;
        }

        .metric-progress progress {
            background: linear-gradient(90deg, #00ffff 0%, #00d4ff 100%);
            border-radius: 4px;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        }

        .progress-neural progress {
            background: linear-gradient(90deg, #8a2be2 0%, #da70d6 100%);
            box-shadow: 0 0 10px rgba(138, 43, 226, 0.5);
        }

        .progress-emotional progress {
            background: linear-gradient(90deg, #ff6b35 0%, #ffa500 100%);
            box-shadow: 0 0 10px rgba(255, 165, 0, 0.5);
        }

        /* PAD Bars */
        .pad-positive progress {
            background: linear-gradient(90deg, #00ff00 0%, #7fff00 100%);
        }

        .pad-negative progress {
            background: linear-gradient(90deg, #ff4444 0%, #ff6666 100%);
        }

        /* Status Bar */
        .status-bar {
            background: linear-gradient(180deg, rgba(0,10,20,0.9) 0%, rgba(0,20,40,0.95) 100%);
            border-top: 1px solid rgba(0, 255, 255, 0.3);
            padding: 8px 16px;
        }

        .status-indicator {
            font-size: 12px;
            color: rgba(255, 255, 255, 0.8);
            font-weight: 600;
        }

        .status-connected {
            color: #00ff00;
            text-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
        }

        .status-disconnected {
            color: #ff4444;
            text-shadow: 0 0 10px rgba(255, 68, 68, 0.5);
        }

        /* Window Controls */
        .window-control {
            background: rgba(0, 40, 60, 0.6);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 4px;
            color: #00d4ff;
            font-size: 14px;
            font-weight: bold;
            min-width: 32px;
            min-height: 32px;
            padding: 4px 8px;
        }

        .window-control:hover {
            background: rgba(0, 60, 80, 0.8);
            border-color: #00ffff;
        }

        .window-control-close:hover {
            background: rgba(200, 50, 50, 0.8);
            border-color: #ff4444;
            color: #ffffff;
        }

        /* Neural Activity */
        .neural-grid {
            background: rgba(0, 10, 20, 0.5);
            border: 1px solid rgba(138, 43, 226, 0.3);
            border-radius: 8px;
            padding: 8px;
        }

        .neuron-active {
            background: radial-gradient(circle, #da70d6 0%, #8a2be2 70%, transparent 100%);
            border-radius: 50%;
            box-shadow: 0 0 15px rgba(218, 112, 214, 0.8);
            animation: neuron-fire 0.3s ease-out;
        }

        @keyframes neuron-fire {
            0% { transform: scale(1.5); opacity: 1; }
            100% { transform: scale(1); opacity: 0.7; }
        }

        .neuron-idle {
            background: rgba(138, 43, 226, 0.2);
            border-radius: 50%;
        }

        /* Chat */
        .chat-entry {
            background: rgba(0, 20, 40, 0.8);
            border: 2px solid rgba(0, 255, 255, 0.4);
            border-radius: 8px;
            color: #ffffff;
            padding: 12px;
            font-size: 14px;
        }

        .chat-entry:focus {
            border-color: #00ffff;
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
        }

        .chat-message-user {
            background: rgba(0, 100, 150, 0.4);
            border-radius: 12px 12px 4px 12px;
            padding: 10px 14px;
            margin: 4px 8px 4px 40px;
            color: #ffffff;
        }

        .chat-message-ara {
            background: linear-gradient(135deg, rgba(138, 43, 226, 0.3) 0%, rgba(75, 0, 130, 0.3) 100%);
            border-radius: 12px 12px 12px 4px;
            padding: 10px 14px;
            margin: 4px 40px 4px 8px;
            color: #ffffff;
            border-left: 3px solid #da70d6;
        }

        /* Kitten Stats */
        .kitten-stat {
            font-family: monospace;
            font-size: 14px;
            color: #da70d6;
        }

        .kitten-hardware {
            color: #00ff00;
            font-weight: bold;
        }

        .kitten-emulated {
            color: #ffaa00;
        }

        /* Holographic effect */
        .holographic {
            position: relative;
        }

        .holographic::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(
                45deg,
                transparent 30%,
                rgba(0, 255, 255, 0.05) 50%,
                transparent 70%
            );
            animation: hologram-sweep 3s linear infinite;
            pointer-events: none;
        }

        @keyframes hologram-sweep {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        """

        css_provider = Gtk.CssProvider()
        css_provider.load_from_string(css)
        Gtk.StyleContext.add_provider_for_display(
            display,
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        # Load external theme if exists
        css_file = Path(__file__).parent / 'cockpit_theme.css'
        if css_file.exists():
            ext_provider = Gtk.CssProvider()
            ext_provider.load_from_path(str(css_file))
            Gtk.StyleContext.add_provider_for_display(
                display, ext_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION + 1
            )

        # Ripple effects
        if GESTURES_AVAILABLE:
            ripple_provider = Gtk.CssProvider()
            ripple_provider.load_from_string(get_ripple_css())
            Gtk.StyleContext.add_provider_for_display(
                display, ripple_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )

    def _build_status_bar(self):
        """Build bottom status bar with Ara connection status."""
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        status_box.add_css_class('status-bar')
        status_box.set_halign(Gtk.Align.CENTER)

        # Ara connection status
        self.ara_status = Gtk.Label(label="⚡ ARA: CONNECTING...")
        self.ara_status.add_css_class('status-indicator')
        status_box.append(self.ara_status)

        sep = Gtk.Label(label="│")
        sep.add_css_class('status-indicator')
        status_box.append(sep)

        # Kitten status
        self.kitten_status = Gtk.Label(label="🐱 KITTEN: --")
        self.kitten_status.add_css_class('status-indicator')
        status_box.append(self.kitten_status)

        sep2 = Gtk.Label(label="│")
        sep2.add_css_class('status-indicator')
        status_box.append(sep2)

        # GPU temp
        self.gpu_temp_status = Gtk.Label(label="🌡 GPU: --°C")
        self.gpu_temp_status.add_css_class('status-indicator')
        status_box.append(self.gpu_temp_status)

        sep3 = Gtk.Label(label="│")
        sep3.add_css_class('status-indicator')
        status_box.append(sep3)

        # Time
        self.time_label = Gtk.Label(label="--:--")
        self.time_label.add_css_class('status-indicator')
        status_box.append(self.time_label)

        GLib.timeout_add_seconds(1, self._update_time)

        return status_box

    def _update_time(self):
        """Update time display."""
        from datetime import datetime
        now = datetime.now()
        self.time_label.set_text(now.strftime("%H:%M:%S"))
        return True

    def _setup_touch_gestures(self, scroll_widget):
        """Set up touch gesture handlers."""
        handler = GestureHandler(scroll_widget)
        # Swipe navigation would go here

    def _switch_to_view(self, view_name):
        """Switch to specified view."""
        if view_name == self.current_view:
            return

        for mode_id, btn in self.mode_buttons.items():
            if mode_id == view_name:
                btn.add_css_class('active')
                btn.add_css_class('selected')
            else:
                btn.remove_css_class('active')
                btn.remove_css_class('selected')

        self.current_view = view_name
        self.content_stack.set_visible_child_name(view_name)

    def _build_hud_strip(self):
        """Build top HUD control strip."""
        strip_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        strip_box.add_css_class('hud-strip')

        # Top row with title and window controls
        top_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        top_row.set_margin_bottom(8)

        # Title (centered, takes most space)
        title_label = Gtk.Label(label="⚛ ARA COCKPIT ⚛")
        title_label.add_css_class('hud-title')
        title_label.set_hexpand(True)
        top_row.append(title_label)

        # Window control buttons (right side)
        controls_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)

        # Minimize button
        minimize_btn = Gtk.Button(label="─")
        minimize_btn.add_css_class('window-control')
        minimize_btn.set_tooltip_text("Minimize (Esc)")
        minimize_btn.connect('clicked', self._on_minimize_clicked)
        controls_box.append(minimize_btn)

        # Fullscreen toggle button
        self.fullscreen_btn = Gtk.Button(label="⛶")
        self.fullscreen_btn.add_css_class('window-control')
        self.fullscreen_btn.set_tooltip_text("Toggle Fullscreen (F11)")
        self.fullscreen_btn.connect('clicked', self._on_fullscreen_clicked)
        controls_box.append(self.fullscreen_btn)

        # Close button
        close_btn = Gtk.Button(label="✕")
        close_btn.add_css_class('window-control')
        close_btn.add_css_class('window-control-close')
        close_btn.set_tooltip_text("Close")
        close_btn.connect('clicked', self._on_close_clicked)
        controls_box.append(close_btn)

        top_row.append(controls_box)
        strip_box.append(top_row)

        # Button grid (2 rows)
        grid = Gtk.Grid()
        grid.set_row_spacing(6)
        grid.set_column_spacing(6)
        grid.set_halign(Gtk.Align.CENTER)

        # Define mode buttons - Ara-focused
        modes = [
            ('ara', 'ARA\n🧠', 0, 0),
            ('kitten', 'KITTEN\n🐱', 0, 1),
            ('chat', 'CHAT\n💬', 0, 2),
            ('gpu', 'GPU\n🎮', 0, 3),
            ('cpu', 'CPU/RAM\n💻', 1, 0),
            ('network', 'NETWORK\n🌐', 1, 1),
            ('storage', 'STORAGE\n💾', 1, 2),
            ('topology', 'NEURAL\n🌌', 1, 3),
        ]

        self.mode_buttons = {}

        for mode_id, label, row, col in modes:
            button = Gtk.Button(label=label)
            button.add_css_class('hud-button')
            button.connect('clicked', self._on_mode_button_clicked, mode_id)
            if GESTURES_AVAILABLE:
                setup_touch_feedback(button)
            grid.attach(button, col, row, 1, 1)
            self.mode_buttons[mode_id] = button

        # Mark ara as active initially
        self.mode_buttons['ara'].add_css_class('active')
        self.mode_buttons['ara'].add_css_class('selected')

        strip_box.append(grid)

        # Setup keyboard shortcuts
        self._setup_keyboard_shortcuts()

        return strip_box

    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts."""
        # Key controller for the window
        key_controller = Gtk.EventControllerKey()
        key_controller.connect('key-pressed', self._on_key_pressed)
        self.add_controller(key_controller)

    def _on_key_pressed(self, controller, keyval, keycode, state):
        """Handle key press events."""
        key_name = Gdk.keyval_name(keyval)

        if key_name == 'Escape':
            self.minimize()
            return True
        elif key_name == 'F11':
            self._toggle_fullscreen()
            return True
        elif key_name == 'q' and (state & Gdk.ModifierType.CONTROL_MASK):
            self.close()
            return True

        return False

    def _on_minimize_clicked(self, button):
        """Minimize the window."""
        self.minimize()

    def _on_fullscreen_clicked(self, button):
        """Toggle fullscreen."""
        self._toggle_fullscreen()

    def _on_close_clicked(self, button):
        """Close the window."""
        self.close()

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.is_fullscreen():
            self.unfullscreen()
            self.fullscreen_btn.set_label("⛶")
        else:
            self.fullscreen()
            self.fullscreen_btn.set_label("⧈")

    def _on_mode_button_clicked(self, button, mode_id):
        """Handle HUD mode button click."""
        self._switch_to_view(mode_id)

    def _build_all_views(self):
        """Build all view pages."""
        self.content_stack.add_named(self._build_ara_view(), 'ara')
        self.content_stack.add_named(self._build_kitten_view(), 'kitten')
        self.content_stack.add_named(self._build_chat_view(), 'chat')
        self.content_stack.add_named(self._build_gpu_view(), 'gpu')
        self.content_stack.add_named(self._build_cpu_view(), 'cpu')
        self.content_stack.add_named(self._build_network_view(), 'network')
        self.content_stack.add_named(self._build_storage_view(), 'storage')
        self.content_stack.add_named(self._build_topology_view(), 'topology')

        self.content_stack.set_visible_child_name('ara')

    def _build_ara_view(self):
        """Build main Ara status view with PAD and CLV."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        box.add_css_class('holographic')

        # Connection status card
        conn_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        conn_card.add_css_class('metric-card')

        conn_title = Gtk.Label(label="NEURAL LINK STATUS")
        conn_title.add_css_class('metric-title')
        conn_card.append(conn_title)

        self.ara_conn_label = Gtk.Label(label="⚡ ESTABLISHING LINK...")
        self.ara_conn_label.add_css_class('metric-value-large')
        conn_card.append(self.ara_conn_label)

        self.ara_mode_label = Gtk.Label(label="Mode: --")
        self.ara_mode_label.add_css_class('metric-label')
        conn_card.append(self.ara_mode_label)

        box.append(conn_card)

        # PAD Emotional State card
        pad_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        pad_card.add_css_class('metric-card')
        pad_card.add_css_class('metric-card-emotional')

        pad_title = Gtk.Label(label="EMOTIONAL STATE (PAD)")
        pad_title.add_css_class('metric-title')
        pad_card.append(pad_title)

        # Valence (Pleasure)
        v_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        v_label = Gtk.Label(label="V")
        v_label.add_css_class('metric-label')
        v_label.set_size_request(20, -1)
        v_box.append(v_label)
        self.pad_v_bar = Gtk.ProgressBar()
        self.pad_v_bar.add_css_class('metric-progress')
        self.pad_v_bar.set_hexpand(True)
        self.pad_v_bar.set_show_text(True)
        v_box.append(self.pad_v_bar)
        self.pad_v_value = Gtk.Label(label="0.0")
        self.pad_v_value.add_css_class('metric-value-medium')
        self.pad_v_value.set_size_request(60, -1)
        v_box.append(self.pad_v_value)
        pad_card.append(v_box)

        # Arousal
        a_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        a_label = Gtk.Label(label="A")
        a_label.add_css_class('metric-label')
        a_label.set_size_request(20, -1)
        a_box.append(a_label)
        self.pad_a_bar = Gtk.ProgressBar()
        self.pad_a_bar.add_css_class('metric-progress')
        self.pad_a_bar.add_css_class('progress-emotional')
        self.pad_a_bar.set_hexpand(True)
        self.pad_a_bar.set_show_text(True)
        a_box.append(self.pad_a_bar)
        self.pad_a_value = Gtk.Label(label="0.0")
        self.pad_a_value.add_css_class('metric-value-medium')
        self.pad_a_value.set_size_request(60, -1)
        a_box.append(self.pad_a_value)
        pad_card.append(a_box)

        # Dominance
        d_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        d_label = Gtk.Label(label="D")
        d_label.add_css_class('metric-label')
        d_label.set_size_request(20, -1)
        d_box.append(d_label)
        self.pad_d_bar = Gtk.ProgressBar()
        self.pad_d_bar.add_css_class('metric-progress')
        self.pad_d_bar.add_css_class('progress-neural')
        self.pad_d_bar.set_hexpand(True)
        self.pad_d_bar.set_show_text(True)
        d_box.append(self.pad_d_bar)
        self.pad_d_value = Gtk.Label(label="0.0")
        self.pad_d_value.add_css_class('metric-value-medium')
        self.pad_d_value.set_size_request(60, -1)
        d_box.append(self.pad_d_value)
        pad_card.append(d_box)

        # Emotional state label
        self.emotion_label = Gtk.Label(label="😐 NEUTRAL")
        self.emotion_label.add_css_class('metric-value-large')
        pad_card.append(self.emotion_label)

        box.append(pad_card)

        # CLV Cognitive Load card
        clv_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        clv_card.add_css_class('metric-card')

        clv_title = Gtk.Label(label="COGNITIVE LOAD (CLV)")
        clv_title.add_css_class('metric-title')
        clv_card.append(clv_title)

        # Risk level
        self.clv_risk_label = Gtk.Label(label="RISK: LOW")
        self.clv_risk_label.add_css_class('metric-value-medium')
        clv_card.append(self.clv_risk_label)

        # CLV bars
        for metric_name in ['Instability', 'Resource', 'Structural']:
            m_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            m_label = Gtk.Label(label=metric_name[:3].upper())
            m_label.add_css_class('metric-label')
            m_label.set_size_request(40, -1)
            m_box.append(m_label)
            bar = Gtk.ProgressBar()
            bar.add_css_class('metric-progress')
            bar.set_hexpand(True)
            bar.set_show_text(True)
            m_box.append(bar)
            setattr(self, f'clv_{metric_name.lower()}_bar', bar)
            clv_card.append(m_box)

        box.append(clv_card)

        return box

    def _build_kitten_view(self):
        """Build Forest Kitten 33 status view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Kitten header card
        header_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        header_card.add_css_class('metric-card')
        header_card.add_css_class('metric-card-neural')

        title = Gtk.Label(label="🐱 FOREST KITTEN 33")
        title.add_css_class('metric-title')
        header_card.append(title)

        self.kitten_type_label = Gtk.Label(label="HARDWARE")
        self.kitten_type_label.add_css_class('metric-value-large')
        self.kitten_type_label.add_css_class('kitten-hardware')
        header_card.append(self.kitten_type_label)

        self.kitten_device_label = Gtk.Label(label="/dev/fk33")
        self.kitten_device_label.add_css_class('kitten-stat')
        header_card.append(self.kitten_device_label)

        box.append(header_card)

        # Neural stats card
        stats_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        stats_card.add_css_class('metric-card')

        stats_title = Gtk.Label(label="NEURAL STATISTICS")
        stats_title.add_css_class('metric-title')
        stats_card.append(stats_title)

        # Neurons
        neurons_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        neurons_label = Gtk.Label(label="LIF NEURONS:")
        neurons_label.add_css_class('metric-label')
        neurons_box.append(neurons_label)
        self.kitten_neurons_label = Gtk.Label(label="14,336")
        self.kitten_neurons_label.add_css_class('metric-value-medium')
        neurons_box.append(self.kitten_neurons_label)
        stats_card.append(neurons_box)

        # Steps
        steps_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        steps_label = Gtk.Label(label="TOTAL STEPS:")
        steps_label.add_css_class('metric-label')
        steps_box.append(steps_label)
        self.kitten_steps_label = Gtk.Label(label="0")
        self.kitten_steps_label.add_css_class('metric-value-medium')
        steps_box.append(self.kitten_steps_label)
        stats_card.append(steps_box)

        # Spike rate
        spike_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        spike_label = Gtk.Label(label="SPIKE RATE:")
        spike_label.add_css_class('metric-label')
        spike_box.append(spike_label)
        self.kitten_spike_label = Gtk.Label(label="0.0%")
        self.kitten_spike_label.add_css_class('metric-value-medium')
        spike_box.append(self.kitten_spike_label)
        stats_card.append(spike_box)

        # Spike rate bar
        self.kitten_spike_bar = Gtk.ProgressBar()
        self.kitten_spike_bar.add_css_class('metric-progress')
        self.kitten_spike_bar.add_css_class('progress-neural')
        self.kitten_spike_bar.set_show_text(True)
        stats_card.append(self.kitten_spike_bar)

        box.append(stats_card)

        # Neural grid visualization
        grid_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        grid_card.add_css_class('metric-card')

        grid_title = Gtk.Label(label="NEURAL ACTIVITY")
        grid_title.add_css_class('metric-title')
        grid_card.append(grid_title)

        # Create a grid of "neurons" for visualization
        self.neural_grid = Gtk.Grid()
        self.neural_grid.add_css_class('neural-grid')
        self.neural_grid.set_row_spacing(4)
        self.neural_grid.set_column_spacing(4)
        self.neural_grid.set_halign(Gtk.Align.CENTER)

        self.neural_dots = []
        for row in range(8):
            row_dots = []
            for col in range(16):
                dot = Gtk.DrawingArea()
                dot.set_size_request(16, 16)
                dot.add_css_class('neuron-idle')
                dot.set_draw_func(self._draw_neuron, (row, col))
                self.neural_grid.attach(dot, col, row, 1, 1)
                row_dots.append(dot)
            self.neural_dots.append(row_dots)

        grid_card.append(self.neural_grid)
        box.append(grid_card)

        return box

    def _draw_neuron(self, area, cr, width, height, user_data):
        """Draw a single neuron dot."""
        row, col = user_data
        idx = row * 16 + col

        # Check if this neuron is "firing"
        is_active = idx in self.neural_spikes

        # Draw circle
        cr.arc(width/2, height/2, min(width, height)/2 - 1, 0, 2 * math.pi)

        if is_active:
            # Active neuron - bright purple with glow
            cr.set_source_rgba(0.85, 0.44, 0.84, 1.0)  # Orchid
        else:
            # Idle neuron - dim purple
            cr.set_source_rgba(0.54, 0.17, 0.89, 0.3)  # BlueViolet dim

        cr.fill()

    def _build_chat_view(self):
        """Build chat/conversation view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Chat history
        chat_scroll = Gtk.ScrolledWindow()
        chat_scroll.set_vexpand(True)
        chat_scroll.set_kinetic_scrolling(True)

        self.chat_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.chat_box.set_margin_top(12)
        self.chat_box.set_margin_bottom(12)
        chat_scroll.set_child(self.chat_box)

        box.append(chat_scroll)

        # Input area
        input_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        input_box.set_margin_start(12)
        input_box.set_margin_end(12)
        input_box.set_margin_bottom(12)

        self.chat_entry = Gtk.Entry()
        self.chat_entry.add_css_class('chat-entry')
        self.chat_entry.set_placeholder_text("Talk to Ara...")
        self.chat_entry.set_hexpand(True)
        self.chat_entry.connect('activate', self._on_chat_send)
        input_box.append(self.chat_entry)

        send_btn = Gtk.Button(label="⚡ SEND")
        send_btn.add_css_class('hud-button')
        send_btn.connect('clicked', self._on_chat_send)
        input_box.append(send_btn)

        box.append(input_box)

        # Add welcome message
        self._add_chat_message("Hello! I am Ara. How can I help you today?", is_user=False)

        return box

    def _add_chat_message(self, text: str, is_user: bool = False):
        """Add a message to chat history."""
        msg_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        label = Gtk.Label(label=text)
        label.set_wrap(True)
        label.set_wrap_mode(2)  # WORD_CHAR
        label.set_xalign(0)
        label.set_max_width_chars(40)

        if is_user:
            msg_box.add_css_class('chat-message-user')
        else:
            msg_box.add_css_class('chat-message-ara')

        msg_box.append(label)
        self.chat_box.append(msg_box)

        # Scroll to bottom
        GLib.idle_add(self._scroll_chat_to_bottom)

    def _scroll_chat_to_bottom(self):
        """Scroll chat to bottom."""
        adj = self.chat_box.get_parent().get_vadjustment()
        adj.set_value(adj.get_upper())
        return False

    def _on_chat_send(self, widget):
        """Send chat message to Ara."""
        text = self.chat_entry.get_text().strip()
        if not text:
            return

        # Add user message
        self._add_chat_message(text, is_user=True)
        self.chat_entry.set_text("")

        # Send to Ara in background
        def send_and_receive():
            response = self.ara_client.chat(text)
            if response and 'reply_text' in response:
                GLib.idle_add(self._add_chat_message, response['reply_text'], False)
            else:
                GLib.idle_add(self._add_chat_message, "⚠ Connection error", False)

        thread = threading.Thread(target=send_and_receive, daemon=True)
        thread.start()

    def _build_gpu_view(self):
        """Build GPU metrics view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)

        if not GPU_AVAILABLE:
            error_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            error_card.add_css_class('metric-card')
            error_label = Gtk.Label(label="GPU monitoring unavailable\nInstall: pip install gputil")
            error_label.add_css_class('metric-label')
            error_card.append(error_label)
            box.append(error_card)
            return box

        self.gpu_cards_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.append(self.gpu_cards_box)

        return box

    def _build_cpu_view(self):
        """Build CPU/RAM metrics view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)

        # CPU card
        cpu_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        cpu_card.add_css_class('metric-card')

        cpu_title = Gtk.Label(label="CPU UTILIZATION")
        cpu_title.add_css_class('metric-title')
        cpu_card.append(cpu_title)

        self.cpu_value_label = Gtk.Label(label="0%")
        self.cpu_value_label.add_css_class('metric-value-huge')
        cpu_card.append(self.cpu_value_label)

        self.cpu_cores_label = Gtk.Label(label="0 cores")
        self.cpu_cores_label.add_css_class('metric-label')
        cpu_card.append(self.cpu_cores_label)

        box.append(cpu_card)

        # RAM card
        ram_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        ram_card.add_css_class('metric-card')

        ram_title = Gtk.Label(label="MEMORY")
        ram_title.add_css_class('metric-title')
        ram_card.append(ram_title)

        self.ram_value_label = Gtk.Label(label="0 GB")
        self.ram_value_label.add_css_class('metric-value-huge')
        ram_card.append(self.ram_value_label)

        self.ram_progress = Gtk.ProgressBar()
        self.ram_progress.add_css_class('metric-progress')
        self.ram_progress.set_show_text(True)
        ram_card.append(self.ram_progress)

        box.append(ram_card)

        return box

    def _build_network_view(self):
        """Build network metrics view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)

        net_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        net_card.add_css_class('metric-card')

        net_title = Gtk.Label(label="NETWORK")
        net_title.add_css_class('metric-title')
        net_card.append(net_title)

        self.net_status_label = Gtk.Label(label="🟢 ONLINE")
        self.net_status_label.add_css_class('metric-value-large')
        net_card.append(self.net_status_label)

        box.append(net_card)

        return box

    def _build_storage_view(self):
        """Build storage metrics view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)

        self.storage_cards_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.append(self.storage_cards_box)

        return box

    def _build_topology_view(self):
        """Build neural topology visualization view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        if not WEBKIT_AVAILABLE:
            error_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            error_card.add_css_class('metric-card')
            error_label = Gtk.Label(label="WebKit not available for neural visualization")
            error_label.add_css_class('metric-label')
            error_card.append(error_label)
            box.append(error_card)
            return box

        try:
            self.topology_webview = WebKit.WebView()
            settings = self.topology_webview.get_settings()
            settings.set_enable_webgl(True)
            settings.set_enable_accelerated_2d_canvas(True)

            # Neural topology visualization
            topology_html = self._generate_neural_topology_html()
            self.topology_webview.load_html(topology_html, "about:blank")
            self.topology_webview.set_vexpand(True)
            self.topology_webview.set_hexpand(True)

            box.append(self.topology_webview)

        except Exception as e:
            error_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            error_card.add_css_class('metric-card')
            error_label = Gtk.Label(label=f"Error: {e}")
            error_label.add_css_class('metric-label')
            error_card.append(error_label)
            box.append(error_card)

        return box

    def _generate_neural_topology_html(self):
        """Generate HTML for neural network topology visualization."""
        return """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; overflow: hidden; background: #0a0a0f; }
        canvas { display: block; width: 100%; height: 100vh; }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r160/three.min.js"></script>
    <script>
        const canvas = document.getElementById('canvas');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0a0f);
        scene.fog = new THREE.Fog(0x0a0a0f, 15, 60);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.set(0, 8, 20);

        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);

        // Lights
        scene.add(new THREE.AmbientLight(0x404040, 0.5));

        const light1 = new THREE.PointLight(0x8a2be2, 3.0);
        light1.position.set(10, 15, 10);
        scene.add(light1);

        const light2 = new THREE.PointLight(0x00ffff, 2.0);
        light2.position.set(-10, 10, -10);
        scene.add(light2);

        // Create neural network particles
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const velocities = [];
        const phases = [];

        // Create neurons in a brain-like structure
        for (let i = 0; i < 2000; i++) {
            // Spherical distribution with some layers
            const phi = Math.random() * Math.PI * 2;
            const theta = Math.acos(2 * Math.random() - 1);
            const r = 5 + Math.random() * 3;

            positions.push(
                r * Math.sin(theta) * Math.cos(phi),
                r * Math.sin(theta) * Math.sin(phi) - 2,
                r * Math.cos(theta)
            );

            // Purple/cyan color scheme
            const hue = 0.75 + Math.random() * 0.15;
            const color = new THREE.Color().setHSL(hue, 0.9, 0.6);
            colors.push(color.r, color.g, color.b);

            velocities.push(
                (Math.random() - 0.5) * 0.01,
                (Math.random() - 0.5) * 0.01,
                (Math.random() - 0.5) * 0.01
            );

            phases.push(Math.random() * Math.PI * 2);
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.12,
            vertexColors: true,
            transparent: true,
            opacity: 0.9,
            blending: THREE.AdditiveBlending
        });
        const points = new THREE.Points(geometry, material);
        scene.add(points);

        // Create connection lines
        const lineGeometry = new THREE.BufferGeometry();
        const linePositions = [];
        const lineColors = [];

        for (let i = 0; i < 500; i++) {
            const idx1 = Math.floor(Math.random() * 2000) * 3;
            const idx2 = Math.floor(Math.random() * 2000) * 3;

            linePositions.push(
                positions[idx1], positions[idx1+1], positions[idx1+2],
                positions[idx2], positions[idx2+1], positions[idx2+2]
            );

            lineColors.push(0.5, 0.2, 0.8, 0.3, 0.8, 0.9);
        }

        lineGeometry.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
        lineGeometry.setAttribute('color', new THREE.Float32BufferAttribute(lineColors, 3));

        const lineMaterial = new THREE.LineBasicMaterial({
            vertexColors: true,
            transparent: true,
            opacity: 0.3,
            blending: THREE.AdditiveBlending
        });
        const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
        scene.add(lines);

        // Animation
        let time = 0;
        function animate() {
            requestAnimationFrame(animate);
            time += 0.016;

            // Pulse neurons
            const posArray = geometry.attributes.position.array;
            const colArray = geometry.attributes.color.array;

            for (let i = 0; i < posArray.length; i += 3) {
                const idx = i / 3;
                const pulse = Math.sin(time * 2 + phases[idx]) * 0.5 + 0.5;

                // Subtle position pulse
                posArray[i] += velocities[i] * pulse;
                posArray[i + 1] += velocities[i + 1] * pulse;
                posArray[i + 2] += velocities[i + 2] * pulse;

                // Color pulse
                if (Math.random() < 0.001) {
                    colArray[i] = 1;
                    colArray[i + 1] = 0.8;
                    colArray[i + 2] = 1;
                } else {
                    colArray[i] *= 0.99;
                    colArray[i + 1] *= 0.99;
                    colArray[i + 2] *= 0.99;
                    colArray[i] = Math.max(colArray[i], 0.5);
                    colArray[i + 2] = Math.max(colArray[i + 2], 0.6);
                }
            }
            geometry.attributes.position.needsUpdate = true;
            geometry.attributes.color.needsUpdate = true;

            // Rotate brain
            points.rotation.y += 0.002;
            lines.rotation.y += 0.002;

            // Camera orbit
            camera.position.x = Math.sin(time * 0.1) * 20;
            camera.position.z = Math.cos(time * 0.1) * 20;
            camera.lookAt(0, 0, 0);

            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        });
    </script>
</body>
</html>
"""

    def _start_monitoring(self):
        """Start background monitoring."""

        def update_ara_status():
            """Update Ara connection and status."""
            status = self.ara_client.get_status()

            if status:
                self.ara_status.set_text("⚡ ARA: ONLINE")
                self.ara_status.remove_css_class('status-disconnected')
                self.ara_status.add_css_class('status-connected')

                self.ara_conn_label.set_text("🟢 NEURAL LINK ACTIVE")
                self.ara_mode_label.set_text(f"Mode: {status.get('mode', 'Unknown')}")

                # Update PAD
                pad = status.get('pad', {})
                v = pad.get('valence', 0)
                a = pad.get('arousal', 0)
                d = pad.get('dominance', 0)

                # Convert -1 to 1 range to 0 to 1 for progress bars
                self.pad_v_bar.set_fraction((v + 1) / 2)
                self.pad_v_bar.set_text(f"Valence")
                self.pad_v_value.set_text(f"{v:.2f}")

                self.pad_a_bar.set_fraction((a + 1) / 2)
                self.pad_a_bar.set_text(f"Arousal")
                self.pad_a_value.set_text(f"{a:.2f}")

                self.pad_d_bar.set_fraction((d + 1) / 2)
                self.pad_d_bar.set_text(f"Dominance")
                self.pad_d_value.set_text(f"{d:.2f}")

                # Determine emotion
                emotion = self._get_emotion_label(v, a, d)
                self.emotion_label.set_text(emotion)

                # Update CLV
                clv = status.get('clv', {})
                risk = clv.get('risk_level', 'LOW')
                self.clv_risk_label.set_text(f"RISK: {risk}")

                if risk == 'HIGH' or risk == 'CRITICAL':
                    self.clv_risk_label.add_css_class('metric-warning')
                else:
                    self.clv_risk_label.remove_css_class('metric-warning')

                inst = clv.get('instability', 0)
                res = clv.get('resource', 0)
                struct = clv.get('structural', 0)

                self.clv_instability_bar.set_fraction(min(inst, 1.0))
                self.clv_instability_bar.set_text(f"{inst:.1%}")
                self.clv_resource_bar.set_fraction(min(res, 1.0))
                self.clv_resource_bar.set_text(f"{res:.1%}")
                self.clv_structural_bar.set_fraction(min(struct, 1.0))
                self.clv_structural_bar.set_text(f"{struct:.1%}")

            else:
                self.ara_status.set_text("⚠ ARA: OFFLINE")
                self.ara_status.remove_css_class('status-connected')
                self.ara_status.add_css_class('status-disconnected')
                self.ara_conn_label.set_text("🔴 NEURAL LINK DOWN")

            return GLib.SOURCE_CONTINUE

        def update_kitten_status():
            """Update Forest Kitten 33 status."""
            kitten = self.ara_client.get_kitten_status()

            if kitten:
                hw_type = kitten.get('hardware_type', 'UNKNOWN')
                self.kitten_type_label.set_text(hw_type)

                if 'HARDWARE' in hw_type:
                    self.kitten_type_label.remove_css_class('kitten-emulated')
                    self.kitten_type_label.add_css_class('kitten-hardware')
                else:
                    self.kitten_type_label.remove_css_class('kitten-hardware')
                    self.kitten_type_label.add_css_class('kitten-emulated')

                device = kitten.get('device_path', '--')
                self.kitten_device_label.set_text(device)

                neurons = kitten.get('neurons', 0)
                self.kitten_neurons_label.set_text(f"{neurons:,}")

                steps = kitten.get('total_steps', 0)
                self.kitten_steps_label.set_text(f"{steps:,}")

                spike_rate = kitten.get('spike_rate', 0)
                self.spike_rate = spike_rate
                self.kitten_spike_label.set_text(f"{spike_rate:.2f}%")
                self.kitten_spike_bar.set_fraction(min(spike_rate / 100, 1.0))
                self.kitten_spike_bar.set_text(f"{spike_rate:.1f}%")

                # Update status bar
                self.kitten_status.set_text(f"🐱 KITTEN: {hw_type}")

                # Update neural grid visualization
                self._update_neural_grid(spike_rate)
            else:
                self.kitten_status.set_text("🐱 KITTEN: OFFLINE")

            return GLib.SOURCE_CONTINUE

        def update_system_metrics():
            """Update system metrics."""
            cpu = MetricsCollector.get_cpu_metrics()
            ram = MetricsCollector.get_ram_metrics()
            gpus = MetricsCollector.get_gpu_metrics()

            # CPU
            self.cpu_value_label.set_text(f"{cpu['usage_total']:.0f}%")
            self.cpu_cores_label.set_text(f"{cpu['cores']} cores @ {cpu['frequency']:.0f} MHz")

            if cpu['usage_total'] > 80:
                self.cpu_value_label.add_css_class('metric-warning')
            else:
                self.cpu_value_label.remove_css_class('metric-warning')

            # RAM
            self.ram_value_label.set_text(f"{ram['used_gb']:.1f} GB")
            self.ram_progress.set_fraction(ram['percent'] / 100.0)
            self.ram_progress.set_text(f"{ram['percent']:.0f}% of {ram['total_gb']:.0f} GB")

            # GPU
            if gpus and GPU_AVAILABLE:
                # Update GPU status bar
                self.gpu_temp_status.set_text(f"🌡 GPU: {gpus[0]['temperature']:.0f}°C")

                # Update GPU cards
                while True:
                    child = self.gpu_cards_box.get_first_child()
                    if not child:
                        break
                    self.gpu_cards_box.remove(child)

                for gpu in gpus:
                    card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
                    card.add_css_class('metric-card')

                    name_label = Gtk.Label(label=gpu['name'])
                    name_label.add_css_class('metric-title')
                    card.append(name_label)

                    util_label = Gtk.Label(label=f"{gpu['load']:.0f}%")
                    util_label.add_css_class('metric-value-huge')
                    if gpu['load'] > 80:
                        util_label.add_css_class('metric-warning')
                    card.append(util_label)

                    vram_label = Gtk.Label(label=f"VRAM: {gpu['memory_used_mb']:.0f} / {gpu['memory_total_mb']:.0f} MB")
                    vram_label.add_css_class('metric-label')
                    card.append(vram_label)

                    temp_label = Gtk.Label(label=f"Temp: {gpu['temperature']:.0f}°C")
                    temp_label.add_css_class('metric-label')
                    if gpu['temperature'] > 80:
                        temp_label.add_css_class('metric-warning')
                    card.append(temp_label)

                    self.gpu_cards_box.append(card)

            # Storage
            storage = MetricsCollector.get_storage_metrics() if hasattr(MetricsCollector, 'get_storage_metrics') else {'disks': []}

            while True:
                child = self.storage_cards_box.get_first_child()
                if not child:
                    break
                self.storage_cards_box.remove(child)

            for disk in storage.get('disks', []):
                card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
                card.add_css_class('metric-card')

                mount_label = Gtk.Label(label=disk.get('mountpoint', '/'))
                mount_label.add_css_class('metric-title')
                card.append(mount_label)

                usage_label = Gtk.Label(label=f"{disk.get('used_gb', 0):.1f} / {disk.get('total_gb', 0):.1f} GB")
                usage_label.add_css_class('metric-value-large')
                card.append(usage_label)

                progress = Gtk.ProgressBar()
                progress.add_css_class('metric-progress')
                progress.set_fraction(disk.get('percent', 0) / 100.0)
                progress.set_text(f"{disk.get('percent', 0):.0f}%")
                progress.set_show_text(True)
                card.append(progress)

                self.storage_cards_box.append(card)

            return GLib.SOURCE_CONTINUE

        # Add get_storage_metrics if missing
        if not hasattr(MetricsCollector, 'get_storage_metrics'):
            @staticmethod
            def get_storage_metrics():
                disks = []
                for partition in psutil.disk_partitions():
                    if partition.fstype:
                        try:
                            usage = psutil.disk_usage(partition.mountpoint)
                            disks.append({
                                'device': partition.device,
                                'mountpoint': partition.mountpoint,
                                'total_gb': usage.total / (1024**3),
                                'used_gb': usage.used / (1024**3),
                                'percent': usage.percent
                            })
                        except:
                            pass
                return {'disks': disks}
            MetricsCollector.get_storage_metrics = get_storage_metrics

        # Start update timers
        GLib.timeout_add_seconds(2, update_ara_status)
        GLib.timeout_add_seconds(1, update_kitten_status)
        GLib.timeout_add_seconds(2, update_system_metrics)

        # Initial updates
        GLib.idle_add(update_ara_status)
        GLib.idle_add(update_kitten_status)
        GLib.idle_add(update_system_metrics)

    def _update_neural_grid(self, spike_rate):
        """Update neural grid visualization based on spike rate."""
        # Calculate how many neurons should be "firing"
        num_active = int((spike_rate / 100) * 128)  # 8x16 = 128 neurons

        # Randomly select which neurons are active
        self.neural_spikes = random.sample(range(128), min(num_active, 128))

        # Queue redraw for all neurons
        for row in self.neural_dots:
            for dot in row:
                dot.queue_draw()

    def _get_emotion_label(self, v, a, d):
        """Get emotion label from PAD values."""
        if v > 0.3 and a > 0.3:
            return "😄 EXCITED"
        elif v > 0.3 and a < -0.3:
            return "😌 CONTENT"
        elif v < -0.3 and a > 0.3:
            return "😠 FRUSTRATED"
        elif v < -0.3 and a < -0.3:
            return "😔 SAD"
        elif v > 0.3:
            return "🙂 HAPPY"
        elif v < -0.3:
            return "😞 UNHAPPY"
        elif a > 0.3:
            return "⚡ ALERT"
        elif a < -0.3:
            return "😴 CALM"
        else:
            return "😐 NEUTRAL"


class CockpitHUDApplication(Adw.Application):
    """Ara Cockpit HUD Application."""

    def __init__(self, **kwargs):
        super().__init__(application_id='com.ara.cockpit', **kwargs)
        self.fullscreen_mode = False

    def do_activate(self):
        """Activate application."""
        win = self.props.active_window
        if not win:
            win = CockpitHUDWindow(application=self)

        # Check command line args for fullscreen
        if self.fullscreen_mode or '--fullscreen' in sys.argv or '-f' in sys.argv:
            win.fullscreen()
        else:
            # Windowed mode - good size for side monitor
            win.set_default_size(800, 1000)

        win.present()


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)

    # Parse args
    import argparse
    parser = argparse.ArgumentParser(description='Ara Cockpit HUD')
    parser.add_argument('-f', '--fullscreen', action='store_true', help='Start in fullscreen mode')
    parser.add_argument('--monitor', type=str, help='Target monitor (e.g., HDMI-1)')
    args, remaining = parser.parse_known_args()

    app = CockpitHUDApplication()
    app.fullscreen_mode = args.fullscreen
    return app.run(remaining)


if __name__ == '__main__':
    sys.exit(main())
