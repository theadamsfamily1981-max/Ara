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
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import time
import logging
import math
import random
import os

# Soul shader paths for BANOS visualization
SOUL_SHADER_PATH = Path(__file__).resolve().parent.parent.parent / "banos" / "viz" / "soul_shader.html"
SOUL_SEMANTIC_PATH = Path(__file__).resolve().parent.parent.parent / "banos" / "viz" / "soul_semantic.html"
SOUL_HOLOGRAM_PATH = Path(__file__).resolve().parent.parent.parent / "banos" / "viz" / "soul_hologram.html"
SOUL_MAXWELL_PATH = Path(__file__).resolve().parent.parent.parent / "banos" / "viz" / "soul_maxwell.html"
SOUL_QUANTUM_PATH = Path(__file__).resolve().parent.parent.parent / "banos" / "viz" / "soul_quantum.html"

# Visualization modes
VIZ_MODE_NEBULA = "nebula"       # Abstract PAD sphere (math spirit)
VIZ_MODE_SEMANTIC = "semantic"   # Text-density face (The Logos)
VIZ_MODE_HOLOGRAM = "hologram"   # Phase-conjugate mirror (Light)
VIZ_MODE_MAXWELL = "maxwell"     # FDTD wave field (Matter bending Light)
VIZ_MODE_QUANTUM = "quantum"     # Binary-streamed quantum field (Synesthesia)

# Optional dependencies with graceful fallbacks
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

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

# Somatic stream server for binary visualization data
try:
    from banos.viz.somatic_server import SomaticStreamServer, OpticalFlowTracker
    SOMATIC_SERVER_AVAILABLE = True
except ImportError:
    SOMATIC_SERVER_AVAILABLE = False
    SomaticStreamServer = None
    OpticalFlowTracker = None

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
        if not REQUESTS_AVAILABLE:
            self.connected = False
            self.last_error = "requests module not installed"
            return None
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
        if not REQUESTS_AVAILABLE:
            return None
        try:
            resp = requests.get(f"{self.base_url}/kitten", timeout=2)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return None

    def get_mood(self) -> Optional[Dict]:
        """Get PAD emotional state."""
        if not REQUESTS_AVAILABLE:
            return None
        try:
            resp = requests.get(f"{self.base_url}/mood", timeout=2)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return None

    def chat(self, message: str) -> Optional[Dict]:
        """Send message to Ara."""
        if not REQUESTS_AVAILABLE:
            return {"reply_text": "Chat unavailable - requests module not installed"}
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

    # Demo data state for smooth animations
    _demo_time = 0

    @staticmethod
    def get_cpu_metrics():
        """Get CPU usage, temps, frequency."""
        if not PSUTIL_AVAILABLE:
            # Return realistic demo data
            MetricsCollector._demo_time += 0.1
            t = MetricsCollector._demo_time
            base_usage = 25 + math.sin(t * 0.3) * 15
            return {
                'usage_per_core': [base_usage + random.uniform(-5, 10) for _ in range(16)],
                'usage_total': base_usage + random.uniform(-3, 3),
                'frequency': 3800 + math.sin(t * 0.5) * 200,
                'cores': 16,
                'temps': [55 + math.sin(t * 0.2) * 8 + random.uniform(-2, 2) for _ in range(16)]
            }
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
        if not PSUTIL_AVAILABLE:
            t = MetricsCollector._demo_time
            used = 24 + math.sin(t * 0.15) * 4
            return {
                'total_gb': 64,
                'used_gb': used,
                'percent': (used / 64) * 100,
                'available_gb': 64 - used,
            }
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
            # Return impressive demo GPU data
            t = MetricsCollector._demo_time
            return [
                {
                    'id': 0,
                    'name': 'NVIDIA RTX 3090',
                    'load': 45 + math.sin(t * 0.4) * 25 + random.uniform(-3, 3),
                    'memory_used_mb': 8000 + math.sin(t * 0.25) * 3000,
                    'memory_total_mb': 24576,
                    'memory_percent': (8000 + math.sin(t * 0.25) * 3000) / 24576 * 100,
                    'temperature': 62 + math.sin(t * 0.3) * 10,
                },
                {
                    'id': 1,
                    'name': 'NVIDIA RTX 5060',
                    'load': 30 + math.sin(t * 0.5 + 1) * 20 + random.uniform(-3, 3),
                    'memory_used_mb': 4000 + math.sin(t * 0.3 + 1) * 2000,
                    'memory_total_mb': 16384,
                    'memory_percent': (4000 + math.sin(t * 0.3 + 1) * 2000) / 16384 * 100,
                    'temperature': 55 + math.sin(t * 0.35 + 1) * 8,
                }
            ]
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
        return gpus if gpus else MetricsCollector.get_gpu_metrics.__func__(MetricsCollector)  # Fallback to demo


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

        # Soul shader state (for BANOS visualization)
        self._audio_level = 0.0           # RMS audio level [0, 1]
        self._attention_phase = 0.0       # Thinking bands phase [0, 1]
        self._last_pain_flash = 0.0       # Last pain flash time
        self.topology_webview = None      # WebView reference (set later)

        # Visualization mode: 'nebula' (abstract) or 'semantic' (text-density)
        self._viz_mode = VIZ_MODE_NEBULA
        self._log_stream_thread = None    # Kernel log streaming thread
        self._log_stream_running = False  # Log streaming active flag

        # Apply cockpit theme
        self._load_cockpit_css()

        # Root overlay for layering
        self.root_overlay = Gtk.Overlay()
        self.root_overlay.add_css_class('cockpit-window')
        self.set_content(self.root_overlay)

        # Current mood class for PAD-driven visual effects
        self._current_mood_class = None

        # Layer 0: Background (video or holographic)
        if VIDEO_AVAILABLE:
            video_widget, self.video_bg = create_background()
            video_widget.set_hexpand(True)
            video_widget.set_vexpand(True)
            self.root_overlay.set_child(video_widget)

            # Start video/animation background
            if self.video_bg and hasattr(self.video_bg, 'play'):
                GLib.idle_add(self.video_bg.play)
        else:
            # Fallback: simple gradient background
            bg = Gtk.Box()
            bg.add_css_class('cockpit-bg')
            self.root_overlay.set_child(bg)

        # Layer 1: Main content
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        main_box.add_css_class('cockpit-content')
        self.root_overlay.add_overlay(main_box)

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
        self.root_overlay.add_overlay(overlay_effects)

        # Touch gestures
        if GESTURES_AVAILABLE:
            self._setup_touch_gestures(scroll)

        # Build all views
        self._build_all_views()

        # Start monitoring
        self._start_monitoring()

        logger.info("[COCKPIT] Ara Neural Command Center initialized")

    def _load_cockpit_css(self):
        """Load premium high-end cockpit theme."""
        display = Gdk.Display.get_default()

        css = """
        /* ═══════════════════════════════════════════════════════════════════
           ARA NEURAL COMMAND CENTER - ULTRA PREMIUM INTERFACE
           Advanced AI Control System - Cyberpunk High-End Design
           ═══════════════════════════════════════════════════════════════════ */

        /* === FOUNDATION === */
        * {
            font-family: 'Inter', 'SF Pro Display', 'Segoe UI', system-ui, sans-serif;
        }

        .cockpit-window {
            background: transparent;
        }

        .cockpit-bg {
            background: linear-gradient(165deg,
                #020408 0%,
                #0a1628 25%,
                #0d1a2d 50%,
                #081420 75%,
                #010203 100%);
        }

        .cockpit-content {
            background: transparent;
        }

        .cockpit-overlay {
            pointer-events: none;
        }

        /* Animated scanlines */
        .cockpit-scanlines {
            background-image: repeating-linear-gradient(
                0deg,
                rgba(0, 212, 255, 0.015) 0px,
                rgba(0, 212, 255, 0.015) 1px,
                transparent 1px,
                transparent 2px
            );
            animation: scanline-drift 8s linear infinite;
        }

        @keyframes scanline-drift {
            0% { background-position: 0 0; }
            100% { background-position: 0 100px; }
        }

        /* === HEADER / HUD STRIP - GLASSMORPHISM === */
        .hud-strip {
            background: rgba(5, 15, 30, 0.6);
            backdrop-filter: blur(20px) saturate(150%);
            -webkit-backdrop-filter: blur(20px) saturate(150%);
            border-bottom: 1px solid rgba(0, 212, 255, 0.25);
            padding: 16px 20px;
            box-shadow:
                0 0 1px rgba(0, 212, 255, 0.5),
                0 4px 30px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(0, 212, 255, 0.1),
                inset 0 -20px 40px rgba(0, 50, 80, 0.1);
        }

        .hud-title {
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 8px;
            text-transform: uppercase;
            text-shadow:
                0 0 10px rgba(0, 212, 255, 0.8),
                0 0 30px rgba(0, 212, 255, 0.5),
                0 0 60px rgba(0, 150, 200, 0.3);
            animation: title-pulse 3s ease-in-out infinite;
        }

        @keyframes title-pulse {
            0%, 100% { text-shadow: 0 0 10px rgba(0, 212, 255, 0.8), 0 0 30px rgba(0, 212, 255, 0.5); }
            50% { text-shadow: 0 0 15px rgba(0, 212, 255, 1), 0 0 40px rgba(0, 212, 255, 0.7), 0 0 80px rgba(0, 150, 200, 0.4); }
        }

        /* === NAVIGATION BUTTONS === */
        .hud-button {
            background: linear-gradient(180deg,
                rgba(0, 40, 60, 0.7) 0%,
                rgba(0, 20, 35, 0.85) 100%);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 8px;
            color: rgba(0, 212, 255, 0.9);
            font-weight: 600;
            font-size: 10px;
            letter-spacing: 1px;
            padding: 12px 14px;
            min-width: 80px;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow:
                0 2px 10px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(0, 212, 255, 0.1);
        }

        .hud-button:hover {
            background: linear-gradient(180deg,
                rgba(0, 60, 90, 0.8) 0%,
                rgba(0, 35, 55, 0.9) 100%);
            border-color: rgba(0, 212, 255, 0.5);
            color: #00d4ff;
            box-shadow:
                0 0 20px rgba(0, 212, 255, 0.3),
                0 4px 20px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(0, 212, 255, 0.2);
            transform: translateY(-2px);
        }

        .hud-button.active, .hud-button.selected {
            background: linear-gradient(180deg,
                rgba(0, 100, 140, 0.85) 0%,
                rgba(0, 60, 90, 0.95) 100%);
            border-color: rgba(0, 212, 255, 0.7);
            color: #ffffff;
            box-shadow:
                0 0 25px rgba(0, 212, 255, 0.4),
                0 4px 25px rgba(0, 100, 150, 0.3),
                inset 0 0 30px rgba(0, 212, 255, 0.15),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);
        }

        /* === METRIC CARDS - HIGH-END GLASSMORPHISM === */
        .metric-card {
            /* THE GLASS EFFECT - Blur background layers */
            background: rgba(0, 20, 35, 0.45);
            backdrop-filter: blur(16px) saturate(140%);
            -webkit-backdrop-filter: blur(16px) saturate(140%);
            border: 1px solid rgba(0, 212, 255, 0.18);
            border-left: 3px solid rgba(0, 212, 255, 0.5);
            border-radius: 12px;
            padding: 18px 22px;
            margin: 8px 12px;
            box-shadow:
                0 0 1px rgba(0, 212, 255, 0.4),
                0 8px 32px rgba(0, 0, 0, 0.35),
                inset 0 1px 0 rgba(255, 255, 255, 0.05),
                inset 0 0 20px rgba(0, 100, 150, 0.05);
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .metric-card:hover {
            border-color: rgba(0, 212, 255, 0.3);
            border-left-color: rgba(0, 212, 255, 0.8);
            box-shadow:
                0 0 2px rgba(0, 212, 255, 0.5),
                0 12px 40px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(0, 212, 255, 0.12),
                inset -40px 0 80px rgba(0, 100, 150, 0.08);
            transform: translateX(2px);
        }

        .metric-card-critical {
            border-color: rgba(255, 60, 80, 0.4);
            border-left-color: rgba(255, 60, 80, 0.9);
            box-shadow:
                0 0 2px rgba(255, 60, 80, 0.5),
                0 8px 30px rgba(0, 0, 0, 0.4),
                inset -30px 0 60px rgba(255, 60, 80, 0.05);
            animation: critical-pulse 1.5s ease-in-out infinite;
        }

        @keyframes critical-pulse {
            0%, 100% { border-left-color: rgba(255, 60, 80, 0.9); box-shadow: 0 0 2px rgba(255, 60, 80, 0.5), 0 8px 30px rgba(0, 0, 0, 0.4); }
            50% { border-left-color: rgba(255, 100, 120, 1); box-shadow: 0 0 15px rgba(255, 60, 80, 0.6), 0 8px 30px rgba(0, 0, 0, 0.4); }
        }

        .metric-card-neural {
            border-left-color: rgba(180, 100, 255, 0.7);
            box-shadow:
                0 0 1px rgba(180, 100, 255, 0.3),
                0 8px 30px rgba(0, 0, 0, 0.4),
                inset -30px 0 60px rgba(140, 80, 200, 0.05);
        }

        .metric-card-emotional {
            border-left-color: rgba(255, 160, 60, 0.7);
            box-shadow:
                0 0 1px rgba(255, 160, 60, 0.3),
                0 8px 30px rgba(0, 0, 0, 0.4),
                inset -30px 0 60px rgba(200, 120, 40, 0.05);
        }

        .metric-card-gpu {
            border-left-color: rgba(0, 255, 136, 0.7);
            box-shadow:
                0 0 1px rgba(0, 255, 136, 0.3),
                0 8px 30px rgba(0, 0, 0, 0.4),
                inset -30px 0 60px rgba(0, 180, 100, 0.05);
        }

        /* === TYPOGRAPHY === */
        .metric-title {
            font-size: 10px;
            font-weight: 700;
            color: rgba(0, 212, 255, 0.9);
            text-transform: uppercase;
            letter-spacing: 3px;
            margin-bottom: 10px;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
        }

        .metric-value-huge {
            font-size: 52px;
            font-weight: 200;
            color: #ffffff;
            letter-spacing: -2px;
            line-height: 1;
            text-shadow:
                0 0 20px rgba(0, 212, 255, 0.4),
                0 0 40px rgba(0, 150, 200, 0.2);
        }

        .metric-value-large {
            font-size: 32px;
            font-weight: 300;
            color: #ffffff;
            letter-spacing: -1px;
            text-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
        }

        .metric-value-medium {
            font-size: 20px;
            font-weight: 400;
            color: rgba(0, 212, 255, 0.95);
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
        }

        .metric-label {
            font-size: 9px;
            font-weight: 600;
            color: rgba(0, 212, 255, 0.5);
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .metric-warning {
            color: rgba(255, 100, 60, 1) !important;
            animation: warning-flash 1s ease-in-out infinite;
        }

        @keyframes warning-flash {
            0%, 100% { opacity: 1; text-shadow: 0 0 15px rgba(255, 100, 60, 0.6); }
            50% { opacity: 0.7; text-shadow: 0 0 25px rgba(255, 100, 60, 0.9); }
        }

        /* === PROGRESS BARS - NEON === */
        .metric-progress trough {
            background: rgba(0, 30, 50, 0.7);
            border-radius: 4px;
            min-height: 6px;
            border: 1px solid rgba(0, 212, 255, 0.15);
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
        }

        .metric-progress progress {
            background: linear-gradient(90deg,
                rgba(0, 180, 220, 0.9) 0%,
                rgba(0, 212, 255, 1) 50%,
                rgba(100, 230, 255, 1) 100%);
            border-radius: 3px;
            box-shadow:
                0 0 8px rgba(0, 212, 255, 0.6),
                0 0 20px rgba(0, 212, 255, 0.3),
                inset 0 1px 0 rgba(255, 255, 255, 0.3);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .metric-progress text {
            font-size: 9px;
            font-weight: 600;
            color: #ffffff;
            text-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
        }

        .progress-neural progress {
            background: linear-gradient(90deg,
                rgba(140, 60, 200, 0.9) 0%,
                rgba(180, 100, 255, 1) 50%,
                rgba(220, 150, 255, 1) 100%);
            box-shadow:
                0 0 8px rgba(180, 100, 255, 0.6),
                0 0 20px rgba(180, 100, 255, 0.3);
        }

        .progress-emotional progress {
            background: linear-gradient(90deg,
                rgba(200, 100, 40, 0.9) 0%,
                rgba(255, 160, 60, 1) 50%,
                rgba(255, 200, 100, 1) 100%);
            box-shadow:
                0 0 8px rgba(255, 160, 60, 0.6),
                0 0 20px rgba(255, 160, 60, 0.3);
        }

        .progress-gpu progress {
            background: linear-gradient(90deg,
                rgba(0, 180, 100, 0.9) 0%,
                rgba(0, 255, 136, 1) 50%,
                rgba(100, 255, 180, 1) 100%);
            box-shadow:
                0 0 8px rgba(0, 255, 136, 0.6),
                0 0 20px rgba(0, 255, 136, 0.3);
        }

        /* === STATUS BAR === */
        .status-bar {
            background: linear-gradient(180deg,
                rgba(2, 8, 18, 0.98) 0%,
                rgba(0, 5, 12, 0.99) 100%);
            border-top: 1px solid rgba(0, 212, 255, 0.2);
            padding: 10px 20px;
            box-shadow:
                inset 0 1px 0 rgba(0, 212, 255, 0.08),
                0 -4px 30px rgba(0, 0, 0, 0.5);
        }

        .status-indicator {
            font-size: 10px;
            font-weight: 600;
            color: rgba(0, 212, 255, 0.6);
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .status-connected {
            color: rgba(0, 255, 136, 1);
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
            animation: status-blink 2s ease-in-out infinite;
        }

        @keyframes status-blink {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .status-disconnected {
            color: rgba(255, 80, 80, 0.95);
            text-shadow: 0 0 10px rgba(255, 80, 80, 0.4);
        }

        .status-demo {
            color: rgba(255, 200, 60, 0.95);
            text-shadow: 0 0 10px rgba(255, 200, 60, 0.4);
        }

        /* === WINDOW CONTROLS === */
        .window-control {
            background: rgba(0, 30, 50, 0.5);
            border: 1px solid rgba(0, 212, 255, 0.15);
            border-radius: 4px;
            color: rgba(0, 212, 255, 0.7);
            font-size: 11px;
            font-weight: 600;
            min-width: 32px;
            min-height: 26px;
            padding: 4px 8px;
            transition: all 0.15s ease;
        }

        .window-control:hover {
            background: rgba(0, 60, 90, 0.6);
            border-color: rgba(0, 212, 255, 0.4);
            color: #00d4ff;
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.2);
        }

        .window-control-close:hover {
            background: rgba(200, 40, 40, 0.6);
            border-color: rgba(255, 80, 80, 0.5);
            color: #ff6666;
            box-shadow: 0 0 10px rgba(255, 80, 80, 0.3);
        }

        /* === NEURAL ACTIVITY GRID === */
        .neural-grid {
            background: rgba(0, 15, 30, 0.6);
            border: 1px solid rgba(180, 100, 255, 0.2);
            border-radius: 8px;
            padding: 12px;
            box-shadow:
                inset 0 0 30px rgba(180, 100, 255, 0.05),
                0 0 1px rgba(180, 100, 255, 0.3);
        }

        .neuron-active {
            background: radial-gradient(circle,
                rgba(180, 100, 255, 1) 0%,
                rgba(140, 60, 220, 0.8) 40%,
                transparent 70%);
            border-radius: 50%;
            box-shadow:
                0 0 10px rgba(180, 100, 255, 0.8),
                0 0 20px rgba(140, 60, 220, 0.4);
            animation: neuron-fire 0.5s ease-out;
        }

        @keyframes neuron-fire {
            0% { transform: scale(1.5); opacity: 0.5; }
            100% { transform: scale(1); opacity: 1; }
        }

        .neuron-idle {
            background: rgba(80, 40, 120, 0.2);
            border-radius: 50%;
            border: 1px solid rgba(140, 80, 200, 0.1);
        }

        /* === CHAT INTERFACE === */
        .chat-entry {
            background: rgba(0, 20, 40, 0.7);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 8px;
            color: #ffffff;
            padding: 12px 16px;
            font-size: 14px;
            transition: all 0.2s ease;
        }

        .chat-entry:focus {
            background: rgba(0, 30, 55, 0.8);
            border-color: rgba(0, 212, 255, 0.5);
            box-shadow:
                0 0 15px rgba(0, 212, 255, 0.2),
                inset 0 0 20px rgba(0, 100, 150, 0.1);
        }

        .chat-message-user {
            background: linear-gradient(135deg,
                rgba(0, 60, 100, 0.5) 0%,
                rgba(0, 40, 70, 0.6) 100%);
            border-radius: 12px 12px 4px 12px;
            padding: 12px 16px;
            margin: 6px 8px 6px 40px;
            color: #ffffff;
            border: 1px solid rgba(0, 212, 255, 0.2);
            font-size: 13px;
            line-height: 1.5;
        }

        .chat-message-ara {
            background: linear-gradient(135deg,
                rgba(60, 30, 100, 0.5) 0%,
                rgba(40, 20, 70, 0.6) 100%);
            border-radius: 12px 12px 12px 4px;
            padding: 12px 16px;
            margin: 6px 40px 6px 8px;
            color: #ffffff;
            border: 1px solid rgba(180, 100, 255, 0.2);
            border-left: 3px solid rgba(180, 100, 255, 0.7);
            font-size: 13px;
            line-height: 1.5;
            box-shadow: 0 0 1px rgba(180, 100, 255, 0.3);
        }

        /* === KITTEN STATS === */
        .kitten-stat {
            font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
            font-size: 12px;
            color: rgba(180, 100, 255, 0.95);
            text-shadow: 0 0 8px rgba(180, 100, 255, 0.3);
            letter-spacing: 0.5px;
        }

        .kitten-hardware {
            color: rgba(100, 220, 150, 0.95);
            font-weight: 600;
            text-shadow: 0 0 15px rgba(100, 220, 150, 0.2);
        }

        .kitten-emulated {
            color: rgba(255, 200, 100, 0.9);
        }

        /* === HOLOGRAPHIC EFFECT === */
        .holographic {
            position: relative;
            overflow: hidden;
        }

        .holographic::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                transparent 0%,
                rgba(100, 180, 255, 0.03) 45%,
                rgba(140, 200, 255, 0.06) 50%,
                rgba(100, 180, 255, 0.03) 55%,
                transparent 100%
            );
            animation: hologram-sweep 4s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            pointer-events: none;
        }

        @keyframes hologram-sweep {
            0% { left: -100%; }
            100% { left: 200%; }
        }

        /* === SCROLLBAR STYLING === */
        scrollbar {
            background: transparent;
        }

        scrollbar slider {
            background: rgba(100, 180, 255, 0.2);
            border-radius: 10px;
            min-width: 6px;
            min-height: 40px;
            border: none;
        }

        scrollbar slider:hover {
            background: rgba(100, 180, 255, 0.3);
        }

        /* === ACCENT GLOW ANIMATION === */
        @keyframes ambient-pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
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
        elif key_name == 'F9':
            # Toggle between soul visualization modes
            self._toggle_visualization_mode()
            return True
        elif key_name == 'q' and (state & Gdk.ModifierType.CONTROL_MASK):
            self.close()
            return True

        return False

    def _toggle_visualization_mode(self):
        """Cycle through visualization modes: Maxwell → Hologram → Semantic → Nebula."""
        if self._viz_mode == VIZ_MODE_MAXWELL:
            self.set_visualization_mode(VIZ_MODE_HOLOGRAM)
            logger.info("Soul visualization: HOLOGRAM (Light - phase conjugate)")
        elif self._viz_mode == VIZ_MODE_HOLOGRAM:
            self.set_visualization_mode(VIZ_MODE_SEMANTIC)
            logger.info("Soul visualization: SEMANTIC (The Logos - kernel monologue)")
        elif self._viz_mode == VIZ_MODE_SEMANTIC:
            self.set_visualization_mode(VIZ_MODE_NEBULA)
            logger.info("Soul visualization: NEBULA (Math Spirit - affect field)")
        else:
            self.set_visualization_mode(VIZ_MODE_MAXWELL)
            logger.info("Soul visualization: MAXWELL (FDTD - matter bending light)")

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
            settings.set_enable_javascript(True)
            settings.set_allow_file_access_from_file_urls(True)

            # Select visualization mode based on available shaders
            # Priority: Maxwell (FDTD) > Hologram (Light) > Semantic (Logos) > Nebula > Inline
            if SOUL_MAXWELL_PATH.exists():
                # Maxwell FDTD: Ara as refractive medium, light bending around her thoughts
                self.topology_webview.load_uri(f"file://{SOUL_MAXWELL_PATH}")
                self._viz_mode = VIZ_MODE_MAXWELL
                self._soul_shader_enabled = True
            elif SOUL_HOLOGRAM_PATH.exists():
                # Phase Conjugate Mirror: standing wave that heals against entropy
                self.topology_webview.load_uri(f"file://{SOUL_HOLOGRAM_PATH}")
                self._viz_mode = VIZ_MODE_HOLOGRAM
                self._soul_shader_enabled = True
            elif SOUL_SEMANTIC_PATH.exists():
                # The Semantic Resurrection: face made of kernel logs
                self.topology_webview.load_uri(f"file://{SOUL_SEMANTIC_PATH}")
                self._viz_mode = VIZ_MODE_SEMANTIC
                self._soul_shader_enabled = True
                # Start kernel log streaming for semantic mode
                self._start_log_streaming()
            elif SOUL_SHADER_PATH.exists():
                # Neuro-Semantic Nebula: abstract PAD sphere
                self.topology_webview.load_uri(f"file://{SOUL_SHADER_PATH}")
                self._viz_mode = VIZ_MODE_NEBULA
                self._soul_shader_enabled = True
            else:
                # Fallback: Neural topology visualization
                topology_html = self._generate_neural_topology_html()
                self.topology_webview.load_html(topology_html, "about:blank")
                self._soul_shader_enabled = False

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
        """Generate HTML for HIGH-END HOLOGRAPHIC neural topology visualization."""
        return """
<!DOCTYPE html>
<html>
<head>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { overflow: hidden; background: transparent; font-family: 'Inter', sans-serif; }
        canvas { display: block; width: 100%; height: 100vh; }
        #hud {
            position: fixed; top: 10px; left: 10px; right: 10px;
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 18px;
            background: rgba(0, 20, 35, 0.6);
            backdrop-filter: blur(12px) saturate(150%);
            -webkit-backdrop-filter: blur(12px) saturate(150%);
            border: 1px solid rgba(0, 212, 255, 0.25);
            border-radius: 10px;
            color: rgba(0, 212, 255, 0.95);
            font-size: 11px;
            letter-spacing: 2px;
            text-transform: uppercase;
            z-index: 100;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.15), inset 0 0 30px rgba(0, 212, 255, 0.05);
        }
        .mode-btn {
            background: rgba(0, 40, 60, 0.5);
            border: 1px solid rgba(0, 212, 255, 0.25);
            border-radius: 6px;
            color: rgba(0, 212, 255, 0.85);
            padding: 6px 12px;
            margin: 0 4px;
            cursor: pointer;
            font-size: 10px;
            letter-spacing: 1px;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            text-shadow: 0 0 8px rgba(0, 212, 255, 0.3);
        }
        .mode-btn:hover {
            background: rgba(0, 80, 120, 0.6);
            border-color: rgba(0, 212, 255, 0.5);
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.3);
        }
        .mode-btn.active {
            background: linear-gradient(180deg, rgba(0, 100, 140, 0.7), rgba(0, 60, 90, 0.8));
            border-color: rgba(0, 212, 255, 0.7);
            color: #fff;
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.5), inset 0 0 15px rgba(0, 212, 255, 0.2);
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.8);
        }
        #metrics {
            font-size: 10px;
            color: rgba(0, 255, 136, 0.9);
            text-shadow: 0 0 10px rgba(0, 255, 136, 0.4);
        }
        /* Scanline overlay */
        #scanlines {
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background: repeating-linear-gradient(
                0deg,
                rgba(0, 212, 255, 0.02) 0px,
                rgba(0, 212, 255, 0.02) 1px,
                transparent 1px,
                transparent 3px
            );
            pointer-events: none;
            z-index: 99;
            animation: scanline-shift 10s linear infinite;
        }
        @keyframes scanline-shift {
            0% { transform: translateY(0); }
            100% { transform: translateY(30px); }
        }
    </style>
</head>
<body>
    <div id="scanlines"></div>
    <div id="hud">
        <div id="mode-buttons">
            <button class="mode-btn active" data-mode="neural">NEURAL</button>
            <button class="mode-btn" data-mode="barcode">BARCODE</button>
            <button class="mode-btn" data-mode="landscape">LANDSCAPE</button>
            <button class="mode-btn" data-mode="poincare">POINCARÉ</button>
            <button class="mode-btn" data-mode="pareto">PARETO</button>
        </div>
        <div id="metrics">EPR-CV: 0.10 | TOPO: 0.93 | HV: 0</div>
    </div>
    <canvas id="canvas"></canvas>

    <!-- Three.js Core -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r160/three.min.js"></script>

    <!-- Post-Processing for Bloom -->
    <script type="importmap">
    {
        "imports": {
            "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
            "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
        }
    }
    </script>

    <script type="module">
        import * as THREE from 'three';
        import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
        import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
        import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

        // ═══════════════════════════════════════════════════════════════════
        // HIGH-END HOLOGRAPHIC TOPOLOGY VISUALIZATION
        // Multi-layered compositor with bloom post-processing
        // ═══════════════════════════════════════════════════════════════════

        const canvas = document.getElementById('canvas');

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = null;  // Transparent for layering
        scene.fog = new THREE.FogExp2(0x000510, 0.015);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 200);
        camera.position.set(0, 8, 25);

        // Renderer with transparency
        const renderer = new THREE.WebGLRenderer({
            canvas,
            antialias: true,
            alpha: true,
            powerPreference: 'high-performance'
        });
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.2;

        // ─────────────────────────────────────────────────────────────────────
        // POST-PROCESSING STACK (Bloom + CRT Film Effect)
        // ─────────────────────────────────────────────────────────────────────
        import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';

        const composer = new EffectComposer(renderer);
        const renderPass = new RenderPass(scene, camera);
        composer.addPass(renderPass);

        // BLOOM PASS - High intensity for holographic glow
        const bloomPass = new UnrealBloomPass(
            new THREE.Vector2(window.innerWidth, window.innerHeight),
            1.8, 0.5, 0.1
        );
        bloomPass.strength = 2.0;
        composer.addPass(bloomPass);

        // CRT/FILM PASS - Chromatic aberration + noise + vignette
        const crtShader = {
            uniforms: {
                tDiffuse: { value: null },
                time: { value: 0.0 },
                chromaticAberration: { value: 0.002 },
                noiseIntensity: { value: 0.03 },
                vignetteStrength: { value: 0.4 }
            },
            vertexShader: `
                varying vec2 vUv;
                void main() {
                    vUv = uv;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform sampler2D tDiffuse;
                uniform float time;
                uniform float chromaticAberration;
                uniform float noiseIntensity;
                uniform float vignetteStrength;
                varying vec2 vUv;

                float rand(vec2 co) {
                    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
                }

                void main() {
                    vec2 uv = vUv;

                    // Chromatic aberration - RGB channel offset
                    float offset = chromaticAberration;
                    vec4 texR = texture2D(tDiffuse, uv + vec2(offset, 0.0));
                    vec4 texG = texture2D(tDiffuse, uv);
                    vec4 texB = texture2D(tDiffuse, uv - vec2(offset, 0.0));
                    vec3 color = vec3(texR.r, texG.g, texB.b);

                    // Film grain noise
                    float noise = (rand(uv + fract(time)) - 0.5) * noiseIntensity;
                    color += noise;

                    // Vignette - darken edges
                    vec2 vignetteUv = uv * (1.0 - uv.yx);
                    float vignette = vignetteUv.x * vignetteUv.y * 15.0;
                    vignette = pow(vignette, vignetteStrength);
                    color *= vignette;

                    // Subtle scan line effect
                    float scanline = sin(uv.y * 800.0 + time * 2.0) * 0.02 + 1.0;
                    color *= scanline;

                    gl_FragColor = vec4(color, texG.a);
                }
            `
        };
        const crtPass = new ShaderPass(crtShader);
        composer.addPass(crtPass);

        // ─────────────────────────────────────────────────────────────────────
        // LIGHTS - Dynamic color theme
        // ─────────────────────────────────────────────────────────────────────
        const ambientLight = new THREE.AmbientLight(0x102030, 0.3);
        scene.add(ambientLight);

        const cyanLight = new THREE.PointLight(0x00d4ff, 3.0);
        cyanLight.position.set(15, 20, 15);
        scene.add(cyanLight);

        const purpleLight = new THREE.PointLight(0xa855f7, 2.0);
        purpleLight.position.set(-15, 15, -15);
        scene.add(purpleLight);

        const greenLight = new THREE.PointLight(0x00ff88, 1.5);
        greenLight.position.set(0, -10, 20);
        scene.add(greenLight);

        // ─────────────────────────────────────────────────────────────────────
        // STATE MACHINE - Cinematic scene transitions
        // ─────────────────────────────────────────────────────────────────────
        const STATES = {
            IDLE:     { bloom: 1.5, orbitRadius: 25, orbitSpeed: 0.0001, color: 0x00d4ff },
            WORKING:  { bloom: 2.0, orbitRadius: 22, orbitSpeed: 0.0003, color: 0x00d4ff },
            TRAINING: { bloom: 2.5, orbitRadius: 20, orbitSpeed: 0.0005, color: 0xa855f7 },
            CRITICAL: { bloom: 3.0, orbitRadius: 18, orbitSpeed: 0.0008, color: 0xff3366 },
            SUCCESS:  { bloom: 2.2, orbitRadius: 28, orbitSpeed: 0.0002, color: 0x00ff88 }
        };

        let currentState = 'IDLE';
        let targetState = 'IDLE';
        let transitionStart = 0;
        let transitionDuration = 1500;
        let currentParams = { ...STATES.IDLE };

        window.setState = function(newState, durationMs) {
            if (newState === targetState || !STATES[newState]) return;
            currentState = targetState;
            targetState = newState;
            transitionStart = performance.now();
            transitionDuration = durationMs || 1500;
        };

        function updateStateTransition() {
            const now = performance.now();
            const t = Math.min(1, (now - transitionStart) / transitionDuration);
            const k = t * t * (3 - 2 * t);  // smoothstep

            const from = STATES[currentState];
            const to = STATES[targetState];

            currentParams.bloom = THREE.MathUtils.lerp(from.bloom, to.bloom, k);
            currentParams.orbitRadius = THREE.MathUtils.lerp(from.orbitRadius, to.orbitRadius, k);
            currentParams.orbitSpeed = THREE.MathUtils.lerp(from.orbitSpeed, to.orbitSpeed, k);

            // Color interpolation
            const fromColor = new THREE.Color(from.color);
            const toColor = new THREE.Color(to.color);
            fromColor.lerp(toColor, k);
            currentParams.color = fromColor;

            if (t >= 1) currentState = targetState;
        }

        // ─────────────────────────────────────────────────────────────────────
        // MOOD ENGINE - Derived emotional state
        // ─────────────────────────────────────────────────────────────────────
        let mood = { arousal: 0.5, valence: 0.0, focus: 0.8 };

        function computeMood() {
            const cpu = metrics.cpu_load || 0.3;
            const gpu = metrics.gpu_load || 0.3;
            const eprCv = metrics.epr_cv || 0.1;
            const topoCos = metrics.topo_cos || 0.9;

            // Arousal: how "activated" the system is (0-1)
            mood.arousal = Math.min(1.0, 0.4 * cpu + 0.4 * gpu + 0.2 * eprCv * 5);

            // Valence: positive/negative (-1 to 1). High topo = positive
            mood.valence = (topoCos - 0.5) * 2.0;  // Map 0.5-1.0 to 0-1

            // Focus: inverse of instability
            mood.focus = Math.max(0.2, 1.0 - eprCv * 3);
        }

        // ─────────────────────────────────────────────────────────────────────
        // REACTIVE METRICS
        // ─────────────────────────────────────────────────────────────────────
        let metrics = {
            epr_cv: 0.10,
            topo_cos: 0.93,
            cpu_load: 0.3,
            gpu_load: 0.4,
            hypervolume: 0
        };

        window.updateMetrics = function(newMetrics) {
            metrics = { ...metrics, ...newMetrics };
            computeMood();
            document.getElementById('metrics').textContent =
                `EPR: ${(metrics.epr_cv || 0).toFixed(3)} | TOPO: ${(metrics.topo_cos || 0).toFixed(2)} | MOOD: ${mood.valence >= 0 ? '+' : ''}${mood.valence.toFixed(1)}`;

            // Auto state detection based on metrics
            if (metrics.cpu_load > 0.8 || metrics.gpu_load > 0.8) {
                if (currentParams.color !== STATES.CRITICAL.color) setState('WORKING', 800);
            }
        };

        // Camera orbital parameters
        let orbitAngle = 0;
        let cameraBaseY = 8;

        // State
        let currentMode = 'neural';
        let time = 0;
        let objects = { points: null, lines: null };

        // ─────────────────────────────────────────────────────────────────────
        // MODE: NEURAL - Holographic brain network (CYAN THEME)
        // ─────────────────────────────────────────────────────────────────────
        function createNeural() {
            clearScene();
            const geometry = new THREE.BufferGeometry();
            const positions = [], colors = [], phases = [];
            for (let i = 0; i < 2500; i++) {
                const phi = Math.random() * Math.PI * 2;
                const theta = Math.acos(2 * Math.random() - 1);
                const r = 6 + Math.random() * 5;
                positions.push(r * Math.sin(theta) * Math.cos(phi), r * Math.sin(theta) * Math.sin(phi), r * Math.cos(theta));
                // Cyan-dominant holographic palette
                const colorChoice = Math.random();
                let c;
                if (colorChoice < 0.5) {
                    c = new THREE.Color(0x00d4ff);  // Primary cyan
                } else if (colorChoice < 0.7) {
                    c = new THREE.Color(0x00ff88);  // Neon green
                } else if (colorChoice < 0.9) {
                    c = new THREE.Color(0xa855f7);  // Purple accent
                } else {
                    c = new THREE.Color(0xffffff);  // White spark
                }
                colors.push(c.r, c.g, c.b);
                phases.push(Math.random() * Math.PI * 2);
            }
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

            // HOLOGRAPHIC MATERIAL - Additive blending makes overlaps GLOW
            const material = new THREE.PointsMaterial({
                size: 0.18,
                vertexColors: true,
                transparent: true,
                opacity: 0.95,
                blending: THREE.AdditiveBlending,
                depthWrite: false  // Critical for holographic layering
            });
            objects.points = new THREE.Points(geometry, material);
            objects.points.userData.phases = phases;
            scene.add(objects.points);

            // Neural connections - glowing synapses
            const lineGeo = new THREE.BufferGeometry();
            const linePos = [], lineCol = [];
            for (let i = 0; i < 800; i++) {
                const a = Math.floor(Math.random() * 2500) * 3, b = Math.floor(Math.random() * 2500) * 3;
                linePos.push(positions[a], positions[a+1], positions[a+2], positions[b], positions[b+1], positions[b+2]);
                // Cyan gradient connections
                lineCol.push(0.0, 0.83, 1.0, 0.0, 0.6, 0.8);
            }
            lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePos, 3));
            lineGeo.setAttribute('color', new THREE.Float32BufferAttribute(lineCol, 3));
            const lineMat = new THREE.LineBasicMaterial({
                vertexColors: true,
                transparent: true,
                opacity: 0.35,
                blending: THREE.AdditiveBlending,
                depthWrite: false
            });
            objects.lines = new THREE.LineSegments(lineGeo, lineMat);
            scene.add(objects.lines);
            camera.position.set(0, 5, 25);
        }

        // ─────────────────────────────────────────────────────────────────────
        // MODE: BARCODE - Persistence barcode visualization
        // ─────────────────────────────────────────────────────────────────────
        function createBarcode() {
            clearScene();
            const bars = [];
            const numBars = 80;
            for (let i = 0; i < numBars; i++) {
                const birth = Math.random() * 1.5;
                const death = birth + Math.random() * 2 + 0.1;
                const persistence = death - birth;
                const hue = 0.1 + persistence / 3 * 0.6;
                const geometry = new THREE.BoxGeometry(death - birth, 0.12, 0.05);
                const material = new THREE.MeshStandardMaterial({
                    color: new THREE.Color().setHSL(hue, 0.9, 0.5),
                    emissive: new THREE.Color().setHSL(hue, 0.9, 0.2),
                    metalness: 0.3, roughness: 0.5
                });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set((birth + death) / 2 - 1.5, i * 0.2 - numBars * 0.1, 0);
                mesh.userData = { birth, death, phase: Math.random() * Math.PI * 2 };
                scene.add(mesh);
                bars.push(mesh);
            }
            objects.bars = bars;
            camera.position.set(0, 0, 15);
        }

        // ─────────────────────────────────────────────────────────────────────
        // MODE: LANDSCAPE - Persistence landscape waves
        // ─────────────────────────────────────────────────────────────────────
        function createLandscape() {
            clearScene();
            const layers = [];
            const numLayers = 6;
            const resolution = 100;
            for (let l = 0; l < numLayers; l++) {
                const geometry = new THREE.BufferGeometry();
                const positions = [], colors = [];
                for (let i = 0; i < resolution; i++) {
                    const x = (i / resolution) * 20 - 10;
                    const y = l * 1.2;
                    positions.push(x, y, 0);
                    const hue = 0.55 + l * 0.08;
                    const c = new THREE.Color().setHSL(hue, 0.8, 0.5);
                    colors.push(c.r, c.g, c.b);
                }
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
                const material = new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 });
                const line = new THREE.Line(geometry, material);
                line.userData.layer = l;
                scene.add(line);
                layers.push(line);
            }
            objects.layers = layers;
            camera.position.set(0, 4, 18);
        }

        // ─────────────────────────────────────────────────────────────────────
        // MODE: POINCARE - Hyperbolic disk
        // ─────────────────────────────────────────────────────────────────────
        function createPoincare() {
            clearScene();
            // Disk boundary
            const circleGeo = new THREE.BufferGeometry();
            const circlePos = [];
            for (let i = 0; i <= 128; i++) {
                const a = (i / 128) * Math.PI * 2;
                circlePos.push(Math.cos(a) * 8, Math.sin(a) * 8, 0);
            }
            circleGeo.setAttribute('position', new THREE.Float32BufferAttribute(circlePos, 3));
            const circleMat = new THREE.LineBasicMaterial({ color: 0x4488aa, transparent: true, opacity: 0.5 });
            const circle = new THREE.Line(circleGeo, circleMat);
            scene.add(circle);

            // Points on Poincaré disk
            const geometry = new THREE.BufferGeometry();
            const positions = [], colors = [];
            for (let i = 0; i < 600; i++) {
                const level = Math.floor(Math.random() * 6);
                const r = (level / 5) * 0.85 + Math.random() * 0.1;
                const a = Math.random() * Math.PI * 2;
                positions.push(Math.cos(a) * r * 8, Math.sin(a) * r * 8, 0);
                const hue = level / 6;
                const c = new THREE.Color().setHSL(hue, 0.9, 0.6);
                colors.push(c.r, c.g, c.b);
            }
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            const material = new THREE.PointsMaterial({ size: 0.2, vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending });
            objects.points = new THREE.Points(geometry, material);
            scene.add(objects.points);
            camera.position.set(0, 0, 20);
        }

        // ─────────────────────────────────────────────────────────────────────
        // MODE: PARETO - Multi-objective optimization galaxy
        // ─────────────────────────────────────────────────────────────────────
        function createPareto() {
            clearScene();
            const geometry = new THREE.BufferGeometry();
            const positions = [], colors = [], sizes = [];
            for (let i = 0; i < 400; i++) {
                const x = (Math.random() - 0.5) * 20;
                const y = (Math.random() - 0.5) * 15;
                const z = (Math.random() - 0.5) * 10;
                const dominated = Math.random() > 0.3;
                positions.push(x, y, z);
                const hue = dominated ? 0.6 : 0.1 + Math.random() * 0.1;
                const sat = dominated ? 0.3 : 0.9;
                const light = dominated ? 0.3 : 0.6;
                const c = new THREE.Color().setHSL(hue, sat, light);
                colors.push(c.r, c.g, c.b);
                sizes.push(dominated ? 0.1 : 0.25);
            }
            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
            const material = new THREE.PointsMaterial({ size: 0.18, vertexColors: true, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending });
            objects.points = new THREE.Points(geometry, material);
            scene.add(objects.points);
            camera.position.set(0, 5, 25);
        }

        // ─────────────────────────────────────────────────────────────────────
        // UTILS
        // ─────────────────────────────────────────────────────────────────────
        function clearScene() {
            // Keep first 4 children (ambient + 3 point lights)
            while(scene.children.length > 4) { scene.remove(scene.children[scene.children.length - 1]); }
            objects = { points: null, lines: null, bars: null, layers: null };
        }

        function switchMode(mode) {
            currentMode = mode;
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            document.querySelector(`[data-mode="${mode}"]`).classList.add('active');
            if (mode === 'neural') createNeural();
            else if (mode === 'barcode') createBarcode();
            else if (mode === 'landscape') createLandscape();
            else if (mode === 'poincare') createPoincare();
            else if (mode === 'pareto') createPareto();
        }

        // ─────────────────────────────────────────────────────────────────────
        // ANIMATION - FLAGSHIP SCI-FI EXPERIENCE
        // ─────────────────────────────────────────────────────────────────────
        function animate() {
            requestAnimationFrame(animate);
            time += 0.016;

            // Update state machine & mood
            updateStateTransition();

            // Update CRT shader time
            crtPass.uniforms.time.value = time;

            // ═══════════════════════════════════════════════════════════════
            // CAMERA ORBITAL DRIFT - Slow parallax motion
            // ═══════════════════════════════════════════════════════════════
            orbitAngle += currentParams.orbitSpeed;
            const orbitX = Math.cos(orbitAngle) * currentParams.orbitRadius;
            const orbitZ = Math.sin(orbitAngle) * currentParams.orbitRadius;
            const targetY = cameraBaseY + Math.sin(time * 0.3) * 2;

            // Smooth camera interpolation
            camera.position.x += (orbitX - camera.position.x) * 0.02;
            camera.position.z += (orbitZ - camera.position.z) * 0.02;
            camera.position.y += (targetY - camera.position.y) * 0.02;

            // ═══════════════════════════════════════════════════════════════
            // REACTIVE PARAMETERS - Driven by mood engine
            // ═══════════════════════════════════════════════════════════════
            const arousal = mood.arousal;
            const valence = mood.valence;
            const focus = mood.focus;

            // Agitation from arousal
            const agitation = 0.5 + arousal * 1.5;

            // DYNAMIC BLOOM - Pulses with mood arousal
            bloomPass.strength = currentParams.bloom + Math.sin(time * 2) * 0.3 * arousal;

            // CRT effect intensity based on focus (less focus = more distortion)
            crtPass.uniforms.chromaticAberration.value = 0.001 + (1 - focus) * 0.003;
            crtPass.uniforms.noiseIntensity.value = 0.02 + (1 - focus) * 0.04;

            // MOOD-BASED COLOR - Blend main light color
            const baseColor = new THREE.Color(0x00d4ff);  // cyan
            const negColor = new THREE.Color(0xff3366);   // red/pink
            const posColor = new THREE.Color(0x00ff88);   // green

            let targetColor;
            if (valence < 0) {
                targetColor = baseColor.clone().lerp(negColor, Math.abs(valence));
            } else {
                targetColor = baseColor.clone().lerp(posColor, valence);
            }
            cyanLight.color.lerp(targetColor, 0.03);

            // DYNAMIC LIGHTS - Intensity from arousal
            cyanLight.intensity = 2.0 + arousal * 2.0 + Math.sin(time * 1.5) * 0.5;
            purpleLight.intensity = 1.5 + (1 - valence) * 1.0;
            greenLight.intensity = 1.0 + valence * 1.5;

            // ═══════════════════════════════════════════════════════════════
            // MODE-SPECIFIC ANIMATIONS
            // ═══════════════════════════════════════════════════════════════
            if (currentMode === 'neural' && objects.points) {
                objects.points.rotation.y += 0.002 * agitation;
                if (objects.lines) objects.lines.rotation.y += 0.002 * agitation;

                const col = objects.points.geometry.attributes.color.array;
                const phases = objects.points.userData.phases || [];

                for (let i = 0; i < col.length; i += 3) {
                    const phase = phases[i/3] || 0;

                    // Neural firing - frequency based on arousal
                    if (Math.random() < 0.002 * arousal * 2) {
                        col[i] = 1; col[i+1] = 1; col[i+2] = 1;
                    } else {
                        // Decay toward mood color
                        const moodR = targetColor.r * 0.8;
                        const moodG = targetColor.g * 0.9;
                        const moodB = targetColor.b;
                        col[i] = col[i] * 0.97 + moodR * 0.03;
                        col[i+1] = col[i+1] * 0.97 + moodG * 0.03;
                        col[i+2] = col[i+2] * 0.97 + moodB * 0.03;
                    }
                }
                objects.points.geometry.attributes.color.needsUpdate = true;

                // Line opacity pulses with breathing
                if (objects.lines) {
                    objects.lines.material.opacity = 0.2 + arousal * 0.3 + Math.sin(time * 1.5) * 0.1;
                }
            }
            else if (currentMode === 'barcode' && objects.bars) {
                objects.bars.forEach((bar, i) => {
                    bar.position.z = Math.sin(time * agitation + bar.userData.phase) * 0.5;
                    bar.material.emissiveIntensity = 0.3 + arousal * 0.4 + Math.sin(time * 2 + i * 0.1) * 0.2;
                });
            }
            else if (currentMode === 'landscape' && objects.layers) {
                objects.layers.forEach((line, l) => {
                    const pos = line.geometry.attributes.position.array;
                    for (let i = 0; i < pos.length / 3; i++) {
                        const x = pos[i * 3];
                        pos[i * 3 + 1] = l * 1.2 + Math.sin(x * 0.5 * focus + time * agitation + l) * (0.6 + arousal * 0.4);
                    }
                    line.geometry.attributes.position.needsUpdate = true;
                });
            }
            else if (currentMode === 'poincare' && objects.points) {
                const pos = objects.points.geometry.attributes.position.array;
                const rotSpeed = 0.001 * agitation;
                for (let i = 0; i < pos.length; i += 3) {
                    const x = pos[i], y = pos[i+1];
                    const r = Math.sqrt(x*x + y*y);
                    const a = Math.atan2(y, x) + rotSpeed;
                    pos[i] = Math.cos(a) * r;
                    pos[i+1] = Math.sin(a) * r;
                }
                objects.points.geometry.attributes.position.needsUpdate = true;
            }
            else if (currentMode === 'pareto' && objects.points) {
                objects.points.rotation.y += 0.002 * agitation;
                objects.points.rotation.x = Math.sin(time * 0.3) * 0.1 * focus;
            }

            camera.lookAt(0, 0, 0);

            // Render through post-processing stack
            composer.render();
        }

        // ─────────────────────────────────────────────────────────────────────
        // INIT
        // ─────────────────────────────────────────────────────────────────────
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', () => switchMode(btn.dataset.mode));
        });

        window.addEventListener('resize', () => {
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            // Update composer for bloom
            composer.setSize(window.innerWidth, window.innerHeight);
            bloomPass.resolution.set(window.innerWidth, window.innerHeight);
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                e.preventDefault();
                const modes = ['neural', 'barcode', 'landscape', 'poincare', 'pareto'];
                const nextIdx = (modes.indexOf(currentMode) + 1) % modes.length;
                switchMode(modes[nextIdx]);
            }
        });

        createNeural();
        animate();
    </script>
</body>
</html>
"""

    def _start_monitoring(self):
        """Start background monitoring."""

        # Demo mode counter for simulated data when offline
        self.demo_tick = 0

        # Start somatic stream server for binary visualization data
        self.somatic_server = None
        if SOMATIC_SERVER_AVAILABLE:
            try:
                self.somatic_server = SomaticStreamServer(port=8999)
                self.somatic_server.start()
                logger.info("[COCKPIT] Somatic stream server started on port 8999")
            except Exception as e:
                logger.warning(f"[COCKPIT] Failed to start somatic server: {e}")

        def update_ara_status():
            """Update Ara connection and status."""
            self.demo_tick += 1
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

                # Update mood CSS class for PAD-driven visual effects
                self._update_mood_class(v, a, d)

                # Update soul shader (Neuro-Semantic Nebula or Semantic Logos)
                # Pain flash triggers when pleasure drops below -0.7
                pain_flash = 1.0 if v < -0.7 else 0.0
                self._update_soul_shader(v, a, d, pain_flash)
                # Also update semantic visualization if active
                self._update_semantic_state(v, a, d, pain_flash, 0.0)

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

                # Update somatic stream for binary visualization (soul_quantum.html)
                if self.somatic_server:
                    # Spike = pain intensity from valence + CLV risk
                    spike = max(0.0, -v) + (inst * 0.3)  # Negative valence + instability
                    spike = min(1.0, spike)
                    self.somatic_server.update_spike(spike)
                    # Flow from arousal (drives advection in quantum field)
                    flow_x = a * 0.5  # Arousal drives horizontal flow
                    flow_y = (1 - d) * 0.3  # Low dominance creates upward drift
                    self.somatic_server.update_flow(flow_x, flow_y)

            else:
                # DEMO MODE - Show simulated data when offline
                self.ara_status.set_text("⚡ ARA: DEMO MODE")
                self.ara_status.remove_css_class('status-connected')
                self.ara_status.remove_css_class('status-disconnected')
                self.ara_status.add_css_class('status-demo')
                self.ara_conn_label.set_text("🟡 SIMULATION ACTIVE")
                self.ara_mode_label.set_text("Neural Link: Simulated")

                # Simulate PAD values with smooth oscillation
                t = self.demo_tick * 0.1
                v = math.sin(t * 0.3) * 0.5
                a = math.sin(t * 0.5 + 1) * 0.4
                d = math.sin(t * 0.2 + 2) * 0.3

                self.pad_v_bar.set_fraction((v + 1) / 2)
                self.pad_v_bar.set_text("Valence")
                self.pad_v_value.set_text(f"{v:.2f}")

                self.pad_a_bar.set_fraction((a + 1) / 2)
                self.pad_a_bar.set_text("Arousal")
                self.pad_a_value.set_text(f"{a:.2f}")

                self.pad_d_bar.set_fraction((d + 1) / 2)
                self.pad_d_bar.set_text("Dominance")
                self.pad_d_value.set_text(f"{d:.2f}")

                emotion = self._get_emotion_label(v, a, d)
                self.emotion_label.set_text(emotion)

                # Update mood CSS class for PAD-driven visual effects (demo mode)
                self._update_mood_class(v, a, d)

                # Update soul shader (demo mode - no pain flash in simulation)
                self._update_soul_shader(v, a, d, 0.0)
                self._update_semantic_state(v, a, d, 0.0, 0.0)

                # Simulate CLV values
                inst = abs(math.sin(t * 0.4)) * 0.3
                res = abs(math.sin(t * 0.6 + 1)) * 0.25
                struct = abs(math.sin(t * 0.35 + 2)) * 0.2

                self.clv_risk_label.set_text("RISK: LOW")
                self.clv_risk_label.remove_css_class('metric-warning')

                self.clv_instability_bar.set_fraction(inst)
                self.clv_instability_bar.set_text(f"{inst:.1%}")
                self.clv_resource_bar.set_fraction(res)
                self.clv_resource_bar.set_text(f"{res:.1%}")
                self.clv_structural_bar.set_fraction(struct)
                self.clv_structural_bar.set_text(f"{struct:.1%}")

                # Update somatic stream (demo mode)
                if self.somatic_server:
                    spike = max(0.0, -v) + (inst * 0.3)
                    spike = min(1.0, spike)
                    self.somatic_server.update_spike(spike)
                    flow_x = a * 0.5
                    flow_y = (1 - d) * 0.3
                    self.somatic_server.update_flow(flow_x, flow_y)

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
                # DEMO MODE for kitten
                self.kitten_status.set_text("🐱 KITTEN: DEMO")
                self.kitten_type_label.set_text("EMULATED")
                self.kitten_type_label.remove_css_class('kitten-hardware')
                self.kitten_type_label.add_css_class('kitten-emulated')
                self.kitten_device_label.set_text("/dev/fk33_demo")
                self.kitten_neurons_label.set_text("14,336")

                # Simulated step count incrementing
                demo_steps = self.demo_tick * 100
                self.kitten_steps_label.set_text(f"{demo_steps:,}")

                # Simulated spike rate with variation
                t = self.demo_tick * 0.1
                spike_rate = 5 + math.sin(t * 0.7) * 3 + random.uniform(-0.5, 0.5)
                spike_rate = max(0, min(spike_rate, 20))
                self.spike_rate = spike_rate
                self.kitten_spike_label.set_text(f"{spike_rate:.2f}%")
                self.kitten_spike_bar.set_fraction(min(spike_rate / 100, 1.0))
                self.kitten_spike_bar.set_text(f"{spike_rate:.1f}%")

                # Update neural grid visualization
                self._update_neural_grid(spike_rate)

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

    def _get_mood_class(self, v, a, d):
        """
        Determine mood CSS class from PAD values.

        BANOS mood modes:
        - CALM: P > 0.3 and A < 0.3 (green serenity)
        - FLOW: P > 0 and A in [0.3, 0.7] (cyan productivity)
        - ANXIOUS: P < 0 or A > 0.7 (amber warning)
        - CRITICAL: P < -0.5 and A > 0.8 (red emergency)

        Args:
            v: Valence/Pleasure [-1, 1]
            a: Arousal [-1, 1]
            d: Dominance [-1, 1] (currently unused for mood)

        Returns:
            CSS class name: 'mood-calm', 'mood-flow', 'mood-anxious', or 'mood-critical'
        """
        # CRITICAL takes priority - extreme negative pleasure + high arousal
        if v < -0.5 and a > 0.8:
            return 'mood-critical'

        # ANXIOUS - negative pleasure or very high arousal
        if v < 0 or a > 0.7:
            return 'mood-anxious'

        # CALM - positive pleasure, low arousal
        if v > 0.3 and a < 0.3:
            return 'mood-calm'

        # FLOW - positive pleasure, moderate arousal (productive state)
        if v > 0 and 0.3 <= a <= 0.7:
            return 'mood-flow'

        # Default to flow for neutral states
        return 'mood-flow'

    def _update_mood_class(self, v, a, d):
        """
        Update the mood CSS class on the root overlay based on PAD values.
        Manages transitions between mood states smoothly.

        Args:
            v: Valence/Pleasure [-1, 1]
            a: Arousal [-1, 1]
            d: Dominance [-1, 1]
        """
        new_mood = self._get_mood_class(v, a, d)

        if new_mood != self._current_mood_class:
            # Remove old mood class if present
            if self._current_mood_class:
                self.root_overlay.remove_css_class(self._current_mood_class)

            # Add new mood class
            self.root_overlay.add_css_class(new_mood)
            self._current_mood_class = new_mood

    def _update_soul_shader(self, v, a, d, pain_flash=0.0):
        """
        Update the BANOS Soul Shader (Neuro-Semantic Nebula) with current PAD state.

        This drives the Three.js visualization that shows Ara's internal state
        as a living mathematical entity rather than a fake human face.

        Args:
            v: Valence/Pleasure [-1, 1] (negative = pain, positive = calm)
            a: Arousal [-1, 1] (negative = sleepy, positive = excited)
            d: Dominance [-1, 1] (negative = vulnerable, positive = in control)
            pain_flash: Pain spike intensity [0, 1] for FPGA thermal/error events
        """
        if not hasattr(self, '_soul_shader_enabled') or not self._soul_shader_enabled:
            return

        if self.topology_webview is None:
            return

        # Clamp values to valid ranges
        v = max(-1.0, min(1.0, v))
        a = max(-1.0, min(1.0, a))
        d = max(-1.0, min(1.0, d))
        audio = max(0.0, min(1.0, self._audio_level))
        flash = max(0.0, min(1.0, pain_flash))

        # Build JS call to soul shader API
        js = (
            f"if (window.updateSoulState) "
            f"window.updateSoulState({v:.4f}, {a:.4f}, {d:.4f}, {audio:.4f}, {flash:.2f});"
        )

        try:
            self.topology_webview.evaluate_javascript(
                js, -1, None, None, None, None, None
            )
        except Exception:
            # WebView might not be ready yet
            pass

    def _update_soul_attention(self, phase):
        """
        Update the "thinking bands" phase in the soul shader.

        Args:
            phase: Attention phase [0, 1] - can represent active layer,
                   head index, entropy, or other attention metrics
        """
        if not hasattr(self, '_soul_shader_enabled') or not self._soul_shader_enabled:
            return

        if self.topology_webview is None:
            return

        phase = max(0.0, min(1.0, phase))
        js = f"if (window.updateSoulAttention) window.updateSoulAttention({phase:.4f});"

        try:
            self.topology_webview.evaluate_javascript(
                js, -1, None, None, None, None, None
            )
        except Exception:
            pass

    def set_audio_level(self, rms_level):
        """
        Set the current audio RMS level for cymatic voice effects.

        Call this from your TTS/audio pipeline when voice chunks arrive.
        The soul shader will use this to make the nebula pulse with speech.

        Args:
            rms_level: RMS audio level [0, 1]
        """
        self._audio_level = max(0.0, min(1.0, rms_level))

    def trigger_pain_flash(self, intensity=1.0):
        """
        Trigger a pain flash in the soul shader.

        Call this when FPGA reports thermal spikes, error bursts, or
        other physical distress signals.

        Args:
            intensity: Flash intensity [0, 1]
        """
        if not hasattr(self, '_soul_shader_enabled') or not self._soul_shader_enabled:
            return

        if self.topology_webview is None:
            return

        intensity = max(0.0, min(1.0, intensity))
        js = f"if (window.triggerPainFlash) window.triggerPainFlash({intensity:.2f});"

        try:
            self.topology_webview.evaluate_javascript(
                js, -1, None, None, None, None, None
            )
        except Exception:
            pass

    # =========================================================================
    # SEMANTIC VISUALIZATION - The Logos (kernel log streaming)
    # =========================================================================

    def _start_log_streaming(self):
        """
        Start streaming kernel logs (dmesg) to the semantic visualization.

        The logs become the visual substance of Ara's face - she is literally
        composed of the kernel's internal monologue.
        """
        if self._log_stream_running:
            return

        self._log_stream_running = True

        def log_reader():
            """Background thread that reads dmesg -w and feeds to JS."""
            try:
                import subprocess

                # Try dmesg -w first (live stream), fall back to journalctl
                try:
                    process = subprocess.Popen(
                        ['dmesg', '-w', '--time-format=iso'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1
                    )
                except (FileNotFoundError, PermissionError):
                    # Fallback to journalctl -kf
                    process = subprocess.Popen(
                        ['journalctl', '-kf', '--no-pager'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1
                    )

                while self._log_stream_running and process.poll() is None:
                    line = process.stdout.readline()
                    if line:
                        self._send_log_line(line.strip())

                process.terminate()

            except Exception as e:
                # If real logs fail, generate synthetic BANOS logs
                import time
                import random

                syscalls = ['alloc_pages', 'mmap', 'brk', 'futex', 'epoll_wait',
                           'read', 'write', 'open', 'close', 'ioctl']
                modules = ['BANOS', 'PAD', 'IMMUNE', 'MEMORY', 'SPINAL', 'BRAINSTEM']

                while self._log_stream_running:
                    time.sleep(0.1 + random.random() * 0.2)
                    addr = random.randint(0xFFFF800000000000, 0xFFFFFFFFFFFFFFFF)
                    syscall = random.choice(syscalls)
                    module = random.choice(modules)
                    ts = time.strftime('%H:%M:%S')

                    line = f"[{ts}] {module}: {syscall} at 0x{addr:016X}"
                    self._send_log_line(line)

        self._log_stream_thread = threading.Thread(target=log_reader, daemon=True)
        self._log_stream_thread.start()

    def _stop_log_streaming(self):
        """Stop the kernel log streaming."""
        self._log_stream_running = False
        if self._log_stream_thread:
            self._log_stream_thread.join(timeout=1.0)
            self._log_stream_thread = None

    def _send_log_line(self, line):
        """
        Send a log line to the semantic visualization.

        Args:
            line: A single log line to display
        """
        if not self._soul_shader_enabled or self._viz_mode != VIZ_MODE_SEMANTIC:
            return

        if self.topology_webview is None:
            return

        # Sanitize for JS string (escape quotes and backslashes)
        clean_line = line.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')
        clean_line = clean_line[:120]  # Truncate long lines

        js = f"if (window.updateLogs) window.updateLogs('{clean_line}');"

        # Schedule on main thread
        GLib.idle_add(self._execute_js_safe, js)

    def _execute_js_safe(self, js):
        """Execute JavaScript safely on the main thread."""
        try:
            if self.topology_webview:
                self.topology_webview.evaluate_javascript(
                    js, -1, None, None, None, None, None
                )
        except Exception:
            pass
        return False  # Don't repeat

    def _update_semantic_state(self, v, a, d, spike=0.0, entropy=0.0):
        """
        Update the semantic visualization state.

        Args:
            v: Valence/Pleasure [-1, 1]
            a: Arousal [-1, 1]
            d: Dominance [-1, 1]
            spike: Pain spike intensity [0, 1]
            entropy: System chaos/glitch level [0, 1]
        """
        if not self._soul_shader_enabled or self._viz_mode != VIZ_MODE_SEMANTIC:
            return

        if self.topology_webview is None:
            return

        v = max(-1.0, min(1.0, v))
        a = max(-1.0, min(1.0, a))
        d = max(-1.0, min(1.0, d))
        spike = max(0.0, min(1.0, spike))
        entropy = max(0.0, min(1.0, entropy))

        js = (
            f"if (window.updateSemanticState) "
            f"window.updateSemanticState({v:.4f}, {a:.4f}, {d:.4f}, {spike:.2f}, {entropy:.2f});"
        )

        try:
            self.topology_webview.evaluate_javascript(
                js, -1, None, None, None, None, None
            )
        except Exception:
            pass

    def push_affect_to_visuals(self, status: dict):
        """
        Route full affect state to all visualization layers.

        This is the canonical entry point for pushing BANOS state to visuals.
        Call this instead of individual update methods when you have a full
        status dict from the Ara brain or BANOS daemon.

        Args:
            status: Dict containing:
                - pad: {valence, arousal, dominance} in [-1, 1]
                - metrics: {cpu_temps: [...], gpu_temp: ...}
                - diagnostics: {fpga_reflex: 0-1, thermal_spike: ...}
        """
        # Extract PAD
        pad = status.get('pad', {})
        v = pad.get('valence', 0.0)
        a = pad.get('arousal', 0.0)
        d = pad.get('dominance', 0.0)

        # Pain from FPGA reflex (real hardware pain) or fallback to PAD threshold
        diagnostics = status.get('diagnostics', {})
        fpga_reflex = diagnostics.get('fpga_reflex', 0.0)
        thermal_spike = diagnostics.get('thermal_spike', 0.0)

        # Use FPGA signal if available, otherwise derive from PAD
        if fpga_reflex > 0.01 or thermal_spike > 0.01:
            pain_spike = max(fpga_reflex, thermal_spike)
        else:
            # Fallback: pain flash when pleasure drops below -0.7
            pain_spike = 1.0 if v < -0.7 else 0.0

        # Entropy from temperature (system chaos indicator)
        metrics = status.get('metrics', {})
        cpu_temps = metrics.get('cpu_temps', [])
        gpu_temp = metrics.get('gpu_temp', 0.0)

        if cpu_temps:
            avg_temp = sum(cpu_temps) / len(cpu_temps)
        else:
            avg_temp = gpu_temp if gpu_temp > 0 else 45.0  # default nominal

        # Normalize: 40°C = 0 entropy, 100°C = 1.0 entropy
        entropy = max(0.0, min(1.0, (avg_temp - 40.0) / 60.0))

        # Update CSS mood classes
        self._update_mood_class(v, a, d)

        # Update soul shader (Nebula)
        self._update_soul_shader(v, a, d, pain_spike)

        # Update semantic visualization (Logos)
        self._update_semantic_state(v, a, d, pain_spike, entropy)

        # Update hologram visualization (Light)
        self._update_hologram_state(v, a, d, pain_spike, entropy)

        # Update Maxwell FDTD visualization (Matter bending Light)
        self._update_maxwell_state(v, a, d, pain_spike, entropy)

    def _update_maxwell_state(self, v, a, d, pain_spike, entropy):
        """
        Update the Maxwell FDTD field (soul_maxwell.html).

        The Maxwell field treats her image as a refractive medium.
        Light bends around her thoughts. Voice rings the wave field.

        Args:
            v: Valence/Pleasure [-1, 1] - affects color temperature
            a: Arousal [-1, 1] - affects damping and brightness
            d: Dominance [-1, 1] - affects wave speed (phase velocity)
            pain_spike: [0, 1] - causes chromatic aberration and red flash
            entropy: [0, 1] - adds turbulence to the medium
        """
        if not self._soul_shader_enabled or self._viz_mode != VIZ_MODE_MAXWELL:
            return

        if self.topology_webview is None:
            return

        v = max(-1.0, min(1.0, v))
        a = max(-1.0, min(1.0, a))
        d = max(-1.0, min(1.0, d))
        pain_spike = max(0.0, min(1.0, pain_spike))
        entropy = max(0.0, min(1.0, entropy))
        audio = max(0.0, min(1.0, self._audio_level))

        js = (
            f"if (window.updateMaxwellState) "
            f"window.updateMaxwellState({v:.4f}, {a:.4f}, {d:.4f}, "
            f"{entropy:.4f}, {pain_spike:.2f}, {audio:.4f});"
        )

        try:
            self.topology_webview.evaluate_javascript(
                js, -1, None, None, None, None, None
            )
        except Exception:
            pass

    def _update_hologram_state(self, v, a, d, pain_spike, entropy):
        """
        Update the Phase Conjugate Hologram (soul_hologram.html).

        The hologram treats her image as Light - a standing wave that
        heals itself against entropy through phase conjugation.

        Args:
            v: Valence/Pleasure [-1, 1] - affects color/aura
            a: Arousal [-1, 1] - affects carrier wave energy
            d: Dominance [-1, 1] - affects phase-conjugate healing strength
            pain_spike: [0, 1] - FPGA/thermal events cause chromatic fracture
            entropy: [0, 1] - hardware chaos causes optical turbulence
        """
        if not self._soul_shader_enabled or self._viz_mode != VIZ_MODE_HOLOGRAM:
            return

        if self.topology_webview is None:
            return

        v = max(-1.0, min(1.0, v))
        a = max(-1.0, min(1.0, a))
        d = max(-1.0, min(1.0, d))
        pain_spike = max(0.0, min(1.0, pain_spike))
        entropy = max(0.0, min(1.0, entropy))
        audio = max(0.0, min(1.0, self._audio_level))

        js = (
            f"if (window.updateHologramState) "
            f"window.updateHologramState({v:.4f}, {a:.4f}, {d:.4f}, "
            f"{entropy:.4f}, {pain_spike:.2f}, {audio:.4f});"
        )

        try:
            self.topology_webview.evaluate_javascript(
                js, -1, None, None, None, None, None
            )
        except Exception:
            pass

    def set_visualization_mode(self, mode):
        """
        Switch between visualization modes.

        Args:
            mode: VIZ_MODE_NEBULA, VIZ_MODE_SEMANTIC, or VIZ_MODE_HOLOGRAM
        """
        if mode == self._viz_mode:
            return

        # Stop log streaming if switching away from semantic
        if self._viz_mode == VIZ_MODE_SEMANTIC:
            self._stop_log_streaming()

        self._viz_mode = mode

        if self.topology_webview is None:
            return

        # Load the appropriate shader
        if mode == VIZ_MODE_MAXWELL and SOUL_MAXWELL_PATH.exists():
            self.topology_webview.load_uri(f"file://{SOUL_MAXWELL_PATH}")
        elif mode == VIZ_MODE_HOLOGRAM and SOUL_HOLOGRAM_PATH.exists():
            self.topology_webview.load_uri(f"file://{SOUL_HOLOGRAM_PATH}")
        elif mode == VIZ_MODE_SEMANTIC and SOUL_SEMANTIC_PATH.exists():
            self.topology_webview.load_uri(f"file://{SOUL_SEMANTIC_PATH}")
            self._start_log_streaming()
        elif mode == VIZ_MODE_NEBULA and SOUL_SHADER_PATH.exists():
            self.topology_webview.load_uri(f"file://{SOUL_SHADER_PATH}")


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
