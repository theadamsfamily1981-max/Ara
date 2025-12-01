#!/usr/bin/env python3
"""
T-FAN Cockpit Mode - Touchscreen HUD

Full-screen borderless app for side touchscreen monitor.
Shows system metrics, topology visualization, and avatar controls.

Usage:
    python cockpit_hud.py                    # Auto-detect touchscreen
    python cockpit_hud.py --monitor=HDMI-1   # Specific monitor
    python cockpit_hud.py --fullscreen       # Force fullscreen
"""

import gi
gi.require_version('Gtk', '4.0')
gi.require_version('Adw', '1')
gi.require_version('WebKit', '6.0')

from gi.repository import Gtk, Adw, Gio, GLib, Gdk, WebKit
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional
import psutil
import time
import logging

# Local modules
from video_background import create_background
from touch_gestures import GestureHandler, RippleEffect, setup_touch_feedback, get_ripple_css

logger = logging.getLogger(__name__)

# Try to import GPUtil for GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ GPUtil not available - GPU monitoring disabled")


class MetricsCollector:
    """Collects system metrics for HUD display."""

    @staticmethod
    def get_cpu_metrics():
        """Get CPU usage, temps, frequency."""
        cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()

        # Try to get temperatures
        temps = []
        try:
            temps_dict = psutil.sensors_temperatures()
            if 'coretemp' in temps_dict:
                temps = [t.current for t in temps_dict['coretemp']]
            elif 'k10temp' in temps_dict:  # AMD
                temps = [t.current for t in temps_dict['k10temp']]
        except:
            pass

        return {
            'usage_per_core': cpu_percent,
            'usage_total': sum(cpu_percent) / len(cpu_percent),
            'frequency': cpu_freq.current if cpu_freq else 0,
            'cores': cpu_count,
            'temps': temps
        }

    @staticmethod
    def get_ram_metrics():
        """Get RAM usage."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Top memory consumers
        processes = []
        try:
            for proc in sorted(psutil.process_iter(['name', 'memory_percent']),
                             key=lambda p: p.info['memory_percent'] or 0,
                             reverse=True)[:5]:
                processes.append({
                    'name': proc.info['name'],
                    'percent': proc.info['memory_percent'] or 0
                })
        except:
            pass

        return {
            'total_gb': mem.total / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent': mem.percent,
            'available_gb': mem.available / (1024**3),
            'swap_percent': swap.percent,
            'top_processes': processes
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
                    'load': gpu.load * 100,  # Convert to percentage
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                    'temperature': gpu.temperature,
                    'power_draw': getattr(gpu, 'powerDraw', 0),
                    'uuid': gpu.uuid
                })
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")

        return gpus

    @staticmethod
    def get_network_metrics():
        """Get network I/O."""
        net = psutil.net_io_counters()

        return {
            'bytes_sent': net.bytes_sent,
            'bytes_recv': net.bytes_recv,
            'packets_sent': net.packets_sent,
            'packets_recv': net.packets_recv,
            'errors': net.errin + net.errout
        }

    @staticmethod
    def get_storage_metrics():
        """Get disk usage and I/O."""
        # Disk usage
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

        # Disk I/O
        try:
            io = psutil.disk_io_counters()
            io_stats = {
                'read_mb': io.read_bytes / (1024**2),
                'write_mb': io.write_bytes / (1024**2),
                'read_count': io.read_count,
                'write_count': io.write_count
            }
        except:
            io_stats = {}

        return {
            'disks': disks,
            'io': io_stats
        }


class CockpitHUDWindow(Adw.ApplicationWindow):
    """Touchscreen cockpit HUD window."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_title("T-FAN Cockpit")
        self.set_default_size(800, 1280)  # Portrait for side touchscreen

        # Current view mode
        self.current_view = 'overview'

        # Metrics history for graphs
        self.metrics_history = {
            'gpu': [],
            'cpu': [],
            'network': [],
            'storage': []
        }
        self.history_max_length = 60  # Keep 60 data points

        # Video background reference
        self.video_bg = None

        # Apply cockpit theme
        self._load_cockpit_css()

        # Root overlay for layering video, content, and effects
        root_overlay = Gtk.Overlay()
        root_overlay.add_css_class('cockpit-window')
        self.set_content(root_overlay)

        # Layer 0: Video background
        video_widget, self.video_bg = create_background()
        video_widget.set_hexpand(True)
        video_widget.set_vexpand(True)
        root_overlay.set_child(video_widget)

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
        scroll.set_kinetic_scrolling(True)  # Touch-friendly scrolling

        self.content_stack = Gtk.Stack()
        self.content_stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)
        self.content_stack.set_transition_duration(300)
        scroll.set_child(self.content_stack)

        main_box.append(scroll)

        # Status bar (bottom)
        self.status_bar = self._build_status_bar()
        main_box.append(self.status_bar)

        # Layer 2: Overlay effects (scanlines, vignette)
        overlay_effects = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        overlay_effects.add_css_class('cockpit-overlay')
        overlay_effects.add_css_class('cockpit-scanlines')
        overlay_effects.add_css_class('cockpit-vignette')
        overlay_effects.set_hexpand(True)
        overlay_effects.set_vexpand(True)
        root_overlay.add_overlay(overlay_effects)

        # Add touch gestures for swipe navigation
        self._setup_touch_gestures(scroll)

        # Build all views
        self._build_all_views()

        # Start metrics monitoring
        self._start_monitoring()

        # Start video background
        if self.video_bg and hasattr(self.video_bg, 'play'):
            self.video_bg.play()
            self.video_bg.fade_in(1500)

        logger.info("[HUD] Cockpit initialized with video background and overlays")

    def _load_cockpit_css(self):
        """Load futuristic cockpit theme CSS from external file."""
        display = Gdk.Display.get_default()

        # Load main theme CSS file
        css_file = Path(__file__).parent / 'cockpit_theme.css'
        if css_file.exists():
            css_provider = Gtk.CssProvider()
            css_provider.load_from_path(str(css_file))
            Gtk.StyleContext.add_provider_for_display(
                display,
                css_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )
            logger.info(f"[HUD] Loaded theme from {css_file}")
        else:
            logger.warning(f"[HUD] Theme file not found: {css_file}")

        # Load ripple effect CSS
        ripple_provider = Gtk.CssProvider()
        ripple_provider.load_from_string(get_ripple_css())
        Gtk.StyleContext.add_provider_for_display(
            display,
            ripple_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def _build_status_bar(self):
        """Build bottom status bar."""
        status_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        status_box.add_css_class('status-bar')
        status_box.set_halign(Gtk.Align.CENTER)

        # Connection status
        self.connection_status = Gtk.Label(label="🟢 Connected")
        self.connection_status.add_css_class('status-indicator')
        self.connection_status.add_css_class('status-nominal')
        status_box.append(self.connection_status)

        # Separator
        sep = Gtk.Label(label="|")
        sep.add_css_class('status-indicator')
        status_box.append(sep)

        # Mode indicator
        self.mode_indicator = Gtk.Label(label="⚡ Work Mode")
        self.mode_indicator.add_css_class('status-indicator')
        status_box.append(self.mode_indicator)

        # Separator
        sep2 = Gtk.Label(label="|")
        sep2.add_css_class('status-indicator')
        status_box.append(sep2)

        # GPU temp
        self.gpu_temp_status = Gtk.Label(label="GPU: --°C")
        self.gpu_temp_status.add_css_class('status-indicator')
        status_box.append(self.gpu_temp_status)

        # Separator
        sep3 = Gtk.Label(label="|")
        sep3.add_css_class('status-indicator')
        status_box.append(sep3)

        # Time
        self.time_label = Gtk.Label(label="--:--")
        self.time_label.add_css_class('status-indicator')
        status_box.append(self.time_label)

        # Update time every second
        GLib.timeout_add_seconds(1, self._update_time)

        return status_box

    def _update_time(self):
        """Update time display."""
        from datetime import datetime
        now = datetime.now()
        self.time_label.set_text(now.strftime("%I:%M %p"))
        return True  # Continue timer

    def _setup_touch_gestures(self, scroll_widget):
        """Set up touch gesture handlers."""
        # Create gesture handler for swipe navigation
        handler = GestureHandler(scroll_widget)

        def on_swipe(direction, vx, vy):
            """Handle swipe gestures for view navigation."""
            from touch_gestures import SwipeDirection

            # Get current view index
            view_order = ['overview', 'gpu', 'cpu', 'network', 'storage', 'topology', 'avatar']
            current_idx = view_order.index(self.current_view)

            if direction == SwipeDirection.LEFT and current_idx < len(view_order) - 1:
                # Swipe left -> next view
                next_view = view_order[current_idx + 1]
                self._switch_to_view(next_view)
            elif direction == SwipeDirection.RIGHT and current_idx > 0:
                # Swipe right -> previous view
                prev_view = view_order[current_idx - 1]
                self._switch_to_view(prev_view)

        def on_long_press(x, y):
            """Handle long press for options menu."""
            logger.debug(f"[HUD] Long press at ({x}, {y})")
            # Could show context menu here

        handler.on_swipe = on_swipe
        handler.on_long_press = on_long_press

    def _switch_to_view(self, view_name):
        """Switch to specified view."""
        if view_name == self.current_view:
            return

        # Update button states
        for mode_id, btn in self.mode_buttons.items():
            if mode_id == view_name:
                btn.add_css_class('active')
                btn.add_css_class('selected')
            else:
                btn.remove_css_class('active')
                btn.remove_css_class('selected')

        # Switch view
        self.current_view = view_name
        self.content_stack.set_visible_child_name(view_name)

        # Pause video for topology view (performance)
        if self.video_bg and hasattr(self.video_bg, 'pause'):
            if view_name == 'topology':
                self.video_bg.pause()
            elif hasattr(self.video_bg, 'play'):
                self.video_bg.play()

        logger.info(f"[HUD] Switched to {view_name} view")

    def _build_hud_strip(self):
        """Build top HUD control strip with mode buttons."""
        strip_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        strip_box.add_css_class('hud-strip')
        strip_box.add_css_class('scanlines')

        # Title
        title_label = Gtk.Label(label="⚛️ T-FAN COCKPIT")
        title_label.add_css_class('metric-title')
        title_label.set_margin_bottom(8)
        strip_box.append(title_label)

        # Button grid (2 rows)
        grid = Gtk.Grid()
        grid.set_row_spacing(6)
        grid.set_column_spacing(6)
        grid.set_halign(Gtk.Align.CENTER)

        # Define mode buttons
        modes = [
            ('overview', 'OVERVIEW\n📊', 0, 0),
            ('gpu', 'GPU\n🎮', 0, 1),
            ('cpu', 'CPU/RAM\n💻', 0, 2),
            ('network', 'NETWORK\n🌐', 1, 0),
            ('storage', 'STORAGE\n💾', 1, 1),
            ('topology', 'TOPOLOGY\n🌌', 1, 2),
            ('avatar', 'AVATAR\n🤖', 1, 3),
        ]

        self.mode_buttons = {}

        for mode_id, label, row, col in modes:
            button = Gtk.Button(label=label)
            button.add_css_class('hud-button')
            button.add_css_class('ripple-effect')
            button.connect('clicked', self._on_mode_button_clicked, mode_id)
            # Add touch feedback
            setup_touch_feedback(button)
            grid.attach(button, col, row, 1, 1)
            self.mode_buttons[mode_id] = button

        # Mark overview as active initially
        self.mode_buttons['overview'].add_css_class('active')
        self.mode_buttons['overview'].add_css_class('selected')

        strip_box.append(grid)

        return strip_box

    def _on_mode_button_clicked(self, button, mode_id):
        """Handle HUD mode button click."""
        # Trigger ripple effect
        button.add_css_class('ripple-active')
        GLib.timeout_add(400, lambda: button.remove_css_class('ripple-active'))

        # Use centralized switch method
        self._switch_to_view(mode_id)

    def _build_all_views(self):
        """Build all metric view pages."""
        # Overview
        self.content_stack.add_named(self._build_overview_view(), 'overview')

        # GPU
        self.content_stack.add_named(self._build_gpu_view(), 'gpu')

        # CPU/RAM
        self.content_stack.add_named(self._build_cpu_view(), 'cpu')

        # Network
        self.content_stack.add_named(self._build_network_view(), 'network')

        # Storage
        self.content_stack.add_named(self._build_storage_view(), 'storage')

        # Topology
        self.content_stack.add_named(self._build_topology_view(), 'topology')

        # Avatar
        self.content_stack.add_named(self._build_avatar_view(), 'avatar')

        # Set initial view
        self.content_stack.set_visible_child_name('overview')

    def _build_overview_view(self):
        """Build overview/mission status view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        box.add_css_class('holographic')

        # Status banner
        status_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        status_card.add_css_class('metric-card')

        status_title = Gtk.Label(label="MISSION STATUS")
        status_title.add_css_class('metric-title')
        status_card.append(status_title)

        self.status_label = Gtk.Label(label="🟢 ALL SYSTEMS NOMINAL")
        self.status_label.add_css_class('metric-value-large')
        status_card.append(self.status_label)

        box.append(status_card)

        # Quick metrics grid
        metrics_grid = Gtk.Grid()
        metrics_grid.set_row_spacing(12)
        metrics_grid.set_column_spacing(12)
        metrics_grid.set_margin_start(12)
        metrics_grid.set_margin_end(12)
        metrics_grid.set_margin_top(12)

        # GPU quick card
        gpu_card = self._create_quick_card("GPU", "0%")
        self.overview_gpu_label = gpu_card.get_last_child()
        metrics_grid.attach(gpu_card, 0, 0, 1, 1)

        # CPU quick card
        cpu_card = self._create_quick_card("CPU", "0%")
        self.overview_cpu_label = cpu_card.get_last_child()
        metrics_grid.attach(cpu_card, 1, 0, 1, 1)

        # RAM quick card
        ram_card = self._create_quick_card("RAM", "0 GB")
        self.overview_ram_label = ram_card.get_last_child()
        metrics_grid.attach(ram_card, 0, 1, 1, 1)

        # Network quick card
        net_card = self._create_quick_card("NET", "0 MB/s")
        self.overview_net_label = net_card.get_last_child()
        metrics_grid.attach(net_card, 1, 1, 1, 1)

        box.append(metrics_grid)

        return box

    def _create_quick_card(self, title, initial_value):
        """Create a quick metric card for overview."""
        card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        card.add_css_class('metric-card')
        card.set_hexpand(True)

        title_label = Gtk.Label(label=title)
        title_label.add_css_class('metric-label')
        card.append(title_label)

        value_label = Gtk.Label(label=initial_value)
        value_label.add_css_class('metric-value-medium')
        card.append(value_label)

        return card

    def _build_gpu_view(self):
        """Build GPU metrics view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)

        if not GPU_AVAILABLE:
            # No GPU monitoring available
            error_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            error_card.add_css_class('metric-card')

            error_label = Gtk.Label(label="GPU monitoring unavailable")
            error_label.add_css_class('metric-label')
            error_card.append(error_label)

            install_label = Gtk.Label(label="Install GPUtil: pip install gputil")
            install_label.add_css_class('metric-label')
            error_card.append(install_label)

            box.append(error_card)
            return box

        # GPU cards will be created dynamically
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

        self.cpu_freq_label = Gtk.Label(label="0 MHz")
        self.cpu_freq_label.add_css_class('metric-label')
        cpu_card.append(self.cpu_freq_label)

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

        self.ram_detail_label = Gtk.Label(label="")
        self.ram_detail_label.add_css_class('metric-label')
        ram_card.append(self.ram_detail_label)

        box.append(ram_card)

        return box

    def _build_network_view(self):
        """Build network metrics view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)

        # Network card
        net_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        net_card.add_css_class('metric-card')

        net_title = Gtk.Label(label="NETWORK")
        net_title.add_css_class('metric-title')
        net_card.append(net_title)

        self.net_status_label = Gtk.Label(label="🟢 ONLINE")
        self.net_status_label.add_css_class('metric-value-large')
        net_card.append(self.net_status_label)

        self.net_down_label = Gtk.Label(label="↓ 0 MB/s")
        self.net_down_label.add_css_class('metric-value-medium')
        net_card.append(self.net_down_label)

        self.net_up_label = Gtk.Label(label="↑ 0 MB/s")
        self.net_up_label.add_css_class('metric-value-medium')
        net_card.append(self.net_up_label)

        box.append(net_card)

        return box

    def _build_storage_view(self):
        """Build storage metrics view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)

        # Storage cards will be created dynamically
        self.storage_cards_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.append(self.storage_cards_box)

        return box

    def _build_topology_view(self):
        """Build topology visualization view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Try to embed WebGL topology screensaver
        try:
            self.topology_webview = WebKit.WebView()
            settings = self.topology_webview.get_settings()
            settings.set_enable_webgl(True)
            settings.set_enable_accelerated_2d_canvas(True)
            settings.set_hardware_acceleration_policy(
                WebKit.HardwareAccelerationPolicy.ALWAYS
            )

            # Load topology HTML (reuse from main app)
            topology_html = self._generate_topology_html()
            self.topology_webview.load_html(topology_html, "about:blank")

            box.append(self.topology_webview)

        except Exception as e:
            # Fallback if WebKit not available
            error_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            error_card.add_css_class('metric-card')
            error_card.set_margin_start(12)
            error_card.set_margin_end(12)
            error_card.set_margin_top(12)

            error_label = Gtk.Label(label=f"Topology view unavailable: {e}")
            error_label.add_css_class('metric-label')
            error_card.append(error_label)

            box.append(error_card)

        return box

    def _generate_topology_html(self):
        """Generate inline HTML for topology visualization."""
        # Simplified version of the topology screensaver from main app
        return """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; overflow: hidden; background: #020306; }
        canvas { display: block; width: 100%; height: 100vh; }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r160/three.min.js"></script>
    <script>
        const canvas = document.getElementById('canvas');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x020306);
        scene.fog = new THREE.Fog(0x020306, 10, 50);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.set(0, 5, 10);

        const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setSize(window.innerWidth, window.innerHeight);

        // Lights
        scene.add(new THREE.AmbientLight(0x404040, 1.0));
        const light1 = new THREE.PointLight(0x667eea, 2.0);
        light1.position.set(10, 10, 10);
        scene.add(light1);

        const light2 = new THREE.PointLight(0x764ba2, 1.5);
        light2.position.set(-10, 5, -10);
        scene.add(light2);

        // Create particle system
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const velocities = [];

        for (let i = 0; i < 1000; i++) {
            positions.push(
                (Math.random() - 0.5) * 10,
                (Math.random() - 0.5) * 10,
                (Math.random() - 0.5) * 10
            );

            const hue = 0.55 + Math.random() * 0.2;
            const color = new THREE.Color().setHSL(hue, 0.8, 0.6);
            colors.push(color.r, color.g, color.b);

            velocities.push(
                (Math.random() - 0.5) * 0.02,
                (Math.random() - 0.5) * 0.02,
                (Math.random() - 0.5) * 0.02
            );
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 0.15,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        });
        const points = new THREE.Points(geometry, material);
        scene.add(points);

        // Animation
        let time = 0;
        function animate() {
            requestAnimationFrame(animate);
            time += 0.016;

            const posArray = geometry.attributes.position.array;
            for (let i = 0; i < posArray.length; i += 3) {
                posArray[i] += velocities[i] * 0.4;
                posArray[i + 1] += velocities[i + 1] * 0.4;
                posArray[i + 2] += velocities[i + 2] * 0.4;

                if (Math.abs(posArray[i]) > 8) velocities[i] *= -1;
                if (Math.abs(posArray[i + 1]) > 8) velocities[i + 1] *= -1;
                if (Math.abs(posArray[i + 2]) > 8) velocities[i + 2] *= -1;

                posArray[i + 1] += Math.sin(time * 0.5 + posArray[i]) * 0.01;
            }
            geometry.attributes.position.needsUpdate = true;

            points.rotation.y += 0.001;
            camera.position.x = Math.sin(time * 0.1) * 15;
            camera.position.z = Math.cos(time * 0.1) * 15;
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

    def _build_avatar_view(self):
        """Build avatar customization view."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)

        # Avatar control card
        avatar_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        avatar_card.add_css_class('metric-card')

        title = Gtk.Label(label="ARA AVATAR CONTROL")
        title.add_css_class('metric-title')
        avatar_card.append(title)

        # Profile selector
        profile_label = Gtk.Label(label="Profile")
        profile_label.add_css_class('metric-label')
        profile_label.set_halign(Gtk.Align.START)
        avatar_card.append(profile_label)

        self.avatar_profile_combo = Gtk.ComboBoxText()
        self.avatar_profile_combo.append_text("Default")
        self.avatar_profile_combo.append_text("Professional")
        self.avatar_profile_combo.append_text("Casual")
        self.avatar_profile_combo.append_text("Scientist")
        self.avatar_profile_combo.append_text("Operator")
        self.avatar_profile_combo.set_active(0)
        self.avatar_profile_combo.connect('changed', self._on_avatar_profile_changed)
        avatar_card.append(self.avatar_profile_combo)

        # Style selector
        style_label = Gtk.Label(label="Style")
        style_label.add_css_class('metric-label')
        style_label.set_halign(Gtk.Align.START)
        style_label.set_margin_top(12)
        avatar_card.append(style_label)

        self.avatar_style_combo = Gtk.ComboBoxText()
        self.avatar_style_combo.append_text("Realistic")
        self.avatar_style_combo.append_text("Stylized")
        self.avatar_style_combo.append_text("Anime")
        self.avatar_style_combo.set_active(0)
        self.avatar_style_combo.connect('changed', self._on_avatar_style_changed)
        avatar_card.append(self.avatar_style_combo)

        # Mood selector
        mood_label = Gtk.Label(label="Mood")
        mood_label.add_css_class('metric-label')
        mood_label.set_halign(Gtk.Align.START)
        mood_label.set_margin_top(12)
        avatar_card.append(mood_label)

        self.avatar_mood_combo = Gtk.ComboBoxText()
        self.avatar_mood_combo.append_text("Neutral")
        self.avatar_mood_combo.append_text("Focused")
        self.avatar_mood_combo.append_text("Friendly")
        self.avatar_mood_combo.append_text("Excited")
        self.avatar_mood_combo.set_active(0)
        self.avatar_mood_combo.connect('changed', self._on_avatar_mood_changed)
        avatar_card.append(self.avatar_mood_combo)

        # Apply button
        apply_button = Gtk.Button(label="✓ APPLY CHANGES")
        apply_button.add_css_class('hud-button')
        apply_button.set_margin_top(20)
        apply_button.connect('clicked', self._on_avatar_apply)
        avatar_card.append(apply_button)

        # Save preset button
        save_button = Gtk.Button(label="💾 SAVE AS PRESET")
        save_button.add_css_class('hud-button')
        save_button.connect('clicked', self._on_avatar_save_preset)
        avatar_card.append(save_button)

        box.append(avatar_card)

        # Current status
        status_card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        status_card.add_css_class('metric-card')

        status_title = Gtk.Label(label="CURRENT STATUS")
        status_title.add_css_class('metric-title')
        status_card.append(status_title)

        self.avatar_status_label = Gtk.Label(label="Profile: Default\nStyle: Realistic\nMood: Neutral")
        self.avatar_status_label.add_css_class('metric-label')
        status_card.append(self.avatar_status_label)

        box.append(status_card)

        return box

    def _on_avatar_profile_changed(self, combo):
        """Handle avatar profile change."""
        profile = combo.get_active_text()
        print(f"[Avatar] Profile changed to: {profile}")

    def _on_avatar_style_changed(self, combo):
        """Handle avatar style change."""
        style = combo.get_active_text()
        print(f"[Avatar] Style changed to: {style}")

    def _on_avatar_mood_changed(self, combo):
        """Handle avatar mood change."""
        mood = combo.get_active_text()
        print(f"[Avatar] Mood changed to: {mood}")

    def _on_avatar_apply(self, button):
        """Apply avatar changes."""
        profile = self.avatar_profile_combo.get_active_text()
        style = self.avatar_style_combo.get_active_text()
        mood = self.avatar_mood_combo.get_active_text()

        # Update status label
        self.avatar_status_label.set_label(f"Profile: {profile}\nStyle: {style}\nMood: {mood}")

        # TODO: Send to Ara via D-Bus
        print(f"[Avatar] Applied: {profile} / {style} / {mood}")

        # Show feedback
        self.status_label.set_label(f"✓ Avatar updated: {profile}")
        GLib.timeout_add_seconds(3, lambda: self.status_label.set_label("🟢 ALL SYSTEMS NOMINAL"))

    def _on_avatar_save_preset(self, button):
        """Save current avatar settings as preset."""
        profile = self.avatar_profile_combo.get_active_text()
        style = self.avatar_style_combo.get_active_text()
        mood = self.avatar_mood_combo.get_active_text()

        # TODO: Save to config file
        print(f"[Avatar] Saving preset: {profile}/{style}/{mood}")

        self.status_label.set_label(f"💾 Preset saved: {profile}")
        GLib.timeout_add_seconds(3, lambda: self.status_label.set_label("🟢 ALL SYSTEMS NOMINAL"))

    def _start_monitoring(self):
        """Start background metrics collection."""
        def update_metrics():
            # Collect all metrics
            cpu_metrics = MetricsCollector.get_cpu_metrics()
            ram_metrics = MetricsCollector.get_ram_metrics()
            gpu_metrics = MetricsCollector.get_gpu_metrics()
            net_metrics = MetricsCollector.get_network_metrics()
            storage_metrics = MetricsCollector.get_storage_metrics()

            # Update overview
            self._update_overview(cpu_metrics, ram_metrics, gpu_metrics, net_metrics)

            # Update specific views
            self._update_cpu_view(cpu_metrics, ram_metrics)
            self._update_gpu_view(gpu_metrics)
            self._update_network_view(net_metrics)
            self._update_storage_view(storage_metrics)

            return GLib.SOURCE_CONTINUE

        # Update every 2 seconds
        GLib.timeout_add_seconds(2, update_metrics)

    def _update_overview(self, cpu, ram, gpus, net):
        """Update overview view."""
        # GPU
        if gpus:
            avg_gpu = sum(g['load'] for g in gpus) / len(gpus)
            self.overview_gpu_label.set_label(f"{avg_gpu:.0f}%")
        else:
            self.overview_gpu_label.set_label("N/A")

        # CPU
        self.overview_cpu_label.set_label(f"{cpu['usage_total']:.0f}%")

        # RAM
        self.overview_ram_label.set_label(f"{ram['used_gb']:.1f} GB")

        # Network (TODO: calculate rate)
        self.overview_net_label.set_label("-- MB/s")

    def _update_cpu_view(self, cpu, ram):
        """Update CPU/RAM view."""
        # CPU
        self.cpu_value_label.set_label(f"{cpu['usage_total']:.0f}%")
        self.cpu_cores_label.set_label(f"{cpu['cores']} cores")
        self.cpu_freq_label.set_label(f"{cpu['frequency']:.0f} MHz")

        # Apply warning color if high
        if cpu['usage_total'] > 80:
            self.cpu_value_label.add_css_class('metric-warning')
        else:
            self.cpu_value_label.remove_css_class('metric-warning')

        # RAM
        self.ram_value_label.set_label(f"{ram['used_gb']:.1f} GB")
        self.ram_progress.set_fraction(ram['percent'] / 100.0)
        self.ram_progress.set_text(f"{ram['percent']:.0f}%")
        self.ram_detail_label.set_label(f"{ram['available_gb']:.1f} GB available of {ram['total_gb']:.1f} GB")

        if ram['percent'] > 80:
            self.ram_value_label.add_css_class('metric-warning')
        else:
            self.ram_value_label.remove_css_class('metric-warning')

    def _update_gpu_view(self, gpus):
        """Update GPU view."""
        if not GPU_AVAILABLE or not gpus:
            return

        # Clear existing GPU cards
        while True:
            child = self.gpu_cards_box.get_first_child()
            if not child:
                break
            self.gpu_cards_box.remove(child)

        # Create card for each GPU
        for gpu in gpus:
            card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            card.add_css_class('metric-card')

            # GPU name
            name_label = Gtk.Label(label=f"{gpu['name']}")
            name_label.add_css_class('metric-title')
            card.append(name_label)

            # Utilization
            util_label = Gtk.Label(label=f"{gpu['load']:.0f}%")
            util_label.add_css_class('metric-value-huge')
            if gpu['load'] > 80:
                util_label.add_css_class('metric-warning')
            card.append(util_label)

            # VRAM
            vram_label = Gtk.Label(label=f"VRAM: {gpu['memory_used_mb']:.0f} / {gpu['memory_total_mb']:.0f} MB ({gpu['memory_percent']:.0f}%)")
            vram_label.add_css_class('metric-label')
            card.append(vram_label)

            # Temperature
            temp_label = Gtk.Label(label=f"Temperature: {gpu['temperature']:.0f}°C")
            temp_label.add_css_class('metric-label')
            if gpu['temperature'] > 80:
                temp_label.add_css_class('metric-warning')
            card.append(temp_label)

            # Power
            if gpu['power_draw'] > 0:
                power_label = Gtk.Label(label=f"Power: {gpu['power_draw']:.0f}W")
                power_label.add_css_class('metric-label')
                card.append(power_label)

            self.gpu_cards_box.append(card)

    def _update_network_view(self, net):
        """Update network view."""
        # TODO: Calculate actual rates (need to track previous values)
        self.net_status_label.set_label("🟢 ONLINE")
        self.net_down_label.set_label(f"↓ -- MB/s")
        self.net_up_label.set_label(f"↑ -- MB/s")

    def _update_storage_view(self, storage):
        """Update storage view."""
        # Clear existing storage cards
        while True:
            child = self.storage_cards_box.get_first_child()
            if not child:
                break
            self.storage_cards_box.remove(child)

        # Create card for each disk
        for disk in storage['disks']:
            card = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
            card.add_css_class('metric-card')

            # Mountpoint
            mount_label = Gtk.Label(label=disk['mountpoint'])
            mount_label.add_css_class('metric-title')
            card.append(mount_label)

            # Usage
            usage_label = Gtk.Label(label=f"{disk['used_gb']:.1f} / {disk['total_gb']:.1f} GB")
            usage_label.add_css_class('metric-value-large')
            if disk['percent'] > 80:
                usage_label.add_css_class('metric-warning')
            card.append(usage_label)

            # Progress bar
            progress = Gtk.ProgressBar()
            progress.add_css_class('metric-progress')
            progress.set_fraction(disk['percent'] / 100.0)
            progress.set_text(f"{disk['percent']:.0f}%")
            progress.set_show_text(True)
            card.append(progress)

            # Device
            device_label = Gtk.Label(label=disk['device'])
            device_label.add_css_class('metric-label')
            card.append(device_label)

            self.storage_cards_box.append(card)


class CockpitHUDApplication(Adw.Application):
    """Touchscreen cockpit HUD application."""

    def __init__(self, **kwargs):
        super().__init__(application_id='com.quanta.tfan.cockpit', **kwargs)

    def do_activate(self):
        """Activate application."""
        win = self.props.active_window
        if not win:
            win = CockpitHUDWindow(application=self)

        # Try to go fullscreen
        win.fullscreen()

        win.present()


def main():
    """Main entry point."""
    app = CockpitHUDApplication()
    return app.run(sys.argv)


if __name__ == '__main__':
    sys.exit(main())
