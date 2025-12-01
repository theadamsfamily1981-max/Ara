#!/usr/bin/env python3
"""
T-FAN GNOME Dashboard

Modern GTK4/libadwaita application for T-FAN neural network training,
optimization, and monitoring with built-in GitHub repository integration.

Usage:
    tfan-gnome                    # Launch main dashboard
    tfan-gnome --view=pareto      # Open Pareto view
    tfan-gnome --repo=<github-url> # Load from GitHub repo
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
import threading
import time

# Import workspace theming and D-Bus service
try:
    from workspace_themes import get_theme, generate_css, get_random_ara_outfit, get_ara_personality_config
    from dbus_service import TFANDBusService
    DBUS_AVAILABLE = True
except ImportError:
    DBUS_AVAILABLE = False
    print("⚠ D-Bus service or workspace themes not available")


class TFANWindow(Adw.ApplicationWindow):
    """Main T-FAN dashboard window."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_title("T-FAN Neural Optimizer")
        self.set_default_size(1400, 900)

        # Workspace mode ('work' or 'relax')
        self.workspace_mode = 'work'
        self.current_theme = get_theme('work') if DBUS_AVAILABLE else None

        # Initialize D-Bus service for Ara control
        if DBUS_AVAILABLE:
            self.dbus_service = TFANDBusService(self)
            self.dbus_service.register()
        else:
            self.dbus_service = None

        # Apply custom CSS
        self._load_custom_css()

        # Main layout
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(self.main_box)

        # Header bar
        self._build_header()

        # Content area with sidebar
        self._build_content()

        # Start background monitoring
        self._start_monitoring()

    def _load_custom_css(self):
        """Load custom CSS based on workspace theme."""
        css_provider = Gtk.CssProvider()

        # Use theme-generated CSS if available
        if DBUS_AVAILABLE and self.current_theme:
            css = generate_css(self.current_theme)
        else:
            # Fallback CSS
            css = """
        .tfan-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 20px;
            margin: 10px;
            color: white;
        }

        .tfan-metric-card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 16px;
            margin: 8px;
            backdrop-filter: blur(10px);
        }

        .tfan-metric-value {
            font-size: 36px;
            font-weight: bold;
            color: #4ade80;
        }

        .tfan-metric-label {
            font-size: 14px;
            opacity: 0.8;
        }

        .tfan-pareto-point {
            background: #3b82f6;
            border-radius: 50%;
        }

        .tfan-best-config {
            background: #ef4444;
            border-radius: 50%;
            border: 3px solid #fbbf24;
        }

        .tfan-training-active {
            background: linear-gradient(90deg, #10b981, #3b82f6);
            animation: pulse 2s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.6; }
        }

        .tfan-status-bar {
            background: rgba(0, 0, 0, 0.05);
            border-top: 1px solid rgba(0, 0, 0, 0.1);
            padding: 8px 16px;
        }
        """
        css_provider.load_from_data(css.encode())

        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def _build_header(self):
        """Build modern header bar with actions."""
        header = Adw.HeaderBar()

        # Title with logo
        title_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        icon = Gtk.Image.new_from_icon_name("network-workgroup-symbolic")
        icon.set_pixel_size(24)
        title_label = Gtk.Label(label="T-FAN Neural Optimizer")
        title_label.add_css_class("title")
        title_box.append(icon)
        title_box.append(title_label)

        header.set_title_widget(title_box)

        # Actions
        menu_button = Gtk.MenuButton()
        menu_button.set_icon_name("open-menu-symbolic")
        menu = Gio.Menu()
        menu.append("Load from GitHub", "app.load_repo")
        menu.append("Export Config", "app.export_config")
        menu.append("Preferences", "app.preferences")
        menu.append("About", "app.about")
        menu_button.set_menu_model(menu)
        header.pack_end(menu_button)

        # Start Training button
        self.train_button = Gtk.Button(label="🚀 Start Training")
        self.train_button.add_css_class("suggested-action")
        self.train_button.connect("clicked", self._on_start_training)
        header.pack_start(self.train_button)

        # Run Pareto button
        pareto_button = Gtk.Button(label="🎯 Optimize")
        pareto_button.connect("clicked", self._on_run_pareto)
        header.pack_start(pareto_button)

        # Workspace mode switcher
        if DBUS_AVAILABLE:
            mode_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
            mode_box.add_css_class("mode-badge")

            self.mode_icon = Gtk.Image.new_from_icon_name("weather-clear-night-symbolic")
            self.mode_label = Gtk.Label(label="⚡ Work")
            self.mode_label.add_css_class("heading")

            mode_box.append(self.mode_icon)
            mode_box.append(self.mode_label)

            mode_switch_button = Gtk.Button()
            mode_switch_button.set_child(mode_box)
            mode_switch_button.connect("clicked", self._on_toggle_workspace_mode)
            mode_switch_button.set_tooltip_text("Switch between Work and Relaxation modes")

            header.pack_end(mode_switch_button)

        self.main_box.append(header)

    def _build_content(self):
        """Build main content area with sidebar navigation."""
        split_view = Adw.OverlaySplitView()

        # Sidebar
        sidebar = self._build_sidebar()
        split_view.set_sidebar(sidebar)

        # Content stack
        self.content_stack = Gtk.Stack()
        self.content_stack.set_transition_type(Gtk.StackTransitionType.SLIDE_LEFT_RIGHT)

        # Add views
        self.content_stack.add_titled(
            self._build_dashboard_view(),
            "dashboard",
            "Dashboard"
        )
        self.content_stack.add_titled(
            self._build_pareto_view(),
            "pareto",
            "Pareto Optimization"
        )
        self.content_stack.add_titled(
            self._build_training_view(),
            "training",
            "Training Monitor"
        )
        self.content_stack.add_titled(
            self._build_screensaver_view(),
            "screensaver",
            "Topology Screensaver"
        )
        self.content_stack.add_titled(
            self._build_config_view(),
            "config",
            "Configuration"
        )
        self.content_stack.add_titled(
            self._build_repo_view(),
            "repo",
            "Repository"
        )

        split_view.set_content(self.content_stack)

        # Status bar at bottom
        status_bar = self._build_status_bar()

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        content_box.append(split_view)
        content_box.append(status_bar)

        self.main_box.append(content_box)

    def _build_sidebar(self):
        """Build sidebar navigation."""
        sidebar_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        sidebar_box.set_size_request(200, -1)
        sidebar_box.add_css_class("sidebar")

        list_box = Gtk.ListBox()
        list_box.add_css_class("navigation-sidebar")
        list_box.set_selection_mode(Gtk.SelectionMode.SINGLE)

        views = [
            ("📊", "Dashboard", "dashboard"),
            ("🎯", "Pareto", "pareto"),
            ("🚀", "Training", "training"),
            ("🌌", "Screensaver", "screensaver"),
            ("⚙️", "Config", "config"),
            ("📦", "Repository", "repo"),
        ]

        for emoji, label, view_id in views:
            row = Adw.ActionRow()
            row.set_title(f"{emoji}  {label}")
            row.set_activatable(True)
            row.view_id = view_id
            list_box.append(row)

        list_box.connect("row-activated", self._on_sidebar_activated)
        sidebar_box.append(list_box)

        return sidebar_box

    def _on_sidebar_activated(self, list_box, row):
        """Handle sidebar navigation."""
        self.content_stack.set_visible_child_name(row.view_id)

    def _build_dashboard_view(self):
        """Build main dashboard with live metrics."""
        scroll = Gtk.ScrolledWindow()
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(20)
        box.set_margin_end(20)
        box.set_margin_top(20)
        box.set_margin_bottom(20)

        # Welcome card
        welcome = Adw.StatusPage()
        welcome.set_icon_name("network-workgroup-symbolic")
        welcome.set_title("T-FAN Neural Optimizer")
        welcome.set_description(
            "Topological, Fractal, Affective Neural network training\n"
            "with multi-objective Pareto optimization"
        )
        box.append(welcome)

        # Metrics grid
        metrics_frame = Adw.PreferencesGroup()
        metrics_frame.set_title("Live Metrics")

        grid = Gtk.Grid()
        grid.set_column_spacing(12)
        grid.set_row_spacing(12)
        grid.set_column_homogeneous(True)

        # Metric cards
        self.metric_cards = {}
        metrics = [
            ("accuracy", "Accuracy", "0.000", "emblem-ok-symbolic"),
            ("latency", "Latency (ms)", "0.0", "preferences-system-time-symbolic"),
            ("hypervolume", "Hypervolume", "0", "view-dual-symbolic"),
            ("epr_cv", "EPR CV", "0.000", "utilities-system-monitor-symbolic"),
        ]

        for idx, (key, label, initial, icon_name) in enumerate(metrics):
            card = self._create_metric_card(label, initial, icon_name)
            self.metric_cards[key] = card["value_label"]
            grid.attach(card["widget"], idx % 2, idx // 2, 1, 1)

        metrics_frame.add(grid)
        box.append(metrics_frame)

        # Quick actions
        actions_group = Adw.PreferencesGroup()
        actions_group.set_title("Quick Actions")

        action_row1 = Adw.ActionRow()
        action_row1.set_title("Load GitHub Repository")
        action_row1.set_subtitle("Clone and configure T-FAN from GitHub")
        load_btn = Gtk.Button(label="Load")
        load_btn.set_valign(Gtk.Align.CENTER)
        load_btn.connect("clicked", self._on_load_repo)
        action_row1.add_suffix(load_btn)
        actions_group.add(action_row1)

        action_row2 = Adw.ActionRow()
        action_row2.set_title("Run Pareto Optimization")
        action_row2.set_subtitle("Find optimal config trade-offs")
        pareto_btn = Gtk.Button(label="Optimize")
        pareto_btn.set_valign(Gtk.Align.CENTER)
        pareto_btn.connect("clicked", self._on_run_pareto)
        action_row2.add_suffix(pareto_btn)
        actions_group.add(action_row2)

        box.append(actions_group)

        scroll.set_child(box)
        return scroll

    def _create_metric_card(self, label, value, icon_name):
        """Create a metric display card."""
        card = Adw.Bin()
        card.add_css_class("tfan-metric-card")

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)

        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(32)
        icon.set_opacity(0.6)

        value_label = Gtk.Label(label=value)
        value_label.add_css_class("tfan-metric-value")

        name_label = Gtk.Label(label=label)
        name_label.add_css_class("tfan-metric-label")

        box.append(icon)
        box.append(value_label)
        box.append(name_label)

        card.set_child(box)

        return {"widget": card, "value_label": value_label}

    def _build_pareto_view(self):
        """Build Pareto optimization view."""
        scroll = Gtk.ScrolledWindow()
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(20)
        box.set_margin_end(20)
        box.set_margin_top(20)
        box.set_margin_bottom(20)

        # Header
        status = Adw.StatusPage()
        status.set_icon_name("view-dual-symbolic")
        status.set_title("Pareto Optimization")
        status.set_description("Multi-objective config optimization with EHVI")
        box.append(status)

        # Settings
        settings_group = Adw.PreferencesGroup()
        settings_group.set_title("Optimization Settings")

        iterations_row = Adw.SpinRow()
        iterations_row.set_title("Iterations")
        iterations_row.set_subtitle("Number of optimization iterations")
        adjustment = Gtk.Adjustment(value=100, lower=10, upper=1000, step_increment=10)
        iterations_row.set_adjustment(adjustment)
        settings_group.add(iterations_row)

        initial_row = Adw.SpinRow()
        initial_row.set_title("Initial Points")
        initial_row.set_subtitle("Random samples before optimization")
        adjustment2 = Gtk.Adjustment(value=10, lower=5, upper=50, step_increment=5)
        initial_row.set_adjustment(adjustment2)
        settings_group.add(initial_row)

        box.append(settings_group)

        # Run button
        run_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        run_box.set_halign(Gtk.Align.CENTER)
        run_box.set_margin_top(20)

        run_button = Gtk.Button(label="🎯 Run Optimization")
        run_button.add_css_class("pill")
        run_button.add_css_class("suggested-action")
        run_button.set_size_request(200, 48)
        run_button.connect("clicked", lambda _: self._run_pareto_optimization())
        run_box.append(run_button)

        box.append(run_box)

        # Results placeholder
        results_group = Adw.PreferencesGroup()
        results_group.set_title("Results")
        self.pareto_results_label = Gtk.Label(label="No results yet")
        self.pareto_results_label.set_margin_top(20)
        self.pareto_results_label.set_margin_bottom(20)
        results_group.add(self.pareto_results_label)
        box.append(results_group)

        scroll.set_child(box)
        return scroll

    def _build_training_view(self):
        """Build training monitor view."""
        scroll = Gtk.ScrolledWindow()
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(20)
        box.set_margin_end(20)
        box.set_margin_top(20)

        status = Adw.StatusPage()
        status.set_icon_name("media-playback-start-symbolic")
        status.set_title("Training Monitor")
        status.set_description("Real-time training metrics and logs")
        box.append(status)

        # Training controls
        controls_group = Adw.PreferencesGroup()
        controls_group.set_title("Controls")

        config_row = Adw.ComboRow()
        config_row.set_title("Configuration")
        config_row.set_subtitle("Select training config")
        string_list = Gtk.StringList()
        string_list.append("configs/auto/best.yaml")
        string_list.append("configs/7b/quanta_focus.yaml")
        string_list.append("configs/7b/default.yaml")
        config_row.set_model(string_list)
        controls_group.add(config_row)

        box.append(controls_group)

        # Logs
        logs_group = Adw.PreferencesGroup()
        logs_group.set_title("Training Logs")

        self.log_view = Gtk.TextView()
        self.log_view.set_editable(False)
        self.log_view.set_monospace(True)
        self.log_view.set_size_request(-1, 400)
        self.log_buffer = self.log_view.get_buffer()

        scroll_logs = Gtk.ScrolledWindow()
        scroll_logs.set_child(self.log_view)
        logs_group.add(scroll_logs)

        box.append(logs_group)

        scroll.set_child(box)
        return scroll

    def _build_screensaver_view(self):
        """Build topology screensaver view with WebGL visualization."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        # Controls header
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        header_box.set_margin_start(20)
        header_box.set_margin_end(20)
        header_box.set_margin_top(12)
        header_box.set_margin_bottom(12)

        # Mode selector
        mode_label = Gtk.Label(label="Mode:")
        header_box.append(mode_label)

        self.screensaver_mode_combo = Gtk.ComboBoxText()
        self.screensaver_mode_combo.append_text("Barcode Nebula 🌠")
        self.screensaver_mode_combo.append_text("Landscape Waterfall 🌊")
        self.screensaver_mode_combo.append_text("Poincaré Orbits 🪐")
        self.screensaver_mode_combo.append_text("Pareto Galaxy ⭐")
        self.screensaver_mode_combo.set_active(1)  # Default to Landscape
        self.screensaver_mode_combo.connect("changed", self._on_screensaver_mode_changed)
        header_box.append(self.screensaver_mode_combo)

        header_box.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))

        # Particle count slider
        particles_label = Gtk.Label(label="Particles:")
        header_box.append(particles_label)

        self.particles_scale = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 200, 2000, 100
        )
        self.particles_scale.set_value(800)
        self.particles_scale.set_size_request(150, -1)
        self.particles_scale.connect("value-changed", self._on_screensaver_setting_changed)
        header_box.append(self.particles_scale)

        header_box.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))

        # Auto-rotate toggle
        self.autorotate_switch = Gtk.Switch()
        self.autorotate_switch.set_active(True)
        self.autorotate_switch.connect("notify::active", self._on_screensaver_setting_changed)
        autorotate_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        autorotate_box.append(Gtk.Label(label="Auto-Rotate:"))
        autorotate_box.append(self.autorotate_switch)
        header_box.append(autorotate_box)

        header_box.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))

        # Fullscreen button
        fullscreen_btn = Gtk.Button(label="⛶ Fullscreen")
        fullscreen_btn.connect("clicked", self._on_screensaver_fullscreen)
        header_box.append(fullscreen_btn)

        # Reload button
        reload_btn = Gtk.Button(label="🔄")
        reload_btn.connect("clicked", self._on_screensaver_reload)
        header_box.append(reload_btn)

        box.append(header_box)

        # WebView for screensaver
        try:
            self.screensaver_webview = WebKit.WebView()

            # Enable WebGL and other features
            settings = self.screensaver_webview.get_settings()
            settings.set_enable_webgl(True)
            settings.set_enable_accelerated_2d_canvas(True)
            settings.set_hardware_acceleration_policy(WebKit.HardwareAccelerationPolicy.ALWAYS)
            settings.set_javascript_can_access_clipboard(False)

            # Load the screensaver HTML
            self._load_screensaver_content()

            # Expand to fill space
            self.screensaver_webview.set_vexpand(True)
            self.screensaver_webview.set_hexpand(True)

            box.append(self.screensaver_webview)

        except Exception as e:
            # Fallback if WebKit not available
            error_status = Adw.StatusPage()
            error_status.set_icon_name("dialog-error-symbolic")
            error_status.set_title("WebKit Not Available")
            error_status.set_description(
                f"Install WebKitGTK 6.0 to view the screensaver:\n"
                f"sudo apt install gir1.2-webkit-6.0\n\n"
                f"Error: {e}"
            )
            box.append(error_status)

        return box

    def _load_screensaver_content(self):
        """Load WebGL screensaver HTML into WebView."""
        # Get mode index (0-3)
        mode_idx = self.screensaver_mode_combo.get_active()
        modes = ['barcode', 'landscape', 'poincare', 'pareto']
        mode = modes[mode_idx] if 0 <= mode_idx < len(modes) else 'landscape'

        # Get settings
        particle_count = int(self.particles_scale.get_value())
        auto_rotate = self.autorotate_switch.get_active()

        # Path to screensaver web files
        repo_root = Path(__file__).parent.parent.parent
        web_dir = repo_root / "tfan" / "viz" / "screensaver" / "web"

        if web_dir.exists() and (web_dir / "index.html").exists():
            # Load from file system
            html_path = f"file://{web_dir / 'index.html'}"
            self.screensaver_webview.load_uri(html_path)
        else:
            # Embed the screensaver inline
            html_content = self._generate_screensaver_html(mode, particle_count, auto_rotate)
            self.screensaver_webview.load_html(html_content, "about:blank")

    def _generate_screensaver_html(self, mode, particle_count, auto_rotate):
        """Generate inline HTML for screensaver with live metrics integration."""
        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ margin: 0; overflow: hidden; background: #020306; font-family: monospace; color: #fff; }}
        canvas {{ display: block; width: 100%; height: 100vh; }}
        #hud {{
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 1000;
            background: rgba(0, 0, 0, 0.5);
            padding: 15px;
            border-radius: 8px;
            backdrop-filter: blur(10px);
            font-size: 12px;
        }}
        .metric-row {{
            margin: 5px 0;
            display: flex;
            justify-content: space-between;
            gap: 10px;
        }}
        .metric-label {{ color: #888; }}
        .metric-value {{ color: #667eea; font-weight: bold; }}
        .status-connected {{ color: #00ff00; }}
        .status-disconnected {{ color: #ff4444; }}
    </style>
</head>
<body>
    <div id="hud">
        <div style="font-size: 14px; margin-bottom: 10px; color: #667eea;">⚛️ T-FAN Topology</div>
        <div class="metric-row">
            <span class="metric-label">Mode:</span>
            <span class="metric-value" id="mode">{mode}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">EPR-CV:</span>
            <span class="metric-value" id="epr-cv">--</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Topo Gap:</span>
            <span class="metric-value" id="topo-gap">--</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Accuracy:</span>
            <span class="metric-value" id="accuracy">--</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Status:</span>
            <span id="status" class="status-disconnected">⚫ Searching...</span>
        </div>
    </div>
    <canvas id="canvas"></canvas>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r160/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Live metrics from model's topology computation
        let metrics = {{
            epr_cv: 0.10,
            topo_gap: 0.015,
            accuracy: 0.0,
            latency_ms: 0.0,
            training_active: false
        }};

        // Try to connect to T-FAN API
        async function pollMetrics() {{
            try {{
                const response = await fetch('http://localhost:8000/api/metrics');
                if (response.ok) {{
                    const data = await response.json();
                    metrics = data;
                    updateHUD(true);
                }} else {{
                    // Fallback: try metrics.json file via file:// (won't work in WebView, but try anyway)
                    tryLocalMetrics();
                }}
            }} catch (e) {{
                tryLocalMetrics();
            }}
        }}

        async function tryLocalMetrics() {{
            try {{
                // Try metrics bridge on alternate port
                const response = await fetch('http://localhost:9101/metrics');
                if (response.ok) {{
                    const data = await response.json();
                    metrics = data;
                    updateHUD(true);
                }} else {{
                    updateHUD(false);
                }}
            }} catch (e) {{
                updateHUD(false);
            }}
        }}

        function updateHUD(connected) {{
            document.getElementById('epr-cv').textContent = metrics.epr_cv?.toFixed(3) || '--';
            document.getElementById('topo-gap').textContent = metrics.topo_gap?.toFixed(4) || '--';
            document.getElementById('accuracy').textContent = metrics.accuracy?.toFixed(3) || '--';

            const statusEl = document.getElementById('status');
            if (connected) {{
                statusEl.textContent = metrics.training_active ? '🟢 Training Live' : '🟡 Connected';
                statusEl.className = 'status-connected';
            }} else {{
                statusEl.textContent = '⚫ Demo Mode';
                statusEl.className = 'status-disconnected';
            }}
        }}

        // Poll every 2 seconds
        setInterval(pollMetrics, 2000);
        pollMetrics();

        // Three.js scene setup
        const canvas = document.getElementById('canvas');
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x020306);
        scene.fog = new THREE.Fog(0x020306, 10, 50);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.set(0, 5, 10);

        const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);

        const controls = new THREE.OrbitControls(camera, canvas);
        controls.enableDamping = true;
        controls.autoRotate = {str(auto_rotate).lower()};
        controls.autoRotateSpeed = 0.5;

        // Lights
        scene.add(new THREE.AmbientLight(0x404040, 1.0));
        const light1 = new THREE.PointLight(0x667eea, 2.0);
        light1.position.set(10, 10, 10);
        scene.add(light1);

        const light2 = new THREE.PointLight(0x764ba2, 1.5);
        light2.position.set(-10, 5, -10);
        scene.add(light2);

        // Create {mode} visualization
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const colors = [];
        const velocities = [];

        for (let i = 0; i < {particle_count}; i++) {{
            positions.push(
                (Math.random() - 0.5) * 10,
                (Math.random() - 0.5) * 10,
                (Math.random() - 0.5) * 10
            );

            // Color based on mode
            const hue = 0.55 + Math.random() * 0.2; // Purple-blue range
            const color = new THREE.Color().setHSL(hue, 0.8, 0.6);
            colors.push(color.r, color.g, color.b);

            // Random velocities for organic motion
            velocities.push(
                (Math.random() - 0.5) * 0.02,
                (Math.random() - 0.5) * 0.02,
                (Math.random() - 0.5) * 0.02
            );
        }}

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({{
            size: 0.15,
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            blending: THREE.AdditiveBlending
        }});
        const points = new THREE.Points(geometry, material);
        scene.add(points);

        // Animation driven by live topology metrics
        let time = 0;
        function animate() {{
            requestAnimationFrame(animate);
            time += 0.016; // ~60fps

            // EPR-CV drives the "tension" - how much the topology is fluctuating
            const tension = 0.4 + (metrics.epr_cv || 0.10) * 2.0;

            // Topo gap affects particle size (smaller gap = more coherent = bigger particles)
            const gapFactor = 1.0 - (metrics.topo_gap || 0.015) * 20.0;
            material.size = 0.15 * Math.max(0.5, Math.min(1.5, gapFactor));

            // Organic particle motion influenced by topology
            const posArray = geometry.attributes.position.array;
            for (let i = 0; i < posArray.length; i += 3) {{
                // Apply velocity
                posArray[i] += velocities[i] * tension;
                posArray[i + 1] += velocities[i + 1] * tension;
                posArray[i + 2] += velocities[i + 2] * tension;

                // Boundary wrapping
                if (Math.abs(posArray[i]) > 8) velocities[i] *= -1;
                if (Math.abs(posArray[i + 1]) > 8) velocities[i + 1] *= -1;
                if (Math.abs(posArray[i + 2]) > 8) velocities[i + 2] *= -1;

                // Add topology-driven wave motion
                posArray[i + 1] += Math.sin(time * tension + posArray[i]) * 0.01;
            }}
            geometry.attributes.position.needsUpdate = true;

            // Rotate entire system based on tension
            points.rotation.y += 0.001 * tension;

            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""

    def _on_screensaver_mode_changed(self, combo):
        """Handle screensaver mode change."""
        self._load_screensaver_content()

    def _on_screensaver_setting_changed(self, *args):
        """Handle screensaver setting change."""
        self._load_screensaver_content()

    def _on_screensaver_fullscreen(self, button):
        """Toggle fullscreen for screensaver."""
        if self.is_fullscreen():
            self.unfullscreen()
        else:
            self.fullscreen()

    def _on_screensaver_reload(self, button):
        """Reload screensaver."""
        self._load_screensaver_content()

    def _build_config_view(self):
        """Build configuration editor."""
        scroll = Gtk.ScrolledWindow()
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(20)
        box.set_margin_end(20)
        box.set_margin_top(20)

        status = Adw.StatusPage()
        status.set_icon_name("emblem-system-symbolic")
        status.set_title("Configuration")
        status.set_description("Edit T-FAN model and training settings")
        box.append(status)

        # Config editor placeholder
        editor_group = Adw.PreferencesGroup()
        editor_group.set_title("Current Config: configs/auto/best.yaml")

        text_view = Gtk.TextView()
        text_view.set_monospace(True)
        text_view.set_size_request(-1, 500)
        editor_group.add(text_view)

        box.append(editor_group)

        scroll.set_child(box)
        return scroll

    def _build_repo_view(self):
        """Build GitHub repository loader."""
        scroll = Gtk.ScrolledWindow()
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(20)
        box.set_margin_end(20)
        box.set_margin_top(20)

        status = Adw.StatusPage()
        status.set_icon_name("folder-download-symbolic")
        status.set_title("Load from GitHub")
        status.set_description("Clone and auto-configure T-FAN repository")
        box.append(status)

        # Repo URL entry
        repo_group = Adw.PreferencesGroup()
        repo_group.set_title("Repository")

        url_row = Adw.EntryRow()
        url_row.set_title("GitHub URL")
        url_row.set_text("https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis")
        repo_group.add(url_row)

        dir_row = Adw.ActionRow()
        dir_row.set_title("Local Directory")
        dir_row.set_subtitle(str(Path.home() / "tfan-workspace"))
        repo_group.add(dir_row)

        box.append(repo_group)

        # Load button
        load_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        load_box.set_halign(Gtk.Align.CENTER)
        load_box.set_margin_top(20)

        load_button = Gtk.Button(label="📦 Clone & Configure")
        load_button.add_css_class("pill")
        load_button.add_css_class("suggested-action")
        load_button.set_size_request(200, 48)
        load_button.connect("clicked", lambda _: self._clone_repo(url_row.get_text()))
        load_box.append(load_button)

        box.append(load_box)

        scroll.set_child(box)
        return scroll

    def _build_status_bar(self):
        """Build bottom status bar."""
        bar = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        bar.add_css_class("tfan-status-bar")
        bar.set_margin_start(12)
        bar.set_margin_end(12)
        bar.set_margin_top(4)
        bar.set_margin_bottom(4)

        self.status_label = Gtk.Label(label="Ready")
        self.status_label.set_halign(Gtk.Align.START)
        bar.append(self.status_label)

        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        bar.append(spacer)

        self.training_status = Gtk.Label(label="● Idle")
        bar.append(self.training_status)

        return bar

    def _on_start_training(self, button):
        """Start training session."""
        self.status_label.set_label("Starting training...")
        self._log("🚀 Starting training session...")

        # Launch training in terminal
        subprocess.Popen([
            "gnome-terminal",
            "--",
            "bash",
            "-c",
            "cd ~/tfan-workspace && python training/train.py; exec bash"
        ])

    def _on_run_pareto(self, button):
        """Run Pareto optimization."""
        self.content_stack.set_visible_child_name("pareto")

    def _on_load_repo(self, button):
        """Navigate to repo view."""
        self.content_stack.set_visible_child_name("repo")

    def _clone_repo(self, url):
        """Clone GitHub repository."""
        self.status_label.set_label(f"Cloning {url}...")
        self._log(f"📦 Cloning repository: {url}")

        def clone_thread():
            target_dir = Path.home() / "tfan-workspace"
            try:
                subprocess.run([
                    "git", "clone", url, str(target_dir)
                ], check=True, capture_output=True)

                GLib.idle_add(self._on_clone_complete, target_dir)
            except subprocess.CalledProcessError as e:
                GLib.idle_add(self._on_clone_error, str(e))

        threading.Thread(target=clone_thread, daemon=True).start()

    def _on_clone_complete(self, target_dir):
        """Handle successful clone."""
        self.status_label.set_label(f"✓ Repository loaded: {target_dir}")
        self._log(f"✓ Repository cloned successfully to {target_dir}")

        # Auto-install dependencies
        self._log("📦 Installing dependencies...")
        subprocess.Popen([
            "gnome-terminal",
            "--",
            "bash",
            "-c",
            f"cd {target_dir} && pip install -e .; exec bash"
        ])

    def _on_clone_error(self, error):
        """Handle clone error."""
        self.status_label.set_label(f"✗ Clone failed: {error}")
        self._log(f"✗ Clone failed: {error}")

    def _run_pareto_optimization(self):
        """Run Pareto optimization."""
        self.status_label.set_label("Running Pareto optimization...")
        self.pareto_results_label.set_label("Optimizing... This may take several minutes.")

        def pareto_thread():
            try:
                result = subprocess.run([
                    "python", "-c",
                    """
from tfan.pareto_v2 import ParetoRunner, ParetoRunnerConfig
config = ParetoRunnerConfig(n_initial_points=10, n_iterations=50, output_dir='artifacts/pareto')
runner = ParetoRunner(config)
front = runner.run(verbose=False)
print(f"{front.n_dominated},{front.hypervolume:.0f}")
                    """
                ], capture_output=True, text=True, timeout=300)

                n_points, hv = result.stdout.strip().split(',')
                GLib.idle_add(self._on_pareto_complete, int(n_points), float(hv))
            except Exception as e:
                GLib.idle_add(self._on_pareto_error, str(e))

        threading.Thread(target=pareto_thread, daemon=True).start()

    def _on_pareto_complete(self, n_points, hv):
        """Handle Pareto completion."""
        self.status_label.set_label("✓ Pareto optimization complete")
        self.pareto_results_label.set_label(
            f"✓ Found {n_points} Pareto-optimal configurations\n"
            f"Hypervolume: {hv:.0f}\n\n"
            f"Results saved to artifacts/pareto/"
        )
        self._log(f"✓ Pareto complete: {n_points} points, HV={hv:.0f}")

    def _on_pareto_error(self, error):
        """Handle Pareto error."""
        self.status_label.set_label(f"✗ Pareto failed: {error}")
        self.pareto_results_label.set_label(f"✗ Error: {error}")

    def _on_toggle_workspace_mode(self, button):
        """Toggle between work and relaxation workspace modes."""
        if not DBUS_AVAILABLE:
            return

        # Switch mode
        self.workspace_mode = 'relax' if self.workspace_mode == 'work' else 'work'
        self.apply_workspace_theme(self.workspace_mode)

        # Update mode indicator
        if self.workspace_mode == 'work':
            self.mode_label.set_label("⚡ Work")
            self.mode_icon.set_from_icon_name("weather-clear-symbolic")
        else:
            self.mode_label.set_label("🌙 Relax")
            self.mode_icon.set_from_icon_name("weather-clear-night-symbolic")

        # Notify Ara via D-Bus
        if self.dbus_service:
            # Get personality config for Ara
            personality = get_ara_personality_config(self.current_theme)
            print(f"[Workspace] Switched to {self.workspace_mode} mode")
            print(f"[Ara] Personality config: {personality}")

            # Ara can randomize outfit in relax mode
            if self.workspace_mode == 'relax':
                outfit = get_random_ara_outfit('relax')
                print(f"[Ara] Random outfit suggestion: {outfit}")

    def apply_workspace_theme(self, mode):
        """
        Apply workspace theme (work or relax).

        Args:
            mode: 'work' or 'relax'
        """
        if not DBUS_AVAILABLE:
            return

        # Get theme
        self.current_theme = get_theme(mode)
        self.workspace_mode = mode

        # Regenerate CSS
        css_provider = Gtk.CssProvider()
        css = generate_css(self.current_theme)
        css_provider.load_from_data(css.encode())

        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

        print(f"✓ Applied {self.current_theme['display_name']} theme")

    def close_dbus_service(self):
        """Clean up D-Bus service on shutdown."""
        if self.dbus_service:
            self.dbus_service.unregister()

    def _start_monitoring(self):
        """Start background monitoring of metrics."""
        def update_metrics():
            metrics_file = Path.home() / ".cache/tfan/metrics.json"
            if metrics_file.exists():
                try:
                    with open(metrics_file) as f:
                        metrics = json.load(f)

                    self.metric_cards["accuracy"].set_label(
                        f"{metrics.get('accuracy', 0):.3f}"
                    )
                    self.metric_cards["latency"].set_label(
                        f"{metrics.get('latency_ms', 0):.1f}"
                    )
                    self.metric_cards["hypervolume"].set_label(
                        f"{metrics.get('hypervolume', 0):.0f}"
                    )
                    self.metric_cards["epr_cv"].set_label(
                        f"{metrics.get('epr_cv', 0):.3f}"
                    )

                    if metrics.get('training_active'):
                        self.training_status.set_label("● Training")
                        self.training_status.add_css_class("tfan-training-active")
                    else:
                        self.training_status.set_label("● Idle")
                        self.training_status.remove_css_class("tfan-training-active")

                except Exception:
                    pass

            return GLib.SOURCE_CONTINUE

        GLib.timeout_add_seconds(2, update_metrics)

    def _log(self, message):
        """Add message to training log."""
        timestamp = time.strftime("%H:%M:%S")
        self.log_buffer.insert(
            self.log_buffer.get_end_iter(),
            f"[{timestamp}] {message}\n"
        )


class TFANApplication(Adw.Application):
    """Main T-FAN application."""

    def __init__(self, **kwargs):
        super().__init__(application_id='com.quanta.tfan', **kwargs)

        # Register actions
        self.create_action("quit", self.quit, ["<primary>q"])
        self.create_action("about", self.on_about)
        self.create_action("preferences", self.on_preferences)
        self.create_action("load_repo", self.on_load_repo)
        self.create_action("export_config", self.on_export_config)

    def do_activate(self):
        """Activate application."""
        win = self.props.active_window
        if not win:
            win = TFANWindow(application=self)
        win.present()

    def do_shutdown(self):
        """Cleanup on shutdown."""
        # Close D-Bus service
        win = self.props.active_window
        if win and hasattr(win, 'close_dbus_service'):
            win.close_dbus_service()
        Adw.Application.do_shutdown(self)

    def create_action(self, name, callback, shortcuts=None):
        """Create application action."""
        action = Gio.SimpleAction.new(name, None)
        action.connect("activate", callback)
        self.add_action(action)
        if shortcuts:
            self.set_accels_for_action(f"app.{name}", shortcuts)

    def on_about(self, action, param):
        """Show about dialog."""
        about = Adw.AboutWindow(
            transient_for=self.props.active_window,
            application_name="T-FAN Neural Optimizer",
            application_icon="network-workgroup-symbolic",
            developer_name="Quanta-meis-nib-cis",
            version="1.0.0",
            developers=["The Adams Family"],
            copyright="© 2025",
            license_type=Gtk.License.MIT_X11,
            website="https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis",
            issue_url="https://github.com/theadamsfamily1981-max/Quanta-meis-nib-cis/issues",
        )
        about.present()

    def on_preferences(self, action, param):
        """Show preferences."""
        # TODO: Implement preferences dialog
        pass

    def on_load_repo(self, action, param):
        """Load repository action."""
        win = self.props.active_window
        if win:
            win.content_stack.set_visible_child_name("repo")

    def on_export_config(self, action, param):
        """Export configuration."""
        # TODO: Implement export dialog
        pass


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="T-FAN GNOME Dashboard")
    parser.add_argument("--view", default="dashboard", help="Initial view to show")
    parser.add_argument("--repo", help="GitHub repository URL to load")

    args = parser.parse_args()

    app = TFANApplication()

    # Auto-load repo if specified
    if args.repo:
        # TODO: Auto-clone on startup
        pass

    return app.run(sys.argv)


if __name__ == "__main__":
    sys.exit(main())
