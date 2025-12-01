#!/usr/bin/env python3
"""
D-Bus Client Example for Ara Avatar System

Shows how Ara can control the T-FAN GNOME cockpit via D-Bus.
This would be integrated into Ara's voice command routing system.

Usage:
    # From Ara's voice pipeline:
    control_tfan("show topology", mode="landscape", fullscreen=True)
    control_tfan("switch to work mode")
    control_tfan("minimize window")
"""

import gi
gi.require_version('Gio', '2.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gio, GLib


class TFANController:
    """D-Bus client for controlling T-FAN GNOME app from Ara."""

    def __init__(self):
        """Initialize D-Bus connection to T-FAN."""
        self.proxy = None
        self._connect()

    def _connect(self):
        """Connect to T-FAN D-Bus service."""
        try:
            connection = Gio.bus_get_sync(Gio.BusType.SESSION, None)

            self.proxy = Gio.DBusProxy.new_sync(
                connection,
                Gio.DBusProxyFlags.NONE,
                None,
                'com.quanta.tfan',
                '/com/quanta/tfan',
                'com.quanta.tfan.Control',
                None
            )

            # Subscribe to signals
            self.proxy.connect('g-signal', self._on_signal)

            print("✓ Connected to T-FAN D-Bus service")

        except Exception as e:
            print(f"✗ Failed to connect to T-FAN: {e}")
            self.proxy = None

    def _on_signal(self, proxy, sender_name, signal_name, parameters):
        """Handle signals from T-FAN (for Ara to react)."""
        if signal_name == 'TrainingStarted':
            config = parameters[0]
            print(f"[Ara] Training started with config: {config}")
            # Ara could announce: "Training session initiated with {config} configuration"

        elif signal_name == 'MetricsUpdated':
            metrics = parameters[0]
            print(f"[Ara] Metrics updated: {dict(metrics)}")
            # Ara could monitor and proactively announce milestones

        elif signal_name == 'AlertRaised':
            level, message = parameters
            print(f"[Ara] Alert [{level}]: {message}")
            # Ara announces the alert to user

        elif signal_name == 'WorkspaceModeChanged':
            mode = parameters[0]
            print(f"[Ara] Workspace switched to {mode} mode")
            # Ara adjusts her personality

    # View Control

    def switch_view(self, view):
        """Switch to a different view: dashboard, pareto, training, screensaver, config, repo."""
        if not self.proxy:
            return False

        result = self.proxy.call_sync(
            'SwitchView',
            GLib.Variant('(s)', (view,)),
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return result[0]

    def get_current_view(self):
        """Get the currently active view."""
        if not self.proxy:
            return None

        result = self.proxy.call_sync(
            'GetCurrentView',
            None,
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return result[0]

    # Workspace Mode Control

    def set_workspace_mode(self, mode):
        """Set workspace mode: 'work' or 'relax'."""
        if not self.proxy:
            return False

        result = self.proxy.call_sync(
            'SetWorkspaceMode',
            GLib.Variant('(s)', (mode,)),
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return result[0]

    def get_workspace_mode(self):
        """Get current workspace mode."""
        if not self.proxy:
            return None

        result = self.proxy.call_sync(
            'GetWorkspaceMode',
            None,
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return result[0]

    # Topology Screensaver Control

    def show_topology(self, mode=None, fullscreen=False):
        """
        Show topology visualization.

        Args:
            mode: 'barcode', 'landscape', 'poincare', or 'pareto'
            fullscreen: Whether to go fullscreen
        """
        if not self.proxy:
            return False

        result = self.proxy.call_sync(
            'ShowTopology',
            GLib.Variant('(sb)', (mode or '', fullscreen)),
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return result[0]

    def hide_topology(self):
        """Hide topology visualization."""
        if not self.proxy:
            return False

        result = self.proxy.call_sync(
            'HideTopology',
            None,
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return result[0]

    def set_topology_mode(self, mode):
        """Change topology visualization mode."""
        if not self.proxy:
            return False

        result = self.proxy.call_sync(
            'SetTopologyMode',
            GLib.Variant('(s)', (mode,)),
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return result[0]

    # Window Control

    def toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if not self.proxy:
            return False

        result = self.proxy.call_sync(
            'ToggleFullscreen',
            None,
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return result[0]

    def minimize(self):
        """Minimize window."""
        if not self.proxy:
            return False

        result = self.proxy.call_sync(
            'Minimize',
            None,
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return result[0]

    def restore(self):
        """Restore/show window."""
        if not self.proxy:
            return False

        result = self.proxy.call_sync(
            'Restore',
            None,
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return result[0]

    # Training Control

    def start_training(self, config='default'):
        """Start training with specified config."""
        if not self.proxy:
            return False

        result = self.proxy.call_sync(
            'StartTraining',
            GLib.Variant('(s)', (config,)),
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return result[0]

    def stop_training(self):
        """Stop training."""
        if not self.proxy:
            return False

        result = self.proxy.call_sync(
            'StopTraining',
            None,
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return result[0]

    # Status Queries

    def get_status(self):
        """Get current system status."""
        if not self.proxy:
            return {}

        result = self.proxy.call_sync(
            'GetStatus',
            None,
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return dict(result[0])

    def get_metrics(self):
        """Get current metrics."""
        if not self.proxy:
            return {}

        result = self.proxy.call_sync(
            'GetMetrics',
            None,
            Gio.DBusCallFlags.NONE,
            -1,
            None
        )
        return dict(result[0])


# Voice Command Handler (for integration into Ara)

def handle_tfan_voice_command(command, controller):
    """
    Route voice commands to T-FAN cockpit.

    Examples:
        "show topology in landscape mode"
        "switch to work mode"
        "minimize the cockpit"
        "start training"
    """

    cmd_lower = command.lower()

    # Topology commands
    if 'show topology' in cmd_lower or 'engage topology' in cmd_lower:
        mode = None
        if 'barcode' in cmd_lower:
            mode = 'barcode'
        elif 'landscape' in cmd_lower:
            mode = 'landscape'
        elif 'poincare' in cmd_lower or 'poincaré' in cmd_lower:
            mode = 'poincare'
        elif 'pareto' in cmd_lower:
            mode = 'pareto'

        fullscreen = 'fullscreen' in cmd_lower or 'full screen' in cmd_lower
        controller.show_topology(mode, fullscreen)
        return f"Engaging topology visualization" + (f" in {mode} mode" if mode else "")

    elif 'hide topology' in cmd_lower:
        controller.hide_topology()
        return "Topology visualization hidden"

    # Workspace mode
    elif 'work mode' in cmd_lower:
        controller.set_workspace_mode('work')
        return "Switching to work mode"

    elif 'relax mode' in cmd_lower or 'relaxation mode' in cmd_lower:
        controller.set_workspace_mode('relax')
        return "Switching to relaxation mode"

    # View switching
    elif 'show dashboard' in cmd_lower:
        controller.switch_view('dashboard')
        return "Opening dashboard"

    elif 'show pareto' in cmd_lower:
        controller.switch_view('pareto')
        return "Opening Pareto optimization view"

    elif 'show training' in cmd_lower:
        controller.switch_view('training')
        return "Opening training monitor"

    # Window control
    elif 'minimize' in cmd_lower:
        controller.minimize()
        return "Minimizing cockpit"

    elif 'restore' in cmd_lower or 'show cockpit' in cmd_lower:
        controller.restore()
        return "Restoring cockpit"

    elif 'fullscreen' in cmd_lower:
        controller.toggle_fullscreen()
        return "Toggling fullscreen"

    # Training control
    elif 'start training' in cmd_lower:
        controller.start_training()
        return "Initiating training session"

    elif 'stop training' in cmd_lower:
        controller.stop_training()
        return "Stopping training session"

    else:
        return "Command not recognized"


# Example usage / test

if __name__ == '__main__':
    import sys

    controller = TFANController()

    if len(sys.argv) > 1:
        command = ' '.join(sys.argv[1:])
        response = handle_tfan_voice_command(command, controller)
        print(f"\n[Ara]: {response}")
    else:
        print("T-FAN D-Bus Controller")
        print("\nExample commands:")
        print("  python dbus_client_example.py show topology in landscape mode")
        print("  python dbus_client_example.py switch to work mode")
        print("  python dbus_client_example.py minimize")
        print("\nCurrent status:")
        print(f"  View: {controller.get_current_view()}")
        print(f"  Mode: {controller.get_workspace_mode()}")
