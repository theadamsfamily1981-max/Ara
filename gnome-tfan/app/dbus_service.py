#!/usr/bin/env python3
"""
D-Bus Service for T-FAN GNOME App

Provides a D-Bus interface for external control (primarily Ara avatar system).
Allows voice commands to control the GNOME cockpit, switch views, change themes, etc.

D-Bus Interface: com.quanta.tfan
Object Path: /com/quanta/tfan
"""

import gi
gi.require_version('Gio', '2.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gio, GLib

DBUS_INTERFACE = """
<node>
  <interface name='com.quanta.tfan.Control'>
    <!-- View Control -->
    <method name='SwitchView'>
      <arg type='s' name='view' direction='in'/>
      <arg type='b' name='success' direction='out'/>
    </method>

    <method name='GetCurrentView'>
      <arg type='s' name='view' direction='out'/>
    </method>

    <!-- Workspace/Mode Control -->
    <method name='SetWorkspaceMode'>
      <arg type='s' name='mode' direction='in'/>
      <arg type='b' name='success' direction='out'/>
    </method>

    <method name='GetWorkspaceMode'>
      <arg type='s' name='mode' direction='out'/>
    </method>

    <!-- Topology Screensaver Control -->
    <method name='ShowTopology'>
      <arg type='s' name='mode' direction='in'/>
      <arg type='b' name='fullscreen' direction='in'/>
      <arg type='b' name='success' direction='out'/>
    </method>

    <method name='HideTopology'>
      <arg type='b' name='success' direction='out'/>
    </method>

    <method name='SetTopologyMode'>
      <arg type='s' name='mode' direction='in'/>
      <arg type='b' name='success' direction='out'/>
    </method>

    <!-- Metrics Panel Control -->
    <method name='SetMetricsView'>
      <arg type='s' name='view' direction='in'/>
      <arg type='b' name='success' direction='out'/>
    </method>

    <method name='ShowMetrics'>
      <arg type='as' name='metrics' direction='in'/>
      <arg type='b' name='success' direction='out'/>
    </method>

    <method name='HideMetrics'>
      <arg type='as' name='metrics' direction='in'/>
      <arg type='b' name='success' direction='out'/>
    </method>

    <!-- Window Control -->
    <method name='ToggleFullscreen'>
      <arg type='b' name='success' direction='out'/>
    </method>

    <method name='Minimize'>
      <arg type='b' name='success' direction='out'/>
    </method>

    <method name='Restore'>
      <arg type='b' name='success' direction='out'/>
    </method>

    <!-- Training Control -->
    <method name='StartTraining'>
      <arg type='s' name='config' direction='in'/>
      <arg type='b' name='success' direction='out'/>
    </method>

    <method name='StopTraining'>
      <arg type='b' name='success' direction='out'/>
    </method>

    <!-- Status Queries -->
    <method name='GetStatus'>
      <arg type='a{sv}' name='status' direction='out'/>
    </method>

    <method name='GetMetrics'>
      <arg type='a{sv}' name='metrics' direction='out'/>
    </method>

    <!-- Signals (emitted by T-FAN to notify Ara) -->
    <signal name='ViewChanged'>
      <arg type='s' name='view'/>
    </signal>

    <signal name='WorkspaceModeChanged'>
      <arg type='s' name='mode'/>
    </signal>

    <signal name='TrainingStarted'>
      <arg type='s' name='config'/>
    </signal>

    <signal name='TrainingStopped'>
      <arg type='s' name='reason'/>
    </signal>

    <signal name='MetricsUpdated'>
      <arg type='a{sv}' name='metrics'/>
    </signal>

    <signal name='AlertRaised'>
      <arg type='s' name='level'/>
      <arg type='s' name='message'/>
    </signal>
  </interface>
</node>
"""


class TFANDBusService:
    """D-Bus service for T-FAN GNOME app control."""

    def __init__(self, app):
        """
        Initialize D-Bus service.

        Args:
            app: The main TFANGnomeApp instance
        """
        self.app = app
        self.connection = None
        self.registration_id = None
        self.workspace_mode = 'work'  # 'work' or 'relax'

        # Parse interface XML
        self.node_info = Gio.DBusNodeInfo.new_for_xml(DBUS_INTERFACE)
        self.interface_info = self.node_info.interfaces[0]

    def register(self):
        """Register the D-Bus service on the session bus."""
        try:
            self.connection = Gio.bus_get_sync(Gio.BusType.SESSION, None)

            self.registration_id = self.connection.register_object(
                '/com/quanta/tfan',
                self.interface_info,
                self._handle_method_call,
                None,  # get_property
                None   # set_property
            )

            # Request well-known name
            Gio.bus_own_name_on_connection(
                self.connection,
                'com.quanta.tfan',
                Gio.BusNameOwnerFlags.NONE,
                None,  # name_acquired_closure
                None   # name_lost_closure
            )

            print("✓ D-Bus service registered: com.quanta.tfan")
            return True

        except Exception as e:
            print(f"✗ Failed to register D-Bus service: {e}")
            return False

    def unregister(self):
        """Unregister the D-Bus service."""
        if self.connection and self.registration_id:
            self.connection.unregister_object(self.registration_id)
            self.registration_id = None

    def _handle_method_call(self, connection, sender, object_path,
                           interface_name, method_name, parameters, invocation):
        """Handle D-Bus method calls."""

        try:
            # View Control Methods
            if method_name == 'SwitchView':
                view = parameters[0]
                success = self._switch_view(view)
                invocation.return_value(GLib.Variant('(b)', (success,)))

            elif method_name == 'GetCurrentView':
                current = self.app.content_stack.get_visible_child_name() or 'dashboard'
                invocation.return_value(GLib.Variant('(s)', (current,)))

            # Workspace/Mode Control
            elif method_name == 'SetWorkspaceMode':
                mode = parameters[0]
                success = self._set_workspace_mode(mode)
                invocation.return_value(GLib.Variant('(b)', (success,)))

            elif method_name == 'GetWorkspaceMode':
                invocation.return_value(GLib.Variant('(s)', (self.workspace_mode,)))

            # Topology Screensaver Control
            elif method_name == 'ShowTopology':
                mode = parameters[0]
                fullscreen = parameters[1]
                success = self._show_topology(mode, fullscreen)
                invocation.return_value(GLib.Variant('(b)', (success,)))

            elif method_name == 'HideTopology':
                success = self._hide_topology()
                invocation.return_value(GLib.Variant('(b)', (success,)))

            elif method_name == 'SetTopologyMode':
                mode = parameters[0]
                success = self._set_topology_mode(mode)
                invocation.return_value(GLib.Variant('(b)', (success,)))

            # Metrics Panel Control
            elif method_name == 'SetMetricsView':
                view = parameters[0]
                success = self._set_metrics_view(view)
                invocation.return_value(GLib.Variant('(b)', (success,)))

            # Window Control
            elif method_name == 'ToggleFullscreen':
                success = self._toggle_fullscreen()
                invocation.return_value(GLib.Variant('(b)', (success,)))

            elif method_name == 'Minimize':
                self.app.minimize()
                invocation.return_value(GLib.Variant('(b)', (True,)))

            elif method_name == 'Restore':
                self.app.present()
                invocation.return_value(GLib.Variant('(b)', (True,)))

            # Training Control
            elif method_name == 'StartTraining':
                config = parameters[0]
                success = self._start_training(config)
                invocation.return_value(GLib.Variant('(b)', (success,)))

            elif method_name == 'StopTraining':
                success = self._stop_training()
                invocation.return_value(GLib.Variant('(b)', (success,)))

            # Status Queries
            elif method_name == 'GetStatus':
                status = self._get_status()
                invocation.return_value(GLib.Variant('(a{sv})', (status,)))

            elif method_name == 'GetMetrics':
                metrics = self._get_metrics()
                invocation.return_value(GLib.Variant('(a{sv})', (metrics,)))

            else:
                invocation.return_error_literal(
                    Gio.DBusError,
                    Gio.DBusError.UNKNOWN_METHOD,
                    f"Method {method_name} not implemented"
                )

        except Exception as e:
            invocation.return_error_literal(
                Gio.DBusError,
                Gio.DBusError.FAILED,
                f"Error handling {method_name}: {str(e)}"
            )

    # Implementation methods

    def _switch_view(self, view):
        """Switch to a different view."""
        valid_views = ['dashboard', 'pareto', 'training', 'screensaver', 'config', 'repo']
        if view in valid_views:
            self.app.content_stack.set_visible_child_name(view)
            self.emit_signal('ViewChanged', GLib.Variant('(s)', (view,)))
            return True
        return False

    def _set_workspace_mode(self, mode):
        """Set workspace mode (work/relax) and update theme."""
        if mode in ['work', 'relax']:
            self.workspace_mode = mode

            # Apply theme based on mode
            if hasattr(self.app, 'apply_workspace_theme'):
                self.app.apply_workspace_theme(mode)

            self.emit_signal('WorkspaceModeChanged', GLib.Variant('(s)', (mode,)))
            return True
        return False

    def _show_topology(self, mode, fullscreen):
        """Show topology visualization."""
        self._switch_view('screensaver')

        # Set topology mode if specified
        if mode:
            self._set_topology_mode(mode)

        # Toggle fullscreen if requested
        if fullscreen and not self.app.is_fullscreen():
            self.app.fullscreen()

        return True

    def _hide_topology(self):
        """Hide topology visualization."""
        if self.app.is_fullscreen():
            self.app.unfullscreen()
        self._switch_view('dashboard')
        return True

    def _set_topology_mode(self, mode):
        """Set topology visualization mode."""
        valid_modes = ['barcode', 'landscape', 'poincare', 'pareto']
        if mode.lower() in valid_modes and hasattr(self.app, 'screensaver_mode_combo'):
            mode_index = valid_modes.index(mode.lower())
            self.app.screensaver_mode_combo.set_active(mode_index)
            return True
        return False

    def _set_metrics_view(self, view):
        """Set metrics display view."""
        # This would control what metrics are shown in the side panel
        # Implementation depends on metrics panel design
        return True

    def _toggle_fullscreen(self):
        """Toggle fullscreen mode."""
        if self.app.is_fullscreen():
            self.app.unfullscreen()
        else:
            self.app.fullscreen()
        return True

    def _start_training(self, config):
        """Start training with specified config."""
        # This would trigger the training system
        self.emit_signal('TrainingStarted', GLib.Variant('(s)', (config,)))
        return True

    def _stop_training(self):
        """Stop training."""
        self.emit_signal('TrainingStopped', GLib.Variant('(s)', ('user_requested',)))
        return True

    def _get_status(self):
        """Get current system status."""
        return {
            'view': GLib.Variant('s', self.app.content_stack.get_visible_child_name() or 'dashboard'),
            'workspace_mode': GLib.Variant('s', self.workspace_mode),
            'fullscreen': GLib.Variant('b', self.app.is_fullscreen()),
            'training_active': GLib.Variant('b', False)  # TODO: Get real status
        }

    def _get_metrics(self):
        """Get current metrics."""
        # TODO: Pull from actual metrics source
        return {
            'accuracy': GLib.Variant('d', 0.0),
            'latency_ms': GLib.Variant('d', 0.0),
            'epr_cv': GLib.Variant('d', 0.0),
            'topo_gap': GLib.Variant('d', 0.0)
        }

    def emit_signal(self, signal_name, parameters):
        """Emit a D-Bus signal."""
        if self.connection:
            try:
                self.connection.emit_signal(
                    None,  # destination (broadcast)
                    '/com/quanta/tfan',
                    'com.quanta.tfan.Control',
                    signal_name,
                    parameters
                )
            except Exception as e:
                print(f"Failed to emit signal {signal_name}: {e}")

    def emit_metrics_update(self, metrics_dict):
        """Emit MetricsUpdated signal with current metrics."""
        params = {}
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                params[key] = GLib.Variant('d', value)
            elif isinstance(value, int):
                params[key] = GLib.Variant('i', value)
            elif isinstance(value, bool):
                params[key] = GLib.Variant('b', value)
            elif isinstance(value, str):
                params[key] = GLib.Variant('s', value)

        self.emit_signal('MetricsUpdated', GLib.Variant('(a{sv})', (params,)))

    def emit_alert(self, level, message):
        """Emit alert signal for Ara to announce."""
        self.emit_signal('AlertRaised', GLib.Variant('(ss)', (level, message)))
