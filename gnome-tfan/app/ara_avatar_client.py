#!/usr/bin/env python3
"""
Ara Avatar D-Bus Client

Communicates with Ara avatar system to control appearance.
Used by cockpit HUD avatar view.

D-Bus Interface (Ara side - to be implemented in Ara repo):
    Bus Name: com.quanta.ara
    Object Path: /com/quanta/ara/avatar
    Interface: com.quanta.ara.Avatar

Methods:
    SetProfile(profile: str) → bool
    SetStyle(style: str) → bool
    SetMood(mood: str) → bool
    ApplyChanges() → bool
    SavePreset(name: str) → bool
    LoadPreset(name: str) → bool
    GetCurrentConfig() → dict

Signals:
    AvatarChanged(config: dict)
"""

import gi
gi.require_version('Gio', '2.0')
gi.require_version('GLib', '2.0')
from gi.repository import Gio, GLib


class AraAvatarClient:
    """Client for controlling Ara avatar via D-Bus."""

    def __init__(self):
        """Initialize D-Bus connection to Ara avatar service."""
        self.proxy = None
        self.connected = False
        self._connect()

    def _connect(self):
        """Connect to Ara avatar D-Bus service."""
        try:
            connection = Gio.bus_get_sync(Gio.BusType.SESSION, None)

            self.proxy = Gio.DBusProxy.new_sync(
                connection,
                Gio.DBusProxyFlags.NONE,
                None,
                'com.quanta.ara',
                '/com/quanta/ara/avatar',
                'com.quanta.ara.Avatar',
                None
            )

            # Subscribe to avatar change signals
            self.proxy.connect('g-signal', self._on_avatar_signal)

            self.connected = True
            print("✓ Connected to Ara avatar service")

        except Exception as e:
            self.connected = False
            print(f"✗ Failed to connect to Ara avatar service: {e}")
            print("  (Ara may not be running)")

    def _on_avatar_signal(self, proxy, sender_name, signal_name, parameters):
        """Handle signals from Ara avatar."""
        if signal_name == 'AvatarChanged':
            config = parameters[0]
            print(f"[Ara] Avatar changed: {dict(config)}")

    def set_profile(self, profile: str) -> bool:
        """
        Set avatar profile.

        Args:
            profile: Profile name (e.g., 'Professional', 'Casual', 'Scientist')

        Returns:
            bool: Success
        """
        if not self.connected:
            print(f"[Ara] Not connected - Profile would be set to: {profile}")
            return False

        try:
            result = self.proxy.call_sync(
                'SetProfile',
                GLib.Variant('(s)', (profile,)),
                Gio.DBusCallFlags.NONE,
                -1,
                None
            )
            return result[0]
        except Exception as e:
            print(f"[Ara] Error setting profile: {e}")
            return False

    def set_style(self, style: str) -> bool:
        """
        Set avatar style.

        Args:
            style: Style name (e.g., 'Realistic', 'Stylized', 'Anime')

        Returns:
            bool: Success
        """
        if not self.connected:
            print(f"[Ara] Not connected - Style would be set to: {style}")
            return False

        try:
            result = self.proxy.call_sync(
                'SetStyle',
                GLib.Variant('(s)', (style,)),
                Gio.DBusCallFlags.NONE,
                -1,
                None
            )
            return result[0]
        except Exception as e:
            print(f"[Ara] Error setting style: {e}")
            return False

    def set_mood(self, mood: str) -> bool:
        """
        Set avatar mood.

        Args:
            mood: Mood name (e.g., 'Neutral', 'Focused', 'Friendly')

        Returns:
            bool: Success
        """
        if not self.connected:
            print(f"[Ara] Not connected - Mood would be set to: {mood}")
            return False

        try:
            result = self.proxy.call_sync(
                'SetMood',
                GLib.Variant('(s)', (mood,)),
                Gio.DBusCallFlags.NONE,
                -1,
                None
            )
            return result[0]
        except Exception as e:
            print(f"[Ara] Error setting mood: {e}")
            return False

    def apply_changes(self) -> bool:
        """
        Apply pending avatar changes.

        Returns:
            bool: Success
        """
        if not self.connected:
            print(f"[Ara] Not connected - Changes would be applied")
            return False

        try:
            result = self.proxy.call_sync(
                'ApplyChanges',
                None,
                Gio.DBusCallFlags.NONE,
                -1,
                None
            )
            return result[0]
        except Exception as e:
            print(f"[Ara] Error applying changes: {e}")
            return False

    def save_preset(self, name: str) -> bool:
        """
        Save current avatar config as preset.

        Args:
            name: Preset name

        Returns:
            bool: Success
        """
        if not self.connected:
            print(f"[Ara] Not connected - Preset '{name}' would be saved")
            return False

        try:
            result = self.proxy.call_sync(
                'SavePreset',
                GLib.Variant('(s)', (name,)),
                Gio.DBusCallFlags.NONE,
                -1,
                None
            )
            return result[0]
        except Exception as e:
            print(f"[Ara] Error saving preset: {e}")
            return False

    def load_preset(self, name: str) -> bool:
        """
        Load avatar preset.

        Args:
            name: Preset name

        Returns:
            bool: Success
        """
        if not self.connected:
            print(f"[Ara] Not connected - Preset '{name}' would be loaded")
            return False

        try:
            result = self.proxy.call_sync(
                'LoadPreset',
                GLib.Variant('(s)', (name,)),
                Gio.DBusCallFlags.NONE,
                -1,
                None
            )
            return result[0]
        except Exception as e:
            print(f"[Ara] Error loading preset: {e}")
            return False

    def get_current_config(self) -> dict:
        """
        Get current avatar configuration.

        Returns:
            dict: Current config (profile, style, mood)
        """
        if not self.connected:
            return {}

        try:
            result = self.proxy.call_sync(
                'GetCurrentConfig',
                None,
                Gio.DBusCallFlags.NONE,
                -1,
                None
            )
            return dict(result[0])
        except Exception as e:
            print(f"[Ara] Error getting config: {e}")
            return {}

    def set_all(self, profile: str, style: str, mood: str) -> bool:
        """
        Set profile, style, and mood, then apply changes.

        Args:
            profile: Profile name
            style: Style name
            mood: Mood name

        Returns:
            bool: Success
        """
        success = True
        success &= self.set_profile(profile)
        success &= self.set_style(style)
        success &= self.set_mood(mood)
        success &= self.apply_changes()

        return success


# Example usage / testing
if __name__ == '__main__':
    import sys

    client = AraAvatarClient()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'set':
            if len(sys.argv) >= 5:
                profile = sys.argv[2]
                style = sys.argv[3]
                mood = sys.argv[4]
                result = client.set_all(profile, style, mood)
                print(f"Result: {'Success' if result else 'Failed'}")
            else:
                print("Usage: ara_avatar_client.py set <profile> <style> <mood>")

        elif command == 'get':
            config = client.get_current_config()
            print(f"Current config: {config}")

        elif command == 'preset':
            if len(sys.argv) >= 4:
                action = sys.argv[2]  # save or load
                name = sys.argv[3]

                if action == 'save':
                    result = client.save_preset(name)
                    print(f"Saved preset '{name}': {'Success' if result else 'Failed'}")
                elif action == 'load':
                    result = client.load_preset(name)
                    print(f"Loaded preset '{name}': {'Success' if result else 'Failed'}")
            else:
                print("Usage: ara_avatar_client.py preset [save|load] <name>")

        else:
            print(f"Unknown command: {command}")

    else:
        print("Ara Avatar D-Bus Client")
        print("\nCommands:")
        print("  set <profile> <style> <mood>  - Set avatar appearance")
        print("  get                            - Get current config")
        print("  preset save <name>             - Save current as preset")
        print("  preset load <name>             - Load preset")
        print("\nExamples:")
        print("  python ara_avatar_client.py set Professional Realistic Focused")
        print("  python ara_avatar_client.py preset save 'Work Mode'")
        print("  python ara_avatar_client.py get")
