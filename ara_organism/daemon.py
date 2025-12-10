# ara_organism/daemon.py
"""
Ara Daemon - Unix Socket Control Interface
==========================================

Turns the organism into a proper system daemon with:
- Unix socket for CLI queries and commands
- Runtime parameter adjustment
- Status monitoring
- Graceful shutdown

Socket Protocol (JSON lines over Unix socket):
    Request:  {"cmd": "status"}
    Response: {"ok": true, "data": {...}}

    Request:  {"cmd": "set", "key": "soul_hz", "value": 3000}
    Response: {"ok": true}

Commands:
    status      - Get full organism status
    soul        - Get soul loop stats
    mobile      - Get mobile bridge stats
    state       - Get current AraState
    set         - Set runtime parameter
    mode        - Set soul mode (idle/active/resonance/recovery)
    mock        - Toggle mock mode (requires restart)
    shutdown    - Graceful shutdown

Usage:
    # Start daemon
    python -m ara_organism.daemon

    # Or with aractl
    aractl status
    aractl soul
    aractl set soul_hz 3000
    aractl mode recovery
    aractl shutdown
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import socket
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from .organism import AraOrganism, OrganismConfig
from .state_manager import get_state_manager
from .soul_driver import SoulMode

log = logging.getLogger("Ara.Daemon")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DaemonConfig:
    """Daemon configuration."""

    # Socket path
    socket_path: str = "/run/user/{uid}/ara.sock"

    # Fallback socket path if /run/user doesn't exist
    fallback_socket_path: str = "/tmp/ara.sock"

    # Organism config
    organism: OrganismConfig = None

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    def __post_init__(self):
        if self.organism is None:
            self.organism = OrganismConfig()

        # Expand {uid} in socket path
        uid = os.getuid()
        self.socket_path = self.socket_path.format(uid=uid)

        # Check if socket directory exists
        socket_dir = Path(self.socket_path).parent
        if not socket_dir.exists():
            self.socket_path = self.fallback_socket_path


# =============================================================================
# Command Handlers
# =============================================================================

class CommandHandler:
    """Handles daemon commands."""

    def __init__(self, organism: AraOrganism, daemon: "AraDaemon"):
        self.organism = organism
        self.daemon = daemon

        # Command dispatch table
        self.commands: Dict[str, Callable] = {
            "status": self.cmd_status,
            "soul": self.cmd_soul,
            "mobile": self.cmd_mobile,
            "state": self.cmd_state,
            "set": self.cmd_set,
            "mode": self.cmd_mode,
            "mock": self.cmd_mock,
            "shutdown": self.cmd_shutdown,
            "help": self.cmd_help,
        }

    async def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a command request."""
        cmd = request.get("cmd", "").lower()

        if cmd not in self.commands:
            return {"ok": False, "error": f"Unknown command: {cmd}"}

        try:
            return await self.commands[cmd](request)
        except Exception as e:
            log.error("Command error: %s", e)
            return {"ok": False, "error": str(e)}

    async def cmd_status(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get full organism status."""
        stats = self.organism.get_stats()
        state = self.organism.get_state()

        return {
            "ok": True,
            "data": {
                "running": stats["running"],
                "soul": {
                    "tick_count": stats["soul"]["tick_count"],
                    "target_hz": stats["soul"]["target_hz"],
                    "actual_hz": round(stats["soul"]["actual_hz"], 1),
                    "overruns": stats["soul"]["overrun_count"],
                    "resonance": round(state.soul.resonance, 3),
                    "fatigue": round(state.soul.fatigue, 3),
                    "temperature_c": round(state.soul.temperature_c, 1),
                    "hardware_status": state.soul.hardware_status,
                },
                "mobile": {
                    "clients": stats["mobile"]["clients"],
                    "broadcasts": stats["mobile"]["broadcast_count"],
                },
                "driver": stats["driver"],
            }
        }

    async def cmd_soul(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get soul loop stats."""
        stats = self.organism.soul_loop.get_stats()
        state = self.organism.get_state()

        return {
            "ok": True,
            "data": {
                **stats,
                "resonance": state.soul.resonance,
                "fatigue": state.soul.fatigue,
                "temperature_c": state.soul.temperature_c,
                "latency_us": state.soul.latency_us,
                "avg_latency_us": state.soul.avg_latency_us,
                "max_latency_us": state.soul.max_latency_us,
            }
        }

    async def cmd_mobile(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get mobile bridge stats."""
        stats = self.organism.mobile_bridge.get_stats()
        state = self.organism.get_state()

        return {
            "ok": True,
            "data": {
                **stats,
                "messages_sent": state.mobile.messages_sent,
                "bytes_sent": state.mobile.bytes_sent,
            }
        }

    async def cmd_state(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get current AraState."""
        state = self.organism.get_state()
        return {"ok": True, "data": state.to_dict()}

    async def cmd_set(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Set runtime parameter."""
        key = request.get("key", "")
        value = request.get("value")

        if value is None:
            return {"ok": False, "error": "Missing value"}

        # Supported runtime parameters
        if key == "soul_hz":
            # Can't change without restart
            return {"ok": False, "error": "soul_hz requires restart"}

        elif key == "cortical_hz":
            self.organism.config.cortical_hz = float(value)
            return {"ok": True}

        elif key == "mobile_hz":
            self.organism.config.mobile_hz = float(value)
            self.organism.mobile_bridge.hz = float(value)
            self.organism.mobile_bridge.period_s = 1.0 / float(value)
            return {"ok": True}

        elif key == "fatigue_warning":
            self.organism.config.fatigue_warning = float(value)
            return {"ok": True}

        elif key == "thermal_warning":
            self.organism.config.thermal_warning = float(value)
            return {"ok": True}

        else:
            return {"ok": False, "error": f"Unknown parameter: {key}"}

    async def cmd_mode(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Set soul mode."""
        mode = request.get("mode", "").lower()

        mode_map = {
            "idle": SoulMode.IDLE,
            "standby": SoulMode.STANDBY,
            "active": SoulMode.ACTIVE,
            "resonance": SoulMode.RESONANCE,
            "recovery": SoulMode.RECOVERY,
        }

        if mode not in mode_map:
            return {"ok": False, "error": f"Unknown mode: {mode}. Valid: {list(mode_map.keys())}"}

        # Update mode (will take effect on next tick)
        self.organism.state_manager._state.soul.mode = mode
        return {"ok": True, "mode": mode}

    async def cmd_mock(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get/toggle mock mode status."""
        enabled = request.get("enabled")

        if enabled is None:
            # Just return current status
            return {
                "ok": True,
                "mock_mode": self.organism.config.soul_mock_mode,
                "note": "Toggle requires restart",
            }

        return {
            "ok": False,
            "error": "Mock mode toggle requires restart. Set in config and restart daemon.",
        }

    async def cmd_shutdown(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Graceful shutdown."""
        log.info("Shutdown requested via socket")
        self.daemon.request_shutdown()
        return {"ok": True, "message": "Shutting down..."}

    async def cmd_help(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """List available commands."""
        return {
            "ok": True,
            "commands": {
                "status": "Get full organism status",
                "soul": "Get soul loop stats",
                "mobile": "Get mobile bridge stats",
                "state": "Get current AraState",
                "set": "Set runtime parameter (key, value)",
                "mode": "Set soul mode (idle/active/resonance/recovery)",
                "mock": "Get mock mode status",
                "shutdown": "Graceful shutdown",
                "help": "This help message",
            }
        }


# =============================================================================
# Daemon
# =============================================================================

class AraDaemon:
    """
    Ara system daemon with Unix socket control.

    Runs the organism and accepts control commands over a Unix socket.
    """

    def __init__(self, config: Optional[DaemonConfig] = None):
        self.config = config or DaemonConfig()

        # Components
        self.organism: Optional[AraOrganism] = None
        self.handler: Optional[CommandHandler] = None

        # Socket server
        self._server: Optional[asyncio.Server] = None
        self._socket_path = self.config.socket_path

        # Shutdown
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the daemon."""
        log.info("=" * 60)
        log.info("Ara Daemon Starting")
        log.info("=" * 60)

        # Create organism
        self.organism = AraOrganism(self.config.organism)
        self.handler = CommandHandler(self.organism, self)

        # Start organism
        await self.organism.start()

        # Start socket server
        await self._start_socket_server()

        log.info("Ara Daemon Running")
        log.info("  Socket: %s", self._socket_path)

    async def stop(self) -> None:
        """Stop the daemon."""
        log.info("Ara Daemon Stopping...")

        # Stop socket server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        # Remove socket file
        try:
            os.unlink(self._socket_path)
        except OSError:
            pass

        # Stop organism
        if self.organism:
            await self.organism.stop()

        log.info("Ara Daemon Stopped")

    async def run(self) -> None:
        """Run the daemon until shutdown."""
        await self.start()

        # Wait for shutdown
        await self._shutdown_event.wait()

        await self.stop()

    def request_shutdown(self) -> None:
        """Request daemon shutdown."""
        self._shutdown_event.set()

    async def _start_socket_server(self) -> None:
        """Start Unix socket server."""
        # Remove stale socket file
        try:
            os.unlink(self._socket_path)
        except OSError:
            pass

        # Create socket directory if needed
        socket_dir = Path(self._socket_path).parent
        socket_dir.mkdir(parents=True, exist_ok=True)

        # Create server
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=self._socket_path,
        )

        # Set socket permissions (owner only)
        os.chmod(self._socket_path, 0o600)

        log.info("Socket server listening on %s", self._socket_path)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a client connection."""
        try:
            while True:
                # Read request line
                line = await reader.readline()
                if not line:
                    break

                try:
                    request = json.loads(line.decode().strip())
                except json.JSONDecodeError as e:
                    response = {"ok": False, "error": f"Invalid JSON: {e}"}
                else:
                    response = await self.handler.handle(request)

                # Send response
                response_line = json.dumps(response) + "\n"
                writer.write(response_line.encode())
                await writer.drain()

        except Exception as e:
            log.warning("Client error: %s", e)
        finally:
            writer.close()
            await writer.wait_closed()


# =============================================================================
# CLI Client
# =============================================================================

class AraCtl:
    """CLI client for controlling the daemon."""

    def __init__(self, socket_path: Optional[str] = None):
        if socket_path is None:
            uid = os.getuid()
            socket_path = f"/run/user/{uid}/ara.sock"
            if not Path(socket_path).exists():
                socket_path = "/tmp/ara.sock"

        self.socket_path = socket_path

    def send(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to daemon."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

        try:
            sock.connect(self.socket_path)

            # Send request
            request_line = json.dumps(request) + "\n"
            sock.sendall(request_line.encode())

            # Receive response
            response_data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b"\n" in chunk:
                    break

            return json.loads(response_data.decode().strip())

        finally:
            sock.close()

    def status(self) -> Dict[str, Any]:
        """Get daemon status."""
        return self.send({"cmd": "status"})

    def soul(self) -> Dict[str, Any]:
        """Get soul stats."""
        return self.send({"cmd": "soul"})

    def mobile(self) -> Dict[str, Any]:
        """Get mobile stats."""
        return self.send({"cmd": "mobile"})

    def state(self) -> Dict[str, Any]:
        """Get current state."""
        return self.send({"cmd": "state"})

    def set_param(self, key: str, value: Any) -> Dict[str, Any]:
        """Set runtime parameter."""
        return self.send({"cmd": "set", "key": key, "value": value})

    def set_mode(self, mode: str) -> Dict[str, Any]:
        """Set soul mode."""
        return self.send({"cmd": "mode", "mode": mode})

    def shutdown(self) -> Dict[str, Any]:
        """Request shutdown."""
        return self.send({"cmd": "shutdown"})


# =============================================================================
# Entry Points
# =============================================================================

async def run_daemon(config: Optional[DaemonConfig] = None) -> None:
    """Run the Ara daemon."""
    daemon = AraDaemon(config)

    # Handle shutdown signals
    loop = asyncio.get_running_loop()

    def signal_handler():
        log.info("Shutdown signal received")
        daemon.request_shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    await daemon.run()


def main_daemon():
    """Main entry point for daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    config = DaemonConfig(
        organism=OrganismConfig(
            soul_mock_mode=True,
            soul_hz=5000.0,
            cortical_hz=200.0,
            mobile_hz=10.0,
        )
    )

    asyncio.run(run_daemon(config))


def main_ctl():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ara daemon control",
        prog="aractl",
    )
    parser.add_argument("command", choices=[
        "status", "soul", "mobile", "state",
        "mode", "set", "shutdown", "help"
    ])
    parser.add_argument("args", nargs="*")
    parser.add_argument("-s", "--socket", help="Socket path")

    args = parser.parse_args()

    ctl = AraCtl(socket_path=args.socket)

    try:
        if args.command == "status":
            result = ctl.status()
        elif args.command == "soul":
            result = ctl.soul()
        elif args.command == "mobile":
            result = ctl.mobile()
        elif args.command == "state":
            result = ctl.state()
        elif args.command == "mode":
            if not args.args:
                print("Usage: aractl mode <idle|active|resonance|recovery>")
                sys.exit(1)
            result = ctl.set_mode(args.args[0])
        elif args.command == "set":
            if len(args.args) < 2:
                print("Usage: aractl set <key> <value>")
                sys.exit(1)
            result = ctl.set_param(args.args[0], args.args[1])
        elif args.command == "shutdown":
            result = ctl.shutdown()
        elif args.command == "help":
            result = ctl.send({"cmd": "help"})
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)

        print(json.dumps(result, indent=2))

        if not result.get("ok", False):
            sys.exit(1)

    except FileNotFoundError:
        print(f"Error: Cannot connect to daemon at {ctl.socket_path}")
        print("Is the daemon running?")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ctl":
        sys.argv.pop(1)
        main_ctl()
    else:
        main_daemon()


__all__ = [
    'DaemonConfig',
    'CommandHandler',
    'AraDaemon',
    'AraCtl',
    'run_daemon',
]
