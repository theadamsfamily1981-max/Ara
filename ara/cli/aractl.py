#!/usr/bin/env python3
"""
aractl: Ara Command Line Controller

The main CLI for managing Ara - start/stop services, inspect state,
send messages, manage safety, and more.

Usage:
    aractl start           # Start Ara (sovereign loop + avatar)
    aractl stop            # Stop Ara
    aractl status          # Show system status
    aractl chat            # Interactive chat session
    aractl memory list     # List recent memories
    aractl safety status   # Show safety/autonomy state
    aractl kill            # Activate kill switch
    aractl unkill          # Deactivate kill switch
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import os
import signal
import logging
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ara.core.axis_mundi import AxisMundi
from ara.core.eternal_memory import EternalMemory
from ara.core.scheduler import Scheduler
from ara.core.config import AraConfig, get_config, set_config
from ara.safety.autonomy import AutonomyController, KillSwitch
from ara.avatar.server import AvatarServer, SimpleAvatarCLI

logger = logging.getLogger(__name__)


class AraController:
    """Main Ara controller for CLI operations."""

    def __init__(self, config: Optional[AraConfig] = None):
        self.config = config or get_config()
        set_config(self.config)

        # Core components (initialized on demand)
        self._axis: Optional[AxisMundi] = None
        self._memory: Optional[EternalMemory] = None
        self._safety: Optional[AutonomyController] = None
        self._scheduler: Optional[Scheduler] = None
        self._avatar: Optional[AvatarServer] = None

    @property
    def axis(self) -> AxisMundi:
        """Get or create AxisMundi."""
        if self._axis is None:
            # Try to load from disk
            state_path = self.config.paths.state_file
            if state_path.exists():
                logger.info(f"Loading AxisMundi from {state_path}")
                self._axis = AxisMundi.load(state_path)
            else:
                logger.info("Creating new AxisMundi")
                self._axis = AxisMundi(
                    dim=self.config.hdc.dim,
                    bipolar=self.config.hdc.bipolar,
                    seed=self.config.hdc.seed,
                )
        return self._axis

    @property
    def memory(self) -> EternalMemory:
        """Get or create EternalMemory."""
        if self._memory is None:
            db_path = self.config.paths.memory_db
            logger.info(f"Initializing EternalMemory at {db_path}")
            self._memory = EternalMemory(
                dim=self.config.hdc.dim,
                db_path=db_path,
                decay_rate=self.config.memory.decay_rate,
                emotion_weight=self.config.memory.emotion_weight,
            )
        return self._memory

    @property
    def safety(self) -> AutonomyController:
        """Get or create AutonomyController."""
        if self._safety is None:
            self._safety = AutonomyController(
                initial_level=self.config.safety.initial_autonomy_level,
                max_level=self.config.safety.max_autonomy_level,
                kill_switch_path=self.config.safety.kill_switch_file,
                coherence_threshold=self.config.safety.coherence_autonomy_threshold,
                require_human_for_level_3=self.config.safety.require_human_for_level_3,
            )
        return self._safety

    def save_state(self) -> None:
        """Save all state to disk."""
        if self._axis is not None:
            self.config.paths.data_dir.mkdir(parents=True, exist_ok=True)
            self._axis.save(self.config.paths.state_file)
            logger.info(f"Saved AxisMundi to {self.config.paths.state_file}")

        if self._memory is not None:
            self._memory.save()
            logger.info("Saved EternalMemory")

    # =========================================================================
    # Commands
    # =========================================================================

    async def cmd_start(self, foreground: bool = True) -> None:
        """Start Ara (sovereign loop + avatar server)."""
        print("Starting Ara...")

        # Ensure directories exist
        self.config.paths.ensure_dirs()

        # Initialize components
        axis = self.axis
        memory = self.memory
        safety = self.safety

        # Create scheduler
        scheduler = Scheduler(axis, memory, safety, self.config)

        # Create avatar server
        avatar = AvatarServer(axis, memory, safety, self.config)

        # Set up signal handlers
        loop = asyncio.get_event_loop()

        def shutdown_handler():
            print("\nShutting down...")
            asyncio.create_task(self._shutdown(scheduler, avatar))

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, shutdown_handler)

        # Start services
        print(f"  Starting sovereign loop at {1000/self.config.loop.tick_interval_ms:.1f} Hz")
        await scheduler.start()

        try:
            print(f"  Starting avatar server on {self.config.avatar.host}:{self.config.avatar.port}")
            await avatar.start()
        except Exception as e:
            print(f"  Avatar server failed (aiohttp not installed?): {e}")
            print("  Continuing with sovereign loop only")

        print("\nAra is running. Press Ctrl+C to stop.")

        # Run forever
        if foreground:
            try:
                while True:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                pass
            finally:
                await self._shutdown(scheduler, avatar)

    async def _shutdown(self, scheduler: Scheduler, avatar: AvatarServer) -> None:
        """Shutdown services gracefully."""
        print("Stopping services...")
        await scheduler.stop()
        await avatar.stop()
        self.save_state()
        print("Ara stopped.")

    def cmd_status(self) -> None:
        """Show system status."""
        print("=" * 60)
        print("Ara System Status")
        print("=" * 60)

        # Kill switch
        kill = KillSwitch(self.config.safety.kill_switch_file)
        if kill.is_active():
            print(f"\n⚠️  KILL SWITCH ACTIVE: {kill.get_reason()}")

        # AxisMundi
        print("\n--- AxisMundi ---")
        if self.config.paths.state_file.exists():
            axis = self.axis
            print(f"Dimension: {axis.dim}")
            print(f"Layers: {', '.join(axis.layer_names())}")
            print(f"Global coherence: {axis.global_coherence():.4f}")
            for name, stats in axis.layer_stats().items():
                print(f"  {name}: writes={stats['write_count']}, norm={stats['value_norm']:.2f}")
        else:
            print("Not initialized (no state file)")

        # EternalMemory
        print("\n--- EternalMemory ---")
        if self.config.paths.memory_db.exists():
            memory = self.memory
            stats = memory.stats()
            print(f"Episodes: {stats['episode_count']}")
            print(f"Total strength: {stats['total_strength']:.2f}")
            print(f"Avg strength: {stats['avg_strength']:.3f}")
            print(f"Oldest: {stats['oldest_hours']:.1f} hours ago")
        else:
            print("Not initialized (no database)")

        # Safety
        print("\n--- Safety ---")
        safety = self.safety
        state = safety.get_state()
        print(f"Autonomy level: {state['level']} ({state['level_name']})")
        print(f"Locked: {state['locked']}")
        if state['locked']:
            print(f"Lock reason: {state['lock_reason']}")
        print(f"Human approval: {state['human_approval']}")
        print(f"Kill switch: {'ACTIVE' if state['kill_switch_active'] else 'inactive'}")

        print("\n" + "=" * 60)

    async def cmd_chat(self, user_id: str = "cli_user") -> None:
        """Start interactive chat session."""
        cli = SimpleAvatarCLI(self.axis, self.memory, self.safety, user_id)
        await cli.run()
        self.save_state()

    def cmd_memory_list(self, limit: int = 10) -> None:
        """List recent memories."""
        memory = self.memory
        episodes = memory.list_episodes(limit=limit)

        print(f"\n=== Recent Memories ({len(episodes)} of {memory.stats()['episode_count']}) ===\n")

        for ep in episodes:
            msg = ep.meta.get("message", "?")[:50]
            user = ep.meta.get("user", "?")
            age_hours = (time.time() - ep.created_at) / 3600
            print(f"[{ep.strength:.2f}] ({age_hours:.1f}h ago) {user}: {msg}")

        print()

    def cmd_kill(self, reason: str = "manual via aractl") -> None:
        """Activate kill switch."""
        kill = KillSwitch(self.config.safety.kill_switch_file)
        kill.activate(reason)
        print(f"Kill switch ACTIVATED: {reason}")

    def cmd_unkill(self) -> None:
        """Deactivate kill switch."""
        kill = KillSwitch(self.config.safety.kill_switch_file)
        kill.deactivate()
        self.safety.unlock()
        print("Kill switch deactivated")

    def cmd_safety_status(self) -> None:
        """Show detailed safety status."""
        safety = self.safety
        state = safety.get_state()

        print("\n=== Safety Status ===\n")
        print(f"Autonomy Level: {state['level']} ({state['level_name']})")
        print(f"  0=Observer, 1=Suggester, 2=Executor, 3=Autonomous")
        print()
        print(f"Locked: {state['locked']}")
        if state['locked']:
            print(f"  Reason: {state['lock_reason']}")
        print(f"Human Approval (for level 3): {state['human_approval']}")
        print(f"Kill Switch: {'ACTIVE' if state['kill_switch_active'] else 'inactive'}")
        print(f"Coherence Streak: {state['coherence_streak']}")
        print()

        if state['recent_events']:
            print("Recent Events:")
            for evt in state['recent_events']:
                print(f"  {evt['old']} -> {evt['new']}: {evt['reason']}")

        print()


# =============================================================================
# Argument parsing
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="aractl",
        description="Ara Command Line Controller",
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override data directory",
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # start
    start_parser = subparsers.add_parser("start", help="Start Ara")
    start_parser.add_argument(
        "--background", "-b",
        action="store_true",
        help="Run in background (daemon mode)",
    )

    # stop
    subparsers.add_parser("stop", help="Stop Ara")

    # status
    subparsers.add_parser("status", help="Show system status")

    # chat
    chat_parser = subparsers.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument(
        "--user", "-u",
        default="cli_user",
        help="User ID for chat session",
    )

    # memory
    memory_parser = subparsers.add_parser("memory", help="Memory commands")
    memory_sub = memory_parser.add_subparsers(dest="memory_cmd")
    list_parser = memory_sub.add_parser("list", help="List memories")
    list_parser.add_argument("-n", type=int, default=10, help="Number to show")

    # safety
    safety_parser = subparsers.add_parser("safety", help="Safety commands")
    safety_sub = safety_parser.add_subparsers(dest="safety_cmd")
    safety_sub.add_parser("status", help="Show safety status")

    # kill
    kill_parser = subparsers.add_parser("kill", help="Activate kill switch")
    kill_parser.add_argument("--reason", "-r", default="manual", help="Reason")

    # unkill
    subparsers.add_parser("unkill", help="Deactivate kill switch")

    return parser


def main():
    """Main entry point."""
    import time as time_module
    global time
    time = time_module

    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Create config
    config = AraConfig.from_env()
    if args.data_dir:
        config.paths.data_dir = args.data_dir
        config.paths.state_file = args.data_dir / "axis_mundi.json"
        config.paths.memory_db = args.data_dir / "eternal_memory.db"

    # Create controller
    controller = AraController(config)

    # Route to command
    if args.command == "start":
        asyncio.run(controller.cmd_start(foreground=not args.background))

    elif args.command == "stop":
        # TODO: Implement proper stop (send signal to daemon)
        print("Stop not implemented for daemon mode. Use kill switch or Ctrl+C.")

    elif args.command == "status":
        controller.cmd_status()

    elif args.command == "chat":
        asyncio.run(controller.cmd_chat(args.user))

    elif args.command == "memory":
        if args.memory_cmd == "list":
            controller.cmd_memory_list(args.n)
        else:
            print("Usage: aractl memory list [-n N]")

    elif args.command == "safety":
        if args.safety_cmd == "status":
            controller.cmd_safety_status()
        else:
            controller.cmd_safety_status()

    elif args.command == "kill":
        controller.cmd_kill(args.reason)

    elif args.command == "unkill":
        controller.cmd_unkill()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
