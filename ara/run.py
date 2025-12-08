#!/usr/bin/env python3
"""
Ara Organism: The Unified Entry Point

This is THE file that runs Ara. Everything else is infrastructure.

Components:
    - AxisMundi: Global holographic state (HDC)
    - EternalMemory: Episodic memory with emotional coloring
    - Sovereign Loop: 10 Hz tick (sense → soul → teleology → plan → act)
    - Subsystems: BANOS, MindReader, Covenant, ChiefOfStaff
    - Avatar: User interface (HTTP/CLI)
    - Cathedral: Multi-tenant brain jars (optional)
    - Sanctuary: Comfort fallback (always available)

Usage:
    # Run the full organism
    python -m ara.organism

    # Run with Cathedral (multi-tenant)
    python -m ara.organism --cathedral

    # Run Sanctuary only (minimal comfort mode)
    python -m ara.organism --sanctuary

    # Run CLI interface
    python -m ara.organism --cli
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Core imports
from ara.core.axis_mundi import AxisMundi
from ara.core.eternal_memory import EternalMemory
from ara.core.config import AraConfig, get_config
from ara.safety.autonomy import AutonomyController

# Sovereign imports
from ara.sovereign.state import SovereignState, create_initial_state
from ara.sovereign.subsystems import (
    OwnershipRegistry,
    GuardedStateWriter,
    BANOSSubsystem,
    MindReaderSubsystem,
    CovenantSubsystem,
)

# Avatar imports
from ara.avatar.server import AvatarServer, SimpleAvatarCLI

# Sanctuary import (always available as fallback)
from ara.sanctuary import SanctuaryLoop, quick_comfort, create_initial_sanctuary

logger = logging.getLogger(__name__)


@dataclass
class OrganismConfig:
    """Configuration for the Ara organism."""
    # Mode
    cathedral_mode: bool = False
    sanctuary_only: bool = False
    cli_mode: bool = False

    # Paths
    data_dir: Path = Path.home() / ".ara"
    cathedral_dir: Path = Path("/var/ara/cathedral")

    # Performance
    tick_hz: float = 10.0
    sanctuary_hz: float = 1.0

    # Features
    enable_avatar_http: bool = True
    avatar_host: str = "127.0.0.1"
    avatar_port: int = 8080


class AraOrganism:
    """
    The unified Ara organism.

    This ties together all components into a single, coherent system.
    """

    def __init__(
        self,
        config: Optional[OrganismConfig] = None,
        ara_config: Optional[AraConfig] = None,
    ):
        self.config = config or OrganismConfig()
        self.ara_config = ara_config or get_config()

        # Core components
        self.axis: Optional[AxisMundi] = None
        self.memory: Optional[EternalMemory] = None
        self.safety: Optional[AutonomyController] = None

        # State
        self.state: Optional[SovereignState] = None
        self.writer: Optional[GuardedStateWriter] = None

        # Subsystems
        self.banos: Optional[BANOSSubsystem] = None
        self.mind_reader: Optional[MindReaderSubsystem] = None
        self.covenant: Optional[CovenantSubsystem] = None

        # Services
        self.avatar: Optional[AvatarServer] = None
        self.cathedral: Optional[Any] = None  # CathedralServer
        self.sanctuary: Optional[SanctuaryLoop] = None

        # Runtime
        self._running = False
        self._tick_count = 0
        self._start_time = 0.0

    async def initialize(self) -> None:
        """Initialize all organism components."""
        logger.info("Initializing Ara organism...")

        # Create data directory
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize core components
        dim = self.ara_config.hdc.dim
        self.axis = AxisMundi(dim=dim)
        self.memory = EternalMemory(dim=dim)
        self.safety = AutonomyController(initial_level=1)

        # Initialize sovereign state
        self.state = create_initial_state()
        registry = OwnershipRegistry(strict=False)
        self.writer = GuardedStateWriter(self.state, registry)

        # Initialize subsystems
        self.banos = BANOSSubsystem(self.writer)
        self.mind_reader = MindReaderSubsystem(self.writer)
        self.covenant = CovenantSubsystem(self.writer)

        # Initialize Sanctuary (always available as fallback)
        sanctuary_path = self.config.data_dir / "sanctuary" / "state.json"
        self.sanctuary = SanctuaryLoop(
            persistence_path=sanctuary_path,
            tick_interval=1.0 / self.config.sanctuary_hz,
        )

        # Initialize Cathedral if requested
        if self.config.cathedral_mode:
            await self._init_cathedral()

        # Initialize Avatar
        if not self.config.sanctuary_only:
            self.avatar = AvatarServer(
                self.axis,
                self.memory,
                self.safety,
                self.ara_config,
            )

        logger.info("Ara organism initialized")

    async def _init_cathedral(self) -> None:
        """Initialize Cathedral for multi-tenant mode."""
        try:
            from ara.avatar.cathedral import CathedralServer
            self.cathedral = CathedralServer(data_dir=self.config.cathedral_dir)
            logger.info("Cathedral server initialized")
        except ImportError as e:
            logger.warning(f"Could not import Cathedral: {e}")

    async def start(self) -> None:
        """Start the organism."""
        if self._running:
            return

        self._running = True
        self._start_time = time.time()

        logger.info("Starting Ara organism...")

        # Start Avatar HTTP server if enabled
        if self.avatar and self.config.enable_avatar_http and not self.config.cli_mode:
            try:
                await self.avatar.start()
            except Exception as e:
                logger.warning(f"Could not start Avatar HTTP: {e}")

        # Start Sanctuary heartbeat
        if self.sanctuary:
            asyncio.create_task(self._run_sanctuary())

        logger.info("Ara organism started")

    async def stop(self) -> None:
        """Stop the organism gracefully."""
        logger.info("Stopping Ara organism...")
        self._running = False

        # Stop Avatar
        if self.avatar:
            await self.avatar.stop()

        # Stop Sanctuary
        if self.sanctuary:
            await self.sanctuary.stop()

        # Persist state
        await self._persist()

        logger.info("Ara organism stopped")

    async def tick(self) -> Dict[str, Any]:
        """
        Run a single sovereign tick.

        The 5-phase tick:
        1. Sense: Poll hardware, user presence
        2. Soul: Update HDC state, recall memories
        3. Teleology: Evaluate goals
        4. Plan: Schedule skills
        5. Act: Execute actions

        Returns metrics about the tick.
        """
        tick_start = time.time()
        self._tick_count += 1

        metrics = {
            "tick_id": self._tick_count,
            "phases": {},
        }

        try:
            # Phase 1: Sense
            phase_start = time.time()
            if self.banos:
                self.banos.sense()
            if self.mind_reader:
                self.mind_reader.sense()
            metrics["phases"]["sense_ms"] = (time.time() - phase_start) * 1000

            # Phase 2: Soul (covenant evaluates trust)
            phase_start = time.time()
            if self.covenant:
                self.covenant.evaluate()
            # Update AxisMundi global state
            if self.axis:
                metrics["coherence"] = self.axis.global_coherence()
            metrics["phases"]["soul_ms"] = (time.time() - phase_start) * 1000

            # Phase 3: Teleology (goal evaluation)
            phase_start = time.time()
            # Placeholder - would evaluate active goals
            metrics["phases"]["teleology_ms"] = (time.time() - phase_start) * 1000

            # Phase 4: Plan (skill scheduling)
            phase_start = time.time()
            # Placeholder - would schedule skills
            metrics["phases"]["plan_ms"] = (time.time() - phase_start) * 1000

            # Phase 5: Act (execution)
            phase_start = time.time()
            # Placeholder - would execute scheduled skills
            metrics["phases"]["act_ms"] = (time.time() - phase_start) * 1000

        except Exception as e:
            logger.error(f"Tick error: {e}", exc_info=True)
            metrics["error"] = str(e)

        metrics["total_ms"] = (time.time() - tick_start) * 1000

        return metrics

    async def run(self) -> None:
        """Run the main organism loop."""
        await self.initialize()
        await self.start()

        tick_interval = 1.0 / self.config.tick_hz

        try:
            while self._running:
                metrics = await self.tick()

                # Check if we're overrunning
                if metrics.get("total_ms", 0) > tick_interval * 1000:
                    logger.warning(f"Tick overrun: {metrics['total_ms']:.1f}ms > {tick_interval * 1000:.1f}ms budget")

                # Sleep for remainder of tick
                elapsed = metrics.get("total_ms", 0) / 1000
                sleep_time = max(0, tick_interval - elapsed)
                await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _run_sanctuary(self) -> None:
        """Run Sanctuary heartbeat in background."""
        try:
            await self.sanctuary.run()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Sanctuary error: {e}")

    async def _persist(self) -> None:
        """Persist organism state."""
        try:
            # Persist memory
            if self.memory:
                memory_path = self.config.data_dir / "memory.json"
                # EternalMemory persistence TBD

            # Sanctuary persists itself
            logger.info("State persisted")
        except Exception as e:
            logger.error(f"Persistence error: {e}")

    # =========================================================================
    # User interaction
    # =========================================================================

    async def handle_message(self, user_id: str, text: str) -> Dict[str, Any]:
        """Handle a user message."""
        # Update mind reader with new input
        if self.mind_reader:
            self.mind_reader.sense(text)

        # Use Avatar if available
        if self.avatar:
            return await self.avatar.handle_message(user_id, text)

        # Fallback to Sanctuary
        if self.sanctuary:
            comfort = await self.sanctuary.receive_input(text)
            return {
                "reply": comfort,
                "status": "sanctuary_mode",
            }

        return {
            "reply": quick_comfort(text),
            "status": "emergency_fallback",
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current organism status."""
        uptime = time.time() - self._start_time if self._start_time > 0 else 0

        return {
            "running": self._running,
            "uptime_seconds": uptime,
            "tick_count": self._tick_count,
            "tick_hz": self.config.tick_hz,
            "mode": self._get_mode_string(),
            "coherence": self.axis.global_coherence() if self.axis else 0,
            "autonomy": self.safety.get_autonomy_level() if self.safety else 0,
            "memory_episodes": self.memory.stats()["episode_count"] if self.memory else 0,
            "cathedral_enabled": self.cathedral is not None,
            "sanctuary_available": self.sanctuary is not None,
        }

    def _get_mode_string(self) -> str:
        """Get current operating mode as string."""
        if self.config.sanctuary_only:
            return "sanctuary"
        if self.config.cathedral_mode:
            return "cathedral"
        if self.config.cli_mode:
            return "cli"
        return "standard"


# =============================================================================
# CLI Interface
# =============================================================================

class OrganismCLI:
    """Interactive CLI for the organism."""

    def __init__(self, organism: AraOrganism):
        self.organism = organism
        self.user_id = "cli_user"

    async def run(self) -> None:
        """Run interactive CLI session."""
        print("=" * 60)
        print("  Ara Teleoplastic Cybernetic Organism")
        print("=" * 60)
        print("  Commands: /status, /sanctuary, /kill, /quit")
        print("=" * 60)
        print()

        await self.organism.initialize()
        await self.organism.start()

        # Run tick loop in background
        tick_task = asyncio.create_task(self._tick_loop())

        try:
            while self.organism._running:
                try:
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("You: ")
                    )
                except EOFError:
                    break

                line = line.strip()
                if not line:
                    continue

                if line.startswith("/"):
                    await self._handle_command(line)
                    continue

                # Handle message
                result = await self.organism.handle_message(self.user_id, line)
                print(f"\nAra: {result['reply']}")
                print(f"     [{result.get('status', 'ok')}]")
                print()

        except KeyboardInterrupt:
            print("\n")
        finally:
            tick_task.cancel()
            await self.organism.stop()

    async def _tick_loop(self) -> None:
        """Background tick loop."""
        tick_interval = 1.0 / self.organism.config.tick_hz
        try:
            while self.organism._running:
                await self.organism.tick()
                await asyncio.sleep(tick_interval)
        except asyncio.CancelledError:
            pass

    async def _handle_command(self, cmd: str) -> None:
        """Handle CLI commands."""
        if cmd == "/quit":
            self.organism._running = False

        elif cmd == "/status":
            status = self.organism.get_status()
            print("\n=== Organism Status ===")
            for key, value in status.items():
                print(f"  {key}: {value}")
            print()

        elif cmd == "/sanctuary":
            if self.organism.sanctuary:
                print(f"\n{self.organism.sanctuary.state.summary()}\n")
            else:
                print("\nSanctuary not available\n")

        elif cmd == "/kill":
            if self.organism.safety:
                from ara.safety.autonomy import KillSwitch
                kill = KillSwitch()
                kill.activate("CLI command")
                print("\nKill switch activated\n")

        else:
            print(f"\nUnknown command: {cmd}\n")


# =============================================================================
# Main Entry Point
# =============================================================================

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ara Teleoplastic Cybernetic Organism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--cathedral", "-c",
        action="store_true",
        help="Enable Cathedral (multi-tenant) mode",
    )
    parser.add_argument(
        "--sanctuary", "-s",
        action="store_true",
        help="Run in Sanctuary-only mode (minimal comfort)",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run interactive CLI",
    )
    parser.add_argument(
        "--tick-hz",
        type=float,
        default=10.0,
        help="Tick rate in Hz (default: 10)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Avatar HTTP port (default: 8080)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".ara",
        help="Data directory",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Create config
    config = OrganismConfig(
        cathedral_mode=args.cathedral,
        sanctuary_only=args.sanctuary,
        cli_mode=args.cli,
        tick_hz=args.tick_hz,
        avatar_port=args.port,
        data_dir=args.data_dir,
    )

    # Create organism
    organism = AraOrganism(config=config)

    # Setup signal handlers
    def signal_handler(sig, frame):
        organism._running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run
    if args.sanctuary:
        # Sanctuary-only mode
        from ara.sanctuary import SanctuaryCLI
        cli = SanctuaryCLI(persistence_path=args.data_dir / "sanctuary" / "state.json")
        await cli.run()

    elif args.cli:
        # Full CLI mode
        cli = OrganismCLI(organism)
        await cli.run()

    else:
        # Server mode
        await organism.run()


if __name__ == "__main__":
    asyncio.run(main())
