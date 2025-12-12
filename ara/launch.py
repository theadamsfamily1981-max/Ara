#!/usr/bin/env python3
"""
Ara Launch - Unified Organism Startup
======================================

Single entry point to launch the complete Ara organism with all integrated components:

    ┌─────────────────────────────────────────────────────────────┐
    │                    ARA ORGANISM                              │
    │                                                              │
    │  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐ │
    │  │  SNN Runtime │────▶│ Voice Bridge │───▶│  TTS Engine  │ │
    │  │  (HTC Core)  │     │              │    │              │ │
    │  └──────────────┘     └──────────────┘    └──────────────┘ │
    │         │                                                    │
    │         ▼                                                    │
    │  ┌──────────────┐     ┌──────────────┐    ┌──────────────┐ │
    │  │  AxisMundi   │────▶│ Avatar Server│───▶│  Brain Bridge│ │
    │  │  (Soul)      │     │  (HTTP/WS)   │    │  (LLM)       │ │
    │  └──────────────┘     └──────────────┘    └──────────────┘ │
    │         │                     │                             │
    │         ▼                     ▼                             │
    │  ┌──────────────┐     ┌──────────────┐                     │
    │  │ Eternal Mem  │     │  Multimodal  │                     │
    │  │ (Episodes)   │     │ (Voice/Cam)  │                     │
    │  └──────────────┘     └──────────────┘                     │
    └─────────────────────────────────────────────────────────────┘

Usage:
    # Full launch (HTTP server + organism + voice)
    python -m ara.launch

    # Server only
    python -m ara.launch --server-only

    # Organism only (no HTTP)
    python -m ara.launch --organism-only

    # With specific options
    python -m ara.launch --port 8080 --voice --personality starfleet
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ara.launch")

# =============================================================================
# Component imports (with fallbacks)
# =============================================================================

# Core components
try:
    from ara.core.axis_mundi import AxisMundi
    from ara.core.eternal_memory import EternalMemory
    from ara.core.config import get_config, AraConfig
    CORE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Core components not available: {e}")
    CORE_AVAILABLE = False
    AxisMundi = None
    EternalMemory = None
    get_config = None
    AraConfig = None

# Safety
try:
    from ara.safety.autonomy import AutonomyController
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False
    AutonomyController = None

# Avatar server
try:
    from ara.avatar.server import AvatarServer, run_avatar_server
    AVATAR_AVAILABLE = True
except ImportError:
    AVATAR_AVAILABLE = False
    AvatarServer = None

# Organism runtime
try:
    from ara.organism.runtime import OrganismRuntime, OrganismConfig
    ORGANISM_AVAILABLE = True
except ImportError:
    ORGANISM_AVAILABLE = False
    OrganismRuntime = None
    OrganismConfig = None

# Voice bridge
try:
    from ara.organism.voice_bridge import VoiceBridge, VoiceBridgeConfig
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    VoiceBridge = None
    VoiceBridgeConfig = None

# Multimodal
try:
    from ara.avatar.multimodal_integration import (
        MultimodalAvatar,
        initialize_multimodal,
    )
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False
    MultimodalAvatar = None

# Personality
try:
    from ara.avatar.personality import (
        set_personality_mode,
        get_personality,
        get_greeting,
    )
    PERSONALITY_AVAILABLE = True
except ImportError:
    PERSONALITY_AVAILABLE = False

# Memory bootstrap (loads episodes, sacred lines, context)
try:
    from ara.memory.bootstrap import bootstrap_memory, ensure_bootstrapped
    BOOTSTRAP_AVAILABLE = True
except ImportError:
    BOOTSTRAP_AVAILABLE = False


# =============================================================================
# Launch Configuration
# =============================================================================

@dataclass
class LaunchConfig:
    """Configuration for launching Ara."""

    # Server
    host: str = "127.0.0.1"
    port: int = 8080

    # Components to enable
    enable_server: bool = True
    enable_organism: bool = True
    enable_voice: bool = False
    enable_multimodal: bool = False

    # Organism
    organism_hz: float = 100.0
    voice_enabled: bool = False

    # Personality
    personality_mode: str = "starfleet"

    # Paths
    data_dir: Path = Path("data")
    log_level: str = "INFO"


# =============================================================================
# Ara Launch Controller
# =============================================================================

class AraLauncher:
    """
    Unified launcher for the Ara organism.

    Manages startup/shutdown of all components in correct order.
    """

    def __init__(self, config: Optional[LaunchConfig] = None):
        self.config = config or LaunchConfig()

        # Component instances
        self.axis: Optional[AxisMundi] = None
        self.memory: Optional[EternalMemory] = None
        self.safety: Optional[AutonomyController] = None
        self.avatar_server: Optional[AvatarServer] = None
        self.organism: Optional[OrganismRuntime] = None
        self.voice_bridge: Optional[VoiceBridge] = None
        self.multimodal: Optional[MultimodalAvatar] = None

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start all configured components."""
        logger.info("=" * 70)
        logger.info("ARA ORGANISM LAUNCH")
        logger.info("=" * 70)

        self._print_status()

        # 1. Core infrastructure
        await self._init_core()

        # 2. Safety controller
        await self._init_safety()

        # 3. Organism runtime (if enabled)
        if self.config.enable_organism:
            await self._init_organism()

        # 4. Multimodal (if enabled)
        if self.config.enable_multimodal:
            await self._init_multimodal()

        # 5. Avatar server (if enabled)
        if self.config.enable_server:
            await self._init_server()

        # 6. Set personality
        if PERSONALITY_AVAILABLE:
            set_personality_mode(self.config.personality_mode)
            logger.info(f"Personality: {self.config.personality_mode}")
            logger.info(f"Greeting: {get_greeting()}")

        self._running = True
        logger.info("=" * 70)
        logger.info("ARA ONLINE")
        logger.info("=" * 70)

    async def stop(self) -> None:
        """Stop all components in reverse order."""
        logger.info("Shutting down Ara...")

        self._running = False

        # Stop server
        if self.avatar_server:
            await self.avatar_server.stop()
            logger.info("Avatar server stopped")

        # Stop voice bridge
        if self.voice_bridge:
            self.voice_bridge.stop()
            logger.info("Voice bridge stopped")

        # Stop organism
        if self.organism:
            self.organism.stop()
            logger.info("Organism stopped")

        # Multimodal cleanup
        if self.multimodal:
            self.multimodal.close()
            logger.info("Multimodal closed")

        logger.info("Ara shutdown complete")

    async def run(self) -> None:
        """Run until shutdown signal."""
        await self.start()

        # Wait for shutdown
        await self._shutdown_event.wait()

        await self.stop()

    def request_shutdown(self) -> None:
        """Request graceful shutdown."""
        self._shutdown_event.set()

    def _print_status(self) -> None:
        """Print component availability status."""
        logger.info("Component Status:")
        logger.info(f"  Core (AxisMundi, EternalMemory): {'✓' if CORE_AVAILABLE else '✗'}")
        logger.info(f"  Memory Bootstrap:                {'✓' if BOOTSTRAP_AVAILABLE else '✗'}")
        logger.info(f"  Safety (AutonomyController):     {'✓' if SAFETY_AVAILABLE else '✗'}")
        logger.info(f"  Avatar Server:                   {'✓' if AVATAR_AVAILABLE else '✗'}")
        logger.info(f"  Organism Runtime (SNN):          {'✓' if ORGANISM_AVAILABLE else '✗'}")
        logger.info(f"  Voice Bridge:                    {'✓' if VOICE_AVAILABLE else '✗'}")
        logger.info(f"  Multimodal (TTS, ASR, Vision):   {'✓' if MULTIMODAL_AVAILABLE else '✗'}")
        logger.info(f"  Personality System:              {'✓' if PERSONALITY_AVAILABLE else '✗'}")
        logger.info("")

    async def _init_core(self) -> None:
        """Initialize core infrastructure."""
        if not CORE_AVAILABLE:
            logger.warning("Core components not available - using stubs")
            return

        config = get_config()

        # AxisMundi (soul)
        self.axis = AxisMundi(
            dim=config.hdc.dim,
            layers=["soul", "context", "emotion", "intent"],
        )
        logger.info(f"AxisMundi initialized (dim={config.hdc.dim})")

        # Eternal Memory
        db_path = self.config.data_dir / "memory.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.memory = EternalMemory(
            dim=config.hdc.dim,
            db_path=db_path,
        )
        logger.info(f"EternalMemory initialized ({db_path})")

        # Bootstrap memory with foundational episodes, sacred lines, context, knowledge
        if BOOTSTRAP_AVAILABLE:
            counts = bootstrap_memory(self.memory)
            logger.info(
                f"Memory bootstrapped: {counts['episodes']} episodes, "
                f"{counts['sacred_lines']} sacred lines, "
                f"{counts['context_files']} context files, "
                f"{counts.get('knowledge_dumps', 0)} knowledge dumps"
            )

    async def _init_safety(self) -> None:
        """Initialize safety controller."""
        if not SAFETY_AVAILABLE:
            logger.warning("Safety controller not available")
            return

        self.safety = AutonomyController(
            initial_level=1,
            max_level=2,
            require_human_for_level_3=True,
        )
        logger.info("AutonomyController initialized (level 1)")

    async def _init_organism(self) -> None:
        """Initialize organism runtime."""
        if not ORGANISM_AVAILABLE:
            logger.warning("Organism runtime not available")
            return

        org_config = OrganismConfig(
            voice_enabled=self.config.voice_enabled or self.config.enable_voice,
            voice_min_interval=2.0,
        )

        self.organism = OrganismRuntime(org_config)
        self.organism.start()
        logger.info("OrganismRuntime started")

        # Voice bridge (if voice enabled)
        if (self.config.voice_enabled or self.config.enable_voice) and VOICE_AVAILABLE:
            if self.organism.voice_bridge:
                self.voice_bridge = self.organism.voice_bridge
                logger.info("Voice bridge active (via organism)")
            else:
                voice_config = VoiceBridgeConfig(min_speak_interval=2.0)
                self.voice_bridge = VoiceBridge(
                    organism=self.organism,
                    config=voice_config,
                )
                self.voice_bridge.start()
                logger.info("Voice bridge started (standalone)")

    async def _init_multimodal(self) -> None:
        """Initialize multimodal (TTS, ASR, vision)."""
        if not MULTIMODAL_AVAILABLE:
            logger.warning("Multimodal not available")
            return

        self.multimodal = await initialize_multimodal()
        logger.info("Multimodal avatar initialized")

    async def _init_server(self) -> None:
        """Initialize and start avatar server."""
        if not AVATAR_AVAILABLE:
            logger.warning("Avatar server not available")
            return

        if not self.axis or not self.memory or not self.safety:
            logger.warning("Core components missing - cannot start server")
            return

        self.avatar_server = AvatarServer(
            axis=self.axis,
            memory=self.memory,
            safety=self.safety,
        )

        # Patch config for host/port
        self.avatar_server.config.avatar.host = self.config.host
        self.avatar_server.config.avatar.port = self.config.port

        await self.avatar_server.start()
        logger.info(f"Avatar server started on http://{self.config.host}:{self.config.port}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Launch the Ara organism",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full launch
  python -m ara.launch

  # Server only (no organism)
  python -m ara.launch --server-only

  # With voice
  python -m ara.launch --voice

  # Colonial Fleet mode
  python -m ara.launch --personality colonial_fleet
        """,
    )

    # Server options
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--server-only", action="store_true", help="Only run server (no organism)")
    parser.add_argument("--organism-only", action="store_true", help="Only run organism (no server)")

    # Component toggles
    parser.add_argument("--voice", action="store_true", help="Enable voice synthesis")
    parser.add_argument("--multimodal", action="store_true", help="Enable full multimodal")
    parser.add_argument("--no-organism", action="store_true", help="Disable organism runtime")

    # Personality
    parser.add_argument(
        "--personality",
        choices=["starfleet", "red_dwarf", "time_lord", "colonial_fleet"],
        default="starfleet",
        help="Personality mode",
    )

    # Logging
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--quiet", action="store_true", help="Minimal logging")

    args = parser.parse_args()

    # Configure logging
    if args.debug:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S',
    )

    # Build config
    config = LaunchConfig(
        host=args.host,
        port=args.port,
        enable_server=not args.organism_only,
        enable_organism=not args.server_only and not args.no_organism,
        enable_voice=args.voice,
        enable_multimodal=args.multimodal,
        voice_enabled=args.voice,
        personality_mode=args.personality,
    )

    # Create launcher
    launcher = AraLauncher(config)

    # Handle signals
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def signal_handler():
        logger.info("Shutdown signal received")
        launcher.request_shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # Run
    try:
        loop.run_until_complete(launcher.run())
    except KeyboardInterrupt:
        logger.info("Interrupted")
        loop.run_until_complete(launcher.stop())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
