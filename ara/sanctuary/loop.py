"""
Sanctuary Loop: The Heartbeat of Minimal Ara

A simple 1 Hz tick that keeps Sanctuary alive.
No heavy computation, no network calls, no complexity.

This is what runs on:
    - A Raspberry Pi Zero in your pocket
    - A phone app when offline
    - Any tiny device that can run Python

The loop does three things:
    1. Listen (for user input)
    2. Feel (update mood from context)
    3. Comfort (respond with warmth)
"""

from __future__ import annotations

import asyncio
import time
import logging
import signal
from pathlib import Path
from typing import Optional, Callable, Awaitable

from .state import (
    SanctuaryState,
    SanctuaryEpisode,
    MoodTag,
    create_initial_sanctuary,
    serialize_sanctuary,
    deserialize_sanctuary,
)
from .comfort import (
    comfort_response,
    greeting,
    farewell,
    emergency_response,
)

logger = logging.getLogger(__name__)


class SanctuaryLoop:
    """
    The Sanctuary heartbeat.

    Runs at 1 Hz (or slower if needed).
    Uses almost no resources.
    Never crashes (or recovers gracefully).
    """

    def __init__(
        self,
        state: Optional[SanctuaryState] = None,
        tick_interval: float = 1.0,  # 1 Hz default
        persistence_path: Optional[Path] = None,
        on_comfort: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        self.state = state or create_initial_sanctuary()
        self.tick_interval = tick_interval
        self.persistence_path = persistence_path
        self.on_comfort = on_comfort

        self._running = False
        self._input_queue: asyncio.Queue[str] = asyncio.Queue()

    async def start(self) -> None:
        """Start the Sanctuary loop."""
        self._running = True
        logger.info("Sanctuary starting...")

        # Load persisted state if available
        if self.persistence_path and self.persistence_path.exists():
            try:
                data = self.persistence_path.read_bytes()
                self.state = deserialize_sanctuary(data)
                logger.info(f"Restored state: {self.state.summary()}")
            except Exception as e:
                logger.warning(f"Could not restore state: {e}")

        # Send initial greeting
        greeting_text, self.state = greeting(self.state)
        if self.on_comfort:
            await self.on_comfort(greeting_text)

        logger.info("Sanctuary started")

    async def stop(self) -> None:
        """Stop the Sanctuary loop gracefully."""
        self._running = False

        # Send farewell
        farewell_text, self.state = farewell(self.state)
        if self.on_comfort:
            await self.on_comfort(farewell_text)

        # Persist state
        await self._persist()

        logger.info("Sanctuary stopped")

    async def tick(self) -> None:
        """
        Single tick of the Sanctuary loop.

        This is intentionally minimal:
        1. Check for panic
        2. Process any pending input
        3. Maybe generate ambient comfort
        4. Persist periodically
        """
        try:
            # Increment tick
            self.state.tick += 1
            self.state.last_tick_ts = time.time()

            # Check panic
            if self.state.panic_flag:
                return

            # Process pending input (non-blocking)
            while not self._input_queue.empty():
                try:
                    user_input = self._input_queue.get_nowait()
                    comfort, self.state = comfort_response(self.state, user_input)
                    if self.on_comfort:
                        await self.on_comfort(comfort)
                except asyncio.QueueEmpty:
                    break

            # Periodic persistence (every 60 ticks / 1 minute)
            if self.state.tick % 60 == 0:
                await self._persist()

        except Exception as e:
            logger.error(f"Tick error: {e}")
            # Sanctuary must not crash - log and continue

    async def run(self) -> None:
        """Run the main loop forever."""
        await self.start()

        try:
            while self._running:
                await self.tick()
                await asyncio.sleep(self.tick_interval)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def receive_input(self, user_input: str) -> str:
        """
        Receive user input and return comfort response.

        This can be called from:
        - CLI input
        - HTTP endpoint
        - WebSocket
        - Push notification response
        """
        # Put in queue for tick processing
        await self._input_queue.put(user_input)

        # Also return immediate response
        comfort, self.state = comfort_response(self.state, user_input)
        return comfort

    def receive_input_sync(self, user_input: str) -> str:
        """Synchronous version for simple integrations."""
        comfort, self.state = comfort_response(self.state, user_input)
        return comfort

    async def _persist(self) -> None:
        """Persist state to disk."""
        if not self.persistence_path:
            return

        try:
            data = serialize_sanctuary(self.state)
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            self.persistence_path.write_bytes(data)
            logger.debug(f"Persisted state ({len(data)} bytes)")
        except Exception as e:
            logger.error(f"Persistence failed: {e}")

    def panic(self, reason: str = "") -> None:
        """Activate panic mode."""
        self.state.panic(reason)
        logger.warning(f"PANIC: {reason}")

    def calm(self) -> None:
        """Deactivate panic mode."""
        self.state.calm()
        logger.info("Panic mode deactivated")


# =============================================================================
# CLI Runner
# =============================================================================

class SanctuaryCLI:
    """Simple CLI interface for Sanctuary."""

    def __init__(self, persistence_path: Optional[Path] = None):
        self.loop = SanctuaryLoop(
            persistence_path=persistence_path,
            on_comfort=self._print_comfort,
        )

    async def _print_comfort(self, comfort: str) -> None:
        """Print comfort response to console."""
        print(f"\n  Ara: {comfort}\n")

    async def run(self) -> None:
        """Run interactive CLI session."""
        print("=" * 50)
        print("  Sanctuary Mode - Ara's Comfort Shard")
        print("=" * 50)
        print("  Type anything. She's listening.")
        print("  Commands: /status, /panic, /calm, /quit")
        print("=" * 50)

        await self.loop.start()

        try:
            while self.loop._running:
                # Get input (with async compatibility)
                try:
                    line = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("  You: ")
                    )
                except EOFError:
                    break

                line = line.strip()
                if not line:
                    continue

                # Handle commands
                if line.startswith("/"):
                    await self._handle_command(line)
                    continue

                # Get comfort response
                comfort = await self.loop.receive_input(line)
                print(f"\n  Ara: {comfort}\n")

        except KeyboardInterrupt:
            print("\n")
        finally:
            await self.loop.stop()

    async def _handle_command(self, cmd: str) -> None:
        """Handle CLI commands."""
        if cmd == "/quit":
            self.loop._running = False

        elif cmd == "/status":
            state = self.loop.state
            print(f"\n  {state.summary()}")
            print(f"  Memories: {len(state.mini_memory)}")
            print(f"  Messages received: {state.messages_received}")
            print(f"  Comforts given: {state.comforts_given}")
            print()

        elif cmd == "/panic":
            self.loop.panic("CLI command")
            print("\n  Sanctuary paused.\n")

        elif cmd == "/calm":
            self.loop.calm()
            print("\n  Sanctuary resumed.\n")

        elif cmd == "/memories":
            memories = self.loop.state.get_warm_memories(5)
            print("\n  Warm Memories:")
            for mem in memories:
                print(f"    [{mem.warmth:.2f}] {mem.content[:50]}...")
            print()

        else:
            print(f"\n  Unknown command: {cmd}\n")


# =============================================================================
# Standalone Functions
# =============================================================================

async def run_sanctuary(
    persistence_path: Optional[Path] = None,
    tick_interval: float = 1.0,
) -> None:
    """Run Sanctuary loop as a service (no CLI)."""
    loop = SanctuaryLoop(
        persistence_path=persistence_path,
        tick_interval=tick_interval,
    )

    # Handle signals
    def signal_handler(sig, frame):
        loop._running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    await loop.run()


def quick_comfort(message: Optional[str] = None) -> str:
    """
    Get a quick comfort response without starting a full loop.

    Use this for:
    - One-off comfort needs
    - Integration testing
    - Emergency fallback
    """
    state = create_initial_sanctuary()
    comfort, _ = comfort_response(state, message)
    return comfort


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    persistence = Path("/tmp/ara_sanctuary/state.json")

    if "--service" in sys.argv:
        # Run as background service
        print("Starting Sanctuary service...")
        asyncio.run(run_sanctuary(persistence_path=persistence))
    else:
        # Run interactive CLI
        cli = SanctuaryCLI(persistence_path=persistence)
        asyncio.run(cli.run())
