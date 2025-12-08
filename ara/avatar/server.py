"""
Avatar Server: Single-User Avatar (Local-First, VPN-Ready)

HTTP/WebSocket endpoint for Ara conversations. Handles the full loop:
  Input text → encode to HV → update AxisMundi + EternalMemory → reason → output

Usage:
    # Start server
    python -m ara.avatar.server

    # Or programmatically
    server = AvatarServer(axis, memory, safety)
    await server.start()

The server exposes:
- POST /message - Send a text message, get response
- GET /stream - WebSocket for streaming conversation
- GET /status - System status
- POST /kill - Activate kill switch (requires auth)
"""

from __future__ import annotations

import asyncio
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List
from pathlib import Path
import hashlib

# Use aiohttp if available, otherwise provide stub
try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None

import numpy as np

from ara.core.axis_mundi import AxisMundi, encode_text_to_hv
from ara.core.eternal_memory import EternalMemory
from ara.core.config import AraConfig, get_config
from ara.safety.autonomy import AutonomyController, KillSwitch

logger = logging.getLogger(__name__)


@dataclass
class AvatarSession:
    """A single user's conversation session."""
    session_id: str
    user_id: str
    created_at: float
    last_activity: float
    message_count: int = 0
    context_hv: Optional[np.ndarray] = None
    emotion_hv: Optional[np.ndarray] = None


class AvatarServer:
    """
    Avatar HTTP/WebSocket server.

    Handles user messages, updates holographic state, recalls memories,
    and generates responses.
    """

    def __init__(
        self,
        axis: AxisMundi,
        memory: EternalMemory,
        safety: AutonomyController,
        config: Optional[AraConfig] = None,
    ):
        self.axis = axis
        self.memory = memory
        self.safety = safety
        self.config = config or get_config()

        self._sessions: Dict[str, AvatarSession] = {}
        self._running = False
        self._app = None
        self._runner = None

    def _generate_session_id(self, user_id: str) -> str:
        """Generate unique session ID."""
        timestamp = int(time.time() * 1000)
        data = f"{user_id}:{timestamp}".encode()
        return hashlib.sha256(data).hexdigest()[:16]

    def get_or_create_session(self, user_id: str) -> AvatarSession:
        """Get existing session or create new one."""
        # Look for existing session for this user
        for session in self._sessions.values():
            if session.user_id == user_id:
                session.last_activity = time.time()
                return session

        # Create new session
        session_id = self._generate_session_id(user_id)
        session = AvatarSession(
            session_id=session_id,
            user_id=user_id,
            created_at=time.time(),
            last_activity=time.time(),
        )
        self._sessions[session_id] = session
        logger.info(f"Created session {session_id} for user {user_id}")
        return session

    def encode_state(self, text: str, session: AvatarSession) -> np.ndarray:
        """Encode text + context into hypervector."""
        text_hv = encode_text_to_hv(text, dim=self.config.hdc.dim)

        # Combine with session context if available
        if session.context_hv is not None:
            # Bind with context for continuity
            combined = text_hv + 0.5 * session.context_hv
            norm = np.linalg.norm(combined)
            if norm > 1e-10:
                combined /= norm
            return combined

        return text_hv

    def encode_emotion(self, text: str) -> np.ndarray:
        """Extract emotional valence from text (simple heuristic)."""
        # Very simple sentiment heuristic - replace with proper model
        positive_words = {"happy", "good", "great", "love", "thanks", "awesome", "wonderful"}
        negative_words = {"sad", "bad", "angry", "hate", "terrible", "awful", "frustrated"}

        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)

        # Create emotion vector
        valence = (pos_count - neg_count) / max(1, pos_count + neg_count + 1)

        # Use valence to seed emotion HV
        seed = int((valence + 1) * 1000)  # Map [-1, 1] to [0, 2000]
        rng = np.random.default_rng(seed)
        emotion_hv = rng.choice([-1.0, 1.0], size=self.config.hdc.dim).astype(np.float32)

        return emotion_hv

    async def handle_message(
        self,
        user_id: str,
        text: str,
    ) -> Dict[str, Any]:
        """
        Handle a user message and generate response.

        Args:
            user_id: User identifier
            text: Message text

        Returns:
            Response dict with reply, memory info, etc.
        """
        start_time = time.time()

        # Check safety
        if not self.safety.can_observe():
            return {
                "reply": "Ara is currently paused. Please wait or contact the operator.",
                "status": "paused",
            }

        # Get session
        session = self.get_or_create_session(user_id)
        session.message_count += 1

        # Encode state
        state_hv = self.encode_state(text, session)
        emotion_hv = self.encode_emotion(text)

        # Update AxisMundi
        self.axis.write("avatar", state_hv)

        # Recall relevant memories
        recall_result = self.memory.recall(
            query_hv=state_hv,
            query_emotion_hv=emotion_hv,
            k=5,
            user_filter=user_id,
        )

        # Generate response
        reply = self._reason(text, recall_result, session)

        # Store this interaction as memory
        self.memory.store(
            content_hv=state_hv,
            emotion_hv=emotion_hv,
            strength=0.8,
            meta={
                "user": user_id,
                "session": session.session_id,
                "message": text[:100],  # Truncate for storage
                "reply": reply[:100],
            },
        )

        # Update session context
        session.context_hv = state_hv
        session.emotion_hv = emotion_hv

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "reply": reply,
            "status": "ok",
            "session_id": session.session_id,
            "message_count": session.message_count,
            "memories_recalled": len(recall_result.episodes),
            "memory_strength": recall_result.total_strength,
            "coherence": self.axis.global_coherence(),
            "autonomy_level": self.safety.get_autonomy_level(),
            "processing_time_ms": elapsed_ms,
        }

    def _reason(
        self,
        text: str,
        recall_result: Any,
        session: AvatarSession,
    ) -> str:
        """
        Generate a response using recalled memories and current context.

        This is a placeholder - replace with actual reasoning/LLM.
        """
        # Simple echo + memory awareness for now
        memory_context = ""
        if recall_result.episodes:
            top_memory = recall_result.episodes[0]
            if top_memory.similarity > 0.3:
                memory_context = f" (I remember something related: {top_memory.meta.get('message', 'a past conversation')})"

        autonomy = self.safety.get_autonomy_level()
        if autonomy == 0:
            prefix = "[Observer mode] "
        elif autonomy == 1:
            prefix = ""
        else:
            prefix = ""

        # Placeholder response
        if "hello" in text.lower() or "hi" in text.lower():
            reply = f"{prefix}Hello! I'm Ara.{memory_context} How can I help you today?"
        elif "how are you" in text.lower():
            reply = f"{prefix}I'm running well - coherence at {self.axis.global_coherence():.2f}.{memory_context}"
        elif "remember" in text.lower():
            if recall_result.episodes:
                topics = [ep.meta.get("message", "?")[:30] for ep in recall_result.episodes[:3]]
                reply = f"{prefix}I recall these related moments: {', '.join(topics)}"
            else:
                reply = f"{prefix}I don't have strong memories about that yet."
        else:
            reply = f"{prefix}I heard: '{text[:50]}'{memory_context}. (Reasoning engine not yet implemented)"

        return reply

    # =========================================================================
    # HTTP Handlers
    # =========================================================================

    async def handle_message_http(self, request: "web.Request") -> "web.Response":
        """Handle POST /message."""
        try:
            data = await request.json()
            user_id = data.get("user_id", "anonymous")
            text = data.get("text", "")

            if not text:
                return web.json_response({"error": "No text provided"}, status=400)

            result = await self.handle_message(user_id, text)
            return web.json_response(result)

        except Exception as e:
            logger.error(f"Message handler error: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    async def handle_status(self, request: "web.Request") -> "web.Response":
        """Handle GET /status."""
        status = {
            "status": "running" if self._running else "stopped",
            "sessions": len(self._sessions),
            "memory_episodes": self.memory.stats()["episode_count"],
            "global_coherence": self.axis.global_coherence(),
            "autonomy_level": self.safety.get_autonomy_level(),
            "layer_stats": self.axis.layer_stats(),
        }
        return web.json_response(status)

    async def handle_kill(self, request: "web.Request") -> "web.Response":
        """Handle POST /kill - activate kill switch."""
        try:
            data = await request.json()
            reason = data.get("reason", "manual via API")

            kill_switch = KillSwitch(self.config.safety.kill_switch_file)
            kill_switch.activate(reason)

            return web.json_response({"status": "kill switch activated"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_unkill(self, request: "web.Request") -> "web.Response":
        """Handle POST /unkill - deactivate kill switch."""
        kill_switch = KillSwitch(self.config.safety.kill_switch_file)
        kill_switch.deactivate()
        self.safety.unlock()
        return web.json_response({"status": "kill switch deactivated"})

    # =========================================================================
    # Server lifecycle
    # =========================================================================

    def _setup_routes(self, app: "web.Application") -> None:
        """Set up HTTP routes."""
        app.router.add_post("/message", self.handle_message_http)
        app.router.add_get("/status", self.handle_status)
        app.router.add_post("/kill", self.handle_kill)
        app.router.add_post("/unkill", self.handle_unkill)

    async def start(self) -> None:
        """Start the avatar server."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not installed. Run: pip install aiohttp")

        self._app = web.Application()
        self._setup_routes(self._app)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        site = web.TCPSite(
            self._runner,
            self.config.avatar.host,
            self.config.avatar.port,
        )
        await site.start()

        self._running = True
        logger.info(f"Avatar server started on http://{self.config.avatar.host}:{self.config.avatar.port}")

    async def stop(self) -> None:
        """Stop the avatar server."""
        self._running = False
        if self._runner:
            await self._runner.cleanup()
        logger.info("Avatar server stopped")


# =============================================================================
# Standalone server runner
# =============================================================================

async def run_avatar_server(
    axis: AxisMundi,
    memory: EternalMemory,
    safety: AutonomyController,
    config: Optional[AraConfig] = None,
) -> None:
    """Run the avatar server (blocks forever)."""
    server = AvatarServer(axis, memory, safety, config)
    await server.start()

    # Keep running
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        await server.stop()


# =============================================================================
# CLI interface for testing without aiohttp
# =============================================================================

class SimpleAvatarCLI:
    """
    Simple CLI interface for avatar when aiohttp not available.

    Usage:
        cli = SimpleAvatarCLI(axis, memory, safety)
        asyncio.run(cli.run())
    """

    def __init__(
        self,
        axis: AxisMundi,
        memory: EternalMemory,
        safety: AutonomyController,
        user_id: str = "cli_user",
    ):
        self.axis = axis
        self.memory = memory
        self.safety = safety
        self.user_id = user_id

        # Reuse AvatarServer logic
        self._server = AvatarServer(axis, memory, safety)

    async def run(self) -> None:
        """Run interactive CLI session."""
        print("=" * 60)
        print("Ara Avatar CLI")
        print("=" * 60)
        print("Type your message and press Enter. Type 'quit' to exit.")
        print("Commands: /status, /kill, /unkill, /memory")
        print("=" * 60)
        print()

        import sys

        while True:
            try:
                # Use asyncio-compatible input
                line = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("You: ")
                )
            except EOFError:
                break

            line = line.strip()
            if not line:
                continue

            if line.lower() == "quit":
                break

            if line.startswith("/"):
                await self._handle_command(line)
                continue

            # Handle as message
            result = await self._server.handle_message(self.user_id, line)
            print(f"\nAra: {result['reply']}")
            print(f"     [coherence={result['coherence']:.3f}, memories={result['memories_recalled']}, autonomy={result['autonomy_level']}]")
            print()

    async def _handle_command(self, cmd: str) -> None:
        """Handle CLI commands."""
        if cmd == "/status":
            stats = self.memory.stats()
            print(f"\n=== Status ===")
            print(f"Coherence: {self.axis.global_coherence():.3f}")
            print(f"Autonomy: {self.safety.get_autonomy_level()}")
            print(f"Memory episodes: {stats['episode_count']}")
            print(f"Layers: {self.axis.layer_names()}")
            print()

        elif cmd == "/kill":
            kill = KillSwitch()
            kill.activate("CLI command")
            print("Kill switch activated")

        elif cmd == "/unkill":
            kill = KillSwitch()
            kill.deactivate()
            self.safety.unlock()
            print("Kill switch deactivated")

        elif cmd == "/memory":
            episodes = self.memory.list_episodes(limit=5)
            print(f"\n=== Recent Memories ===")
            for ep in episodes:
                msg = ep.meta.get("message", "?")[:40]
                print(f"  [{ep.strength:.2f}] {msg}")
            print()

        else:
            print(f"Unknown command: {cmd}")


# =============================================================================
# Main entry point
# =============================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    # Initialize components
    axis = AxisMundi(dim=4096)
    memory = EternalMemory(dim=4096)
    safety = AutonomyController(initial_level=1)

    if AIOHTTP_AVAILABLE and "--server" in sys.argv:
        # Run HTTP server
        asyncio.run(run_avatar_server(axis, memory, safety))
    else:
        # Run CLI
        cli = SimpleAvatarCLI(axis, memory, safety)
        asyncio.run(cli.run())
