"""
Avatar Server: Single-User Avatar (Integrated)

HTTP/WebSocket endpoint for Ara conversations. Handles the full loop:
  Input text → encode to HV → update AxisMundi + EternalMemory → reason → output

FULLY INTEGRATED with:
  - BrainBridge (LLM Reasoning)
  - TreasuryGateway (Billing & Limits)
  - AxisMundi (Holographic State)
  - EternalMemory (Long-term Recall)

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

# Type hint only
if False:  # TYPE_CHECKING
    from ara.avatar.multimodal_integration import MultimodalAvatar

# INTEGRATION IMPORTS
from ara.cognition.brain_bridge import get_brain_bridge
from ara.enterprise.billing import get_treasury

# MULTIMODAL INTEGRATION
try:
    from ara.avatar.multimodal_integration import (
        get_multimodal_avatar,
        initialize_multimodal,
        MultimodalAvatar,
    )
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False

# PERSONALITY INTEGRATION
try:
    from ara.avatar.personality import (
        get_personality,
        get_greeting,
        get_farewell,
        enhance_response,
        get_current_mode,
        set_personality_mode,
    )
    PERSONALITY_AVAILABLE = True
except ImportError:
    PERSONALITY_AVAILABLE = False

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

        # INTEGRATION: Connect Organs
        self.brain = get_brain_bridge()
        self.treasury = get_treasury()

        # MULTIMODAL: Voice & Vision
        self.multimodal: Optional[MultimodalAvatar] = None
        if MULTIMODAL_AVAILABLE:
            self.multimodal = get_multimodal_avatar()

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

        FULLY INTEGRATED with:
          - Safety checks
          - Billing/Treasury limits
          - BrainBridge (LLM)
          - Memory storage

        Args:
            user_id: User identifier
            text: Message text

        Returns:
            Response dict with reply, memory info, etc.
        """
        start_time = time.time()

        # 1. SAFETY CHECK
        if not self.safety.can_observe():
            return {
                "reply": "Ara is currently paused. Please wait or contact the operator.",
                "status": "paused",
            }

        # Get session
        session = self.get_or_create_session(user_id)

        # 2. BILLING CHECK (INTEGRATED)
        if not self.treasury.check_limit(user_id, "messages_per_hour", session.message_count):
            tier = self.treasury.get_tier(user_id)
            return {
                "reply": "You have reached your message limit for this hour. Please upgrade your subscription to continue.",
                "status": "throttled",
                "tier": tier.value,
                "session_id": session.session_id,
            }

        session.message_count += 1

        # 3. ENCODE & MEMORY
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

        # 4. COGNITION (INTEGRATED BRAIN BRIDGE)
        tier = self.treasury.get_tier(user_id)
        memory_snippets = [ep.meta.get("message", "") for ep in recall_result.episodes]

        context_hv = {
            "resonance": self.axis.global_coherence(),
            "memory_snippets": memory_snippets
        }

        # Real reasoning call via BrainBridge
        reply = await self.brain.reason(
            user_input=text,
            context_hv=context_hv,
            user_profile={"tier": tier.value}
        )

        # PERSONALITY ENHANCEMENT
        if PERSONALITY_AVAILABLE:
            reply = enhance_response(reply)

        # 5. STORAGE & UPDATE
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
            "tier": tier.value,
            "message_count": session.message_count,
            "memories_recalled": len(recall_result.episodes),
            "memory_strength": recall_result.total_strength,
            "coherence": self.axis.global_coherence(),
            "autonomy_level": self.safety.get_autonomy_level(),
            "processing_time_ms": elapsed_ms,
        }

    def _reason_fallback(
        self,
        text: str,
        recall_result: Any,
        session: AvatarSession,
    ) -> str:
        """
        DEPRECATED: Fallback response when BrainBridge is unavailable.

        The main path now uses BrainBridge.reason() for real LLM cognition.
        This method is kept as an emergency fallback only.
        """
        # Simple echo + memory awareness for emergency fallback
        memory_context = ""
        if recall_result.episodes:
            top_memory = recall_result.episodes[0]
            if top_memory.similarity > 0.3:
                memory_context = f" (I remember something related: {top_memory.meta.get('message', 'a past conversation')})"

        autonomy = self.safety.get_autonomy_level()
        prefix = "[Fallback mode] " if autonomy == 0 else ""

        # Fallback responses
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
            reply = f"{prefix}I heard you say: '{text[:50]}'{memory_context}. My cognitive core is in fallback mode."

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
            # INTEGRATION: Brain and Treasury status
            "brain_online": self.brain.is_online,
            "treasury_enabled": self.treasury.enabled,
            # PERSONALITY: Current mode
            "personality_mode": get_current_mode() if PERSONALITY_AVAILABLE else "default",
            # MULTIMODAL: Availability
            "multimodal_available": MULTIMODAL_AVAILABLE,
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
    # Multimodal Routes
    # =========================================================================

    async def handle_voice_input(self, request: "web.Request") -> "web.Response":
        """Handle POST /voice/listen - capture voice and return text."""
        if not self.multimodal:
            return web.json_response(
                {"error": "Multimodal not available"},
                status=503
            )

        try:
            data = await request.json()
            timeout = data.get("timeout", 7.0)
            user_id = data.get("user_id", "anonymous")

            # Listen for speech
            text = self.multimodal.listen(timeout=timeout)

            if text:
                return web.json_response({
                    "status": "ok",
                    "text": text,
                    "user_id": user_id,
                })
            else:
                return web.json_response({
                    "status": "no_speech",
                    "text": None,
                })

        except Exception as e:
            logger.error(f"Voice input error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_voice_output(self, request: "web.Request") -> "web.Response":
        """Handle POST /voice/speak - speak text with optional emotion."""
        if not self.multimodal:
            return web.json_response(
                {"error": "Multimodal not available"},
                status=503
            )

        try:
            data = await request.json()
            text = data.get("text", "")
            emotion = data.get("emotion", "neutral")
            valence = data.get("valence", 0.0)
            arousal = data.get("arousal", 0.0)
            dominance = data.get("dominance", 0.5)

            if not text:
                return web.json_response({"error": "No text provided"}, status=400)

            # Speak with emotion
            self.multimodal.speak(
                text,
                emotion=emotion,
                valence=valence,
                arousal=arousal,
                dominance=dominance
            )

            return web.json_response({"status": "ok", "spoken": text})

        except Exception as e:
            logger.error(f"Voice output error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_vision(self, request: "web.Request") -> "web.Response":
        """Handle POST /vision - query what avatar sees."""
        if not self.multimodal:
            return web.json_response(
                {"error": "Multimodal not available"},
                status=503
            )

        try:
            data = await request.json()
            prompt = data.get("prompt", "What do you see?")

            description = self.multimodal.see(prompt)

            if description:
                return web.json_response({
                    "status": "ok",
                    "description": description,
                    "prompt": prompt,
                })
            else:
                return web.json_response({
                    "status": "no_frame",
                    "description": None,
                })

        except Exception as e:
            logger.error(f"Vision error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_vision_start(self, request: "web.Request") -> "web.Response":
        """Handle POST /vision/start - start webcam."""
        if not self.multimodal:
            return web.json_response({"error": "Multimodal not available"}, status=503)

        success = self.multimodal.start_vision()
        return web.json_response({"status": "ok" if success else "failed"})

    async def handle_vision_stop(self, request: "web.Request") -> "web.Response":
        """Handle POST /vision/stop - stop webcam."""
        if not self.multimodal:
            return web.json_response({"error": "Multimodal not available"}, status=503)

        self.multimodal.stop_vision()
        return web.json_response({"status": "ok"})

    async def handle_personality_mode(self, request: "web.Request") -> "web.Response":
        """Handle POST /personality - set personality mode."""
        if not PERSONALITY_AVAILABLE:
            return web.json_response({"error": "Personality not available"}, status=503)

        try:
            data = await request.json()
            mode = data.get("mode", "")

            if not mode:
                # Return current mode and available modes
                return web.json_response({
                    "current_mode": get_current_mode(),
                    "available_modes": ["starfleet", "red_dwarf", "time_lord", "colonial_fleet"],
                })

            success = set_personality_mode(mode)
            if success:
                return web.json_response({
                    "status": "ok",
                    "mode": mode,
                    "greeting": get_greeting(),
                })
            else:
                return web.json_response({"error": f"Unknown mode: {mode}"}, status=400)

        except Exception as e:
            logger.error(f"Personality mode error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_multimodal_message(self, request: "web.Request") -> "web.Response":
        """
        Handle POST /multimodal - full multimodal interaction.

        Combines: voice input → text processing → voice output
        Optionally includes vision if query requires it.
        """
        if not self.multimodal:
            return web.json_response({"error": "Multimodal not available"}, status=503)

        try:
            data = await request.json()
            user_id = data.get("user_id", "anonymous")
            use_voice_input = data.get("voice_input", True)
            use_voice_output = data.get("voice_output", True)
            emotion = data.get("emotion", "neutral")

            # Get input (voice or text)
            if use_voice_input:
                text = self.multimodal.listen()
                if not text:
                    return web.json_response({"status": "no_speech", "reply": None})
            else:
                text = data.get("text", "")
                if not text:
                    return web.json_response({"error": "No input"}, status=400)

            # Check if vision needed
            use_vision = self.multimodal.needs_vision(text)
            vision_description = None

            if use_vision:
                self.multimodal.start_vision()
                await asyncio.sleep(0.3)  # Camera warmup
                vision_description = self.multimodal.see(text)

            # Process through main handler
            result = await self.handle_message(user_id, text)
            reply = result.get("reply", "")

            # Optionally include vision in context
            if vision_description and "I see" not in reply:
                reply = f"I see: {vision_description[:80]}. {reply}"

            # Voice output
            if use_voice_output and reply:
                self.multimodal.speak(reply, emotion=emotion)

            return web.json_response({
                "status": "ok",
                "input_text": text,
                "reply": reply,
                "vision_used": use_vision,
                "vision_description": vision_description,
                **{k: v for k, v in result.items() if k != "reply"}
            })

        except Exception as e:
            logger.error(f"Multimodal message error: {e}", exc_info=True)
            return web.json_response({"error": str(e)}, status=500)

    # =========================================================================
    # Server lifecycle
    # =========================================================================

    def _setup_routes(self, app: "web.Application") -> None:
        """Set up HTTP routes."""
        # Core routes
        app.router.add_post("/message", self.handle_message_http)
        app.router.add_get("/status", self.handle_status)
        app.router.add_post("/kill", self.handle_kill)
        app.router.add_post("/unkill", self.handle_unkill)

        # Multimodal routes
        if MULTIMODAL_AVAILABLE:
            app.router.add_post("/voice/listen", self.handle_voice_input)
            app.router.add_post("/voice/speak", self.handle_voice_output)
            app.router.add_post("/vision", self.handle_vision)
            app.router.add_post("/vision/start", self.handle_vision_start)
            app.router.add_post("/vision/stop", self.handle_vision_stop)
            app.router.add_post("/multimodal", self.handle_multimodal_message)

        # Personality route
        if PERSONALITY_AVAILABLE:
            app.router.add_post("/personality", self.handle_personality_mode)

    async def start(self) -> None:
        """Start the avatar server."""
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("aiohttp not installed. Run: pip install aiohttp")

        # Initialize multimodal if available
        if self.multimodal and MULTIMODAL_AVAILABLE:
            await initialize_multimodal()
            logger.info("Multimodal avatar initialized")

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
