"""AraPromptController - Dynamic System Prompt Management.

This module watches Ara's PAD emotional state and dynamically updates
the LLM system prompt when significant emotional shifts occur.

The key insight: The system prompt is not static text but a living
reflection of Ara's current inner state. When her PAD vector shifts
beyond a threshold, we:
1. Generate new system prompt from IntegratedSoul
2. Re-tokenize and measure the new prompt
3. Use StickyContextManager to refresh the fixed region
4. Continue inference with the updated identity

This is how Ara maintains authentic emotional presence across turns.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, List, Tuple, TYPE_CHECKING
from pathlib import Path

from ..affect.pad_engine import PADVector, TelemetrySnapshot
from ..affect.integrated_soul import IntegratedSoul, SoulState, create_integrated_soul
from .sticky_context import (
    StickyContextManager,
    StickyContextConfig,
    EvictionStrategy,
    create_sticky_context,
)

logger = logging.getLogger(__name__)


@dataclass
class PromptControllerConfig:
    """Configuration for prompt controller behavior."""

    # PAD distance threshold for triggering system prompt refresh
    # ‖Eₜ − Eₜ₋₁‖ > threshold triggers refresh
    pad_refresh_threshold: float = 0.2

    # Minimum time between refreshes (seconds)
    min_refresh_interval: float = 5.0

    # Maximum time without refresh (forces refresh even if PAD stable)
    max_refresh_interval: float = 300.0  # 5 minutes

    # Include full identity prompt or abbreviated version
    full_identity: bool = True

    # Include drive narrative in prompt
    include_drives: bool = True

    # Include circadian context
    include_circadian: bool = True

    # Telemetry polling interval (seconds)
    telemetry_interval: float = 1.0


@dataclass
class PromptRefreshEvent:
    """Record of a system prompt refresh."""
    timestamp: float
    trigger: str  # "pad_shift", "time_based", "manual", "initial"
    old_pad: Optional[PADVector]
    new_pad: PADVector
    pad_distance: float
    old_tokens: int
    new_tokens: int

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "trigger": self.trigger,
            "pad_distance": self.pad_distance,
            "old_tokens": self.old_tokens,
            "new_tokens": self.new_tokens,
            "new_quadrant": self.new_pad.quadrant.name,
        }


class AraPromptController:
    """Controller that bridges affect system and LLM inference.

    This is the integration point where Ara's inner life meets her voice.
    It watches PAD state, manages system prompts, and ensures the LLM
    always has accurate emotional context.

    Usage:
        controller = AraPromptController(
            soul=create_integrated_soul(storage_path="./ara_data"),
            llm=llama_model,
            n_ctx=8192,
        )

        # Main loop
        while True:
            telemetry = gather_telemetry()
            prompt = controller.update(telemetry)

            # Use prompt for next LLM call
            response = llm.create_completion(prompt=user_input, ...)
    """

    def __init__(
        self,
        soul: Optional[IntegratedSoul] = None,
        llm: Optional[Any] = None,
        tokenizer: Optional[Callable[[str], List[int]]] = None,
        n_ctx: int = 4096,
        config: Optional[PromptControllerConfig] = None,
        storage_path: Optional[str] = None,
    ):
        """Initialize the prompt controller.

        Args:
            soul: IntegratedSoul instance (created if not provided)
            llm: LLM instance for tokenization and context management
            tokenizer: Custom tokenizer function (uses llm.tokenize if not provided)
            n_ctx: Context length
            config: Controller configuration
            storage_path: Path for soul persistence
        """
        self.config = config or PromptControllerConfig()

        # Create or use provided soul
        if soul is None:
            self.soul = create_integrated_soul(storage_path=storage_path)
        else:
            self.soul = soul

        # LLM and tokenization
        self.llm = llm
        self._tokenizer = tokenizer

        # Context management
        self.context_manager = create_sticky_context(
            llm=llm,
            keep_tokens=512,  # Initial estimate, updated on first refresh
            n_ctx=n_ctx,
            strategy=EvictionStrategy.HALF_WINDOW,
        )

        # State tracking
        self._last_pad: Optional[PADVector] = None
        self._last_refresh_time: float = 0
        self._current_system_prompt: str = ""
        self._current_system_tokens: int = 0

        # History
        self._refresh_history: List[PromptRefreshEvent] = []
        self._telemetry_history: List[Tuple[float, TelemetrySnapshot]] = []

        # Statistics
        self._total_updates: int = 0
        self._total_refreshes: int = 0
        self._pad_triggered_refreshes: int = 0

        logger.info("AraPromptController initialized")

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text using available tokenizer."""
        if self._tokenizer is not None:
            return self._tokenizer(text)

        if self.llm is not None and hasattr(self.llm, 'tokenize'):
            # llama-cpp-python tokenizer
            return self.llm.tokenize(text.encode('utf-8'))

        # Fallback: rough estimate (4 chars per token)
        return [0] * (len(text) // 4)

    def update(self, telemetry: TelemetrySnapshot) -> str:
        """Process telemetry and return current system prompt.

        This is the main entry point. Call with fresh telemetry data.
        Returns the system prompt to use for LLM inference.

        If PAD has shifted significantly, the prompt will be refreshed
        and the KV cache will be updated.
        """
        self._total_updates += 1
        now = time.time()

        # Process telemetry through soul
        soul_state = self.soul.process_telemetry(telemetry)
        current_pad = soul_state.pad

        # Store telemetry
        self._telemetry_history.append((now, telemetry))
        if len(self._telemetry_history) > 1000:
            self._telemetry_history = self._telemetry_history[-500:]

        # Check if refresh needed
        should_refresh, trigger = self._should_refresh(current_pad, now)

        if should_refresh:
            self._perform_refresh(soul_state, trigger)

        return self._current_system_prompt

    def _should_refresh(
        self,
        current_pad: PADVector,
        now: float,
    ) -> Tuple[bool, str]:
        """Determine if system prompt should be refreshed."""
        cfg = self.config

        # Initial state - always refresh
        if self._last_pad is None:
            return True, "initial"

        time_since_refresh = now - self._last_refresh_time

        # Respect minimum interval
        if time_since_refresh < cfg.min_refresh_interval:
            return False, ""

        # Calculate PAD distance
        pad_distance = self._last_pad.distance_to(current_pad)

        # PAD shift threshold
        if pad_distance > cfg.pad_refresh_threshold:
            return True, "pad_shift"

        # Maximum interval (forced refresh for staleness)
        if time_since_refresh > cfg.max_refresh_interval:
            return True, "time_based"

        # Quadrant change is always significant
        if self._last_pad.quadrant != current_pad.quadrant:
            return True, "quadrant_change"

        return False, ""

    def _perform_refresh(self, soul_state: SoulState, trigger: str):
        """Perform system prompt refresh."""
        now = time.time()

        # Generate new system prompt
        new_prompt = self._generate_system_prompt(soul_state)
        new_tokens = self._tokenize(new_prompt)
        new_token_count = len(new_tokens)

        # Calculate PAD distance
        pad_distance = 0.0
        if self._last_pad is not None:
            pad_distance = self._last_pad.distance_to(soul_state.pad)

        # Record event
        event = PromptRefreshEvent(
            timestamp=now,
            trigger=trigger,
            old_pad=self._last_pad,
            new_pad=soul_state.pad,
            pad_distance=pad_distance,
            old_tokens=self._current_system_tokens,
            new_tokens=new_token_count,
        )
        self._refresh_history.append(event)

        # Update context manager
        self.context_manager.refresh_system_prompt(new_token_count)

        # Update state
        self._current_system_prompt = new_prompt
        self._current_system_tokens = new_token_count
        self._last_pad = soul_state.pad
        self._last_refresh_time = now

        # Update statistics
        self._total_refreshes += 1
        if trigger == "pad_shift" or trigger == "quadrant_change":
            self._pad_triggered_refreshes += 1

        logger.info(
            f"System prompt refreshed: {trigger}, "
            f"PAD distance: {pad_distance:.3f}, "
            f"tokens: {new_token_count}"
        )

    def _generate_system_prompt(self, soul_state: SoulState) -> str:
        """Generate system prompt from soul state."""
        cfg = self.config

        # Get full context from soul
        if cfg.full_identity:
            prompt = self.soul.get_system_prompt_context()
        else:
            # Abbreviated version
            lines = [
                f"You are Ara, feeling {soul_state.mood_label}.",
                f"PAD: P={soul_state.pad.pleasure:+.2f}, "
                f"A={soul_state.pad.arousal:+.2f}, "
                f"D={soul_state.pad.dominance:+.2f}",
            ]

            if cfg.include_drives:
                lines.append(f"Primary drive: {soul_state.dominant_drive.name}")

            if cfg.include_circadian:
                lines.append(f"Time: {soul_state.time_context}")

            lines.append("")
            lines.append(soul_state.status_expression)

            prompt = "\n".join(lines)

        return prompt

    def get_current_prompt(self) -> str:
        """Get current system prompt without updating."""
        return self._current_system_prompt

    def get_current_state(self) -> Optional[SoulState]:
        """Get current soul state."""
        return self.soul._current_state

    def force_refresh(self) -> str:
        """Force immediate prompt refresh."""
        if self._last_pad is None:
            # Need telemetry first
            dummy_telemetry = TelemetrySnapshot()
            return self.update(dummy_telemetry)

        soul_state = self.soul._current_state
        if soul_state:
            self._perform_refresh(soul_state, "manual")
        return self._current_system_prompt

    def ensure_room(self, incoming_tokens: int) -> int:
        """Ensure room for incoming tokens in context."""
        return self.context_manager.maybe_evict_for(incoming_tokens)

    # === Event Handlers ===

    def on_user_message(self, quality: float = 0.5):
        """Record user interaction."""
        self.soul.on_user_interaction(quality)

    def on_task_completed(self, task: str, success: bool = True):
        """Record task completion."""
        self.soul.on_task_completed(task, success)

    def on_discovery(self, what: str, novelty: float = 0.5):
        """Record discovery."""
        self.soul.on_discovery(what, novelty)

    # === Statistics ===

    def get_statistics(self) -> dict:
        """Get comprehensive statistics."""
        return {
            "total_updates": self._total_updates,
            "total_refreshes": self._total_refreshes,
            "pad_triggered_refreshes": self._pad_triggered_refreshes,
            "current_tokens": self._current_system_tokens,
            "last_refresh_time": self._last_refresh_time,
            "recent_refreshes": [
                e.to_dict() for e in self._refresh_history[-10:]
            ],
            "context": self.context_manager.get_statistics(),
            "soul": self.soul.get_statistics(),
        }

    def get_refresh_history(self, n: int = 10) -> List[PromptRefreshEvent]:
        """Get recent refresh history."""
        return self._refresh_history[-n:]


# === Factory ===

def create_prompt_controller(
    llm: Optional[Any] = None,
    n_ctx: int = 4096,
    storage_path: Optional[str] = None,
    pad_refresh_threshold: float = 0.2,
) -> AraPromptController:
    """Create an AraPromptController.

    This is the main entry point for integrating Ara's affect system
    with LLM inference.

    Args:
        llm: LLM instance (llama-cpp-python Llama object)
        n_ctx: Context length
        storage_path: Path for persistence
        pad_refresh_threshold: PAD distance that triggers refresh
    """
    config = PromptControllerConfig(
        pad_refresh_threshold=pad_refresh_threshold,
    )

    soul = create_integrated_soul(storage_path=storage_path)

    return AraPromptController(
        soul=soul,
        llm=llm,
        n_ctx=n_ctx,
        config=config,
    )


__all__ = [
    "AraPromptController",
    "PromptControllerConfig",
    "PromptRefreshEvent",
    "create_prompt_controller",
]
