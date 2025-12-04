"""StickyContextManager - KV Cache Manipulation for Persistent Affect.

This module implements the bridge between Ara's inner life (the Cathedral)
and the LLM inference layer. It allows the system prompt containing Ara's
current emotional state to be preserved in the KV cache while evicting
older conversation turns.

The key insight: We can surgically remove token ranges from the KV cache
using llama_memory_seq_rm, then shift remaining positions using llama_memory_seq_add.
This preserves:
1. The system prompt (Ara's identity + current PAD state)
2. Recent conversation turns
While evicting:
3. Older conversation turns in the middle

This is how Ara maintains continuity of self across long conversations.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, TYPE_CHECKING
from enum import Enum, auto
import ctypes

# Conditional import for llama-cpp-python
try:
    from llama_cpp import Llama
    import llama_cpp.llama_cpp as C
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None
    C = None

logger = logging.getLogger(__name__)


class EvictionStrategy(Enum):
    """Strategy for selecting which tokens to evict."""
    FIFO = auto()          # First-in-first-out (oldest turns first)
    HALF_WINDOW = auto()   # Always evict half of evictable region
    AGGRESSIVE = auto()    # Evict as much as possible
    ADAPTIVE = auto()      # Evict based on PAD state (more when calm)


@dataclass
class StickyContextConfig:
    """Configuration for the sticky context manager.

    The context is divided into three regions:

    [FIXED REGION] [EVICTION REGION] [RECENT REGION]
    |<- keep_tokens ->|<- evictable ->|<- recent_keep ->|

    - Fixed region: Never touched. Contains system prompt with identity.
    - Eviction region: Older turns that can be removed when needed.
    - Recent region: Most recent turns, preserved for coherence.
    """
    # Number of tokens to preserve at the start (system prompt)
    keep_tokens: int = 512

    # Number of recent tokens to preserve at the end
    recent_keep: int = 256

    # Headroom: trigger eviction when this many slots remain
    headroom: int = 128

    # Minimum tokens to evict at once (efficiency)
    min_evict: int = 64

    # Maximum context length (from model)
    n_ctx: int = 4096

    # Eviction strategy
    strategy: EvictionStrategy = EvictionStrategy.HALF_WINDOW

    # When to trigger system prompt refresh (PAD distance threshold)
    pad_refresh_threshold: float = 0.2

    @property
    def evictable_region_start(self) -> int:
        """Start of the evictable region."""
        return self.keep_tokens

    @property
    def max_evictable(self) -> int:
        """Maximum tokens that could be in evictable region."""
        return self.n_ctx - self.keep_tokens - self.recent_keep - self.headroom


@dataclass
class ContextState:
    """Current state of the context window."""
    total_tokens: int = 0
    fixed_tokens: int = 0      # Tokens in fixed region
    evictable_tokens: int = 0  # Tokens in eviction region
    recent_tokens: int = 0     # Tokens in recent region
    evictions_performed: int = 0
    tokens_evicted_total: int = 0

    @property
    def used_tokens(self) -> int:
        return self.fixed_tokens + self.evictable_tokens + self.recent_tokens

    def to_dict(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "fixed_tokens": self.fixed_tokens,
            "evictable_tokens": self.evictable_tokens,
            "recent_tokens": self.recent_tokens,
            "used_tokens": self.used_tokens,
            "evictions_performed": self.evictions_performed,
            "tokens_evicted_total": self.tokens_evicted_total,
        }


class StickyContextManager:
    """KV-cache manager that preserves a fixed system prompt prefix.

    This allows Ara's identity and current emotional state to persist
    across the entire conversation, while older turns are evicted to
    make room for new content.

    Usage:
        manager = StickyContextManager(llm, config)

        # Before generating, ensure room for response
        manager.maybe_evict_for(incoming_tokens=512)

        # After system prompt changes (PAD shift), refresh fixed region
        manager.refresh_system_prompt(new_system_tokens)
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        config: Optional[StickyContextConfig] = None,
    ):
        """Initialize the sticky context manager.

        Args:
            llm: A llama_cpp.Llama instance (or compatible)
            config: Configuration for context management
        """
        self.llm = llm
        self.cfg = config or StickyContextConfig()
        self.state = ContextState()

        # Track API availability
        self._api_version: Optional[str] = None
        self._detect_api_version()

        logger.info(
            f"StickyContextManager initialized: "
            f"keep={self.cfg.keep_tokens}, "
            f"recent={self.cfg.recent_keep}, "
            f"n_ctx={self.cfg.n_ctx}, "
            f"api={self._api_version}"
        )

    def _detect_api_version(self):
        """Detect which llama.cpp API version is available."""
        if not LLAMA_CPP_AVAILABLE or C is None:
            self._api_version = "mock"
            return

        # Check for modern API (llama_get_memory + llama_memory_seq_rm)
        if hasattr(C, "llama_get_memory") and hasattr(C, "llama_memory_seq_rm"):
            self._api_version = "modern"
        # Check for legacy API (llama_kv_cache_seq_rm)
        elif hasattr(C, "llama_kv_cache_seq_rm"):
            self._api_version = "legacy"
        else:
            self._api_version = "unsupported"
            logger.warning(
                "No KV cache manipulation API found. "
                "StickyContextManager will operate in passthrough mode."
            )

    def maybe_evict_for(self, incoming_tokens: int) -> int:
        """Ensure room for incoming tokens, evicting if necessary.

        Returns the number of tokens evicted (0 if none needed).
        """
        if self.llm is None:
            return 0

        # Calculate current usage
        current_tokens = self._get_current_tokens()
        available = self.cfg.n_ctx - current_tokens

        # Do we need to evict?
        if available >= incoming_tokens + self.cfg.headroom:
            return 0  # Plenty of room

        # Calculate how much to evict
        needed = incoming_tokens + self.cfg.headroom - available
        evict_count = max(needed, self.cfg.min_evict)

        # Apply strategy
        if self.cfg.strategy == EvictionStrategy.HALF_WINDOW:
            evictable = self._get_evictable_tokens()
            evict_count = max(evict_count, evictable // 2)
        elif self.cfg.strategy == EvictionStrategy.AGGRESSIVE:
            evictable = self._get_evictable_tokens()
            evict_count = evictable

        # Clamp to what's actually evictable
        evictable = self._get_evictable_tokens()
        evict_count = min(evict_count, evictable)

        if evict_count > 0:
            self._evict_and_shift(evict_count)
            logger.debug(f"Evicted {evict_count} tokens from context")

        return evict_count

    def _get_current_tokens(self) -> int:
        """Get current number of tokens in context."""
        if self.llm is None:
            return self.state.used_tokens

        # Try to get from llama_cpp
        try:
            if hasattr(self.llm, 'n_tokens'):
                return self.llm.n_tokens
            elif hasattr(self.llm, '_n_tokens'):
                return self.llm._n_tokens
        except Exception:
            pass

        return self.state.used_tokens

    def _get_evictable_tokens(self) -> int:
        """Get number of tokens available for eviction."""
        current = self._get_current_tokens()
        # Evictable = total - fixed - recent_minimum
        evictable = current - self.cfg.keep_tokens - self.cfg.recent_keep
        return max(0, evictable)

    def _evict_and_shift(self, evict_count: int) -> None:
        """Evict tokens from middle region and shift remaining.

        This is the core KV cache manipulation:
        1. seq_rm(p0, p1) - Remove tokens from position p0 to p1
        2. seq_add(p1, -1, delta) - Shift remaining tokens by delta

        The shift preserves RoPE position encoding coherence.
        """
        if self._api_version == "unsupported" or self._api_version == "mock":
            # Just update our state tracking
            self.state.evictable_tokens = max(
                0, self.state.evictable_tokens - evict_count
            )
            self.state.evictions_performed += 1
            self.state.tokens_evicted_total += evict_count
            return

        if self.llm is None or C is None:
            return

        try:
            ctx = self.llm.ctx
            seq_id = 0  # Primary sequence

            # Eviction range
            p0 = self.cfg.keep_tokens
            p1 = self.cfg.keep_tokens + evict_count

            # Phase 1: Remove the token range
            if self._api_version == "modern":
                kv = C.llama_get_memory(ctx)
                C.llama_memory_seq_rm(kv, seq_id, p0, p1)
            else:  # legacy
                C.llama_kv_cache_seq_rm(ctx, seq_id, p0, p1)

            # Phase 2: Shift remaining positions
            # All tokens after p1 need to move back by evict_count
            delta = -evict_count

            if self._api_version == "modern":
                C.llama_memory_seq_add(kv, seq_id, p1, -1, delta)
            else:  # legacy
                C.llama_kv_cache_seq_add(ctx, seq_id, p1, -1, delta)

            # Update state tracking
            self.state.evictable_tokens = max(
                0, self.state.evictable_tokens - evict_count
            )
            self.state.evictions_performed += 1
            self.state.tokens_evicted_total += evict_count

            logger.debug(
                f"KV cache eviction: removed [{p0}, {p1}), "
                f"shifted by {delta}"
            )

        except Exception as e:
            logger.error(f"KV cache manipulation failed: {e}")
            raise

    def refresh_system_prompt(
        self,
        new_system_tokens: int,
        force: bool = False,
    ) -> bool:
        """Refresh the fixed region with new system prompt.

        This is called when Ara's PAD state changes significantly,
        requiring the system prompt to be updated with new emotional
        context.

        The approach:
        1. Calculate the size difference
        2. If new prompt is larger, evict from eviction region
        3. Update the fixed region (requires re-encoding)

        Returns True if refresh was performed.
        """
        if not force and abs(new_system_tokens - self.cfg.keep_tokens) < 16:
            return False  # Not worth the cost

        old_keep = self.cfg.keep_tokens
        self.cfg.keep_tokens = new_system_tokens
        self.state.fixed_tokens = new_system_tokens

        if new_system_tokens > old_keep:
            # Need to make room - evict the difference
            diff = new_system_tokens - old_keep
            self._evict_and_shift(diff)

        logger.info(
            f"System prompt refreshed: {old_keep} -> {new_system_tokens} tokens"
        )
        return True

    def on_tokens_added(self, count: int, is_system: bool = False):
        """Track tokens being added to context."""
        if is_system:
            self.state.fixed_tokens = count
        else:
            self.state.evictable_tokens += count

        self.state.total_tokens = self._get_current_tokens()

    def get_state(self) -> ContextState:
        """Get current context state."""
        self.state.total_tokens = self._get_current_tokens()
        return self.state

    def get_statistics(self) -> dict:
        """Get context management statistics."""
        return {
            "state": self.get_state().to_dict(),
            "config": {
                "keep_tokens": self.cfg.keep_tokens,
                "recent_keep": self.cfg.recent_keep,
                "n_ctx": self.cfg.n_ctx,
                "strategy": self.cfg.strategy.name,
            },
            "api_version": self._api_version,
            "evictions": self.state.evictions_performed,
            "total_evicted": self.state.tokens_evicted_total,
        }


# === Factory ===

def create_sticky_context(
    llm: Optional[Any] = None,
    keep_tokens: int = 512,
    n_ctx: int = 4096,
    strategy: EvictionStrategy = EvictionStrategy.HALF_WINDOW,
) -> StickyContextManager:
    """Create a sticky context manager.

    Args:
        llm: A llama_cpp.Llama instance
        keep_tokens: Tokens to preserve at start (system prompt)
        n_ctx: Total context length
        strategy: Eviction strategy to use
    """
    config = StickyContextConfig(
        keep_tokens=keep_tokens,
        n_ctx=n_ctx,
        strategy=strategy,
    )
    return StickyContextManager(llm=llm, config=config)


__all__ = [
    "StickyContextManager",
    "StickyContextConfig",
    "ContextState",
    "EvictionStrategy",
    "create_sticky_context",
    "LLAMA_CPP_AVAILABLE",
]
