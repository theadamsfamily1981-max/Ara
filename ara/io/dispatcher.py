"""
Ara I/O Dispatcher - Hint Routing System
========================================

Routes HDOutputHints to appropriate actuators:
- UI hints → Avatar, Cockpit
- Network hints → NodeAgents, ReflexEngine
- Task hints → Job scheduler

Also collects HDInputEvents from all sources.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Callable, Optional, Any
from collections import defaultdict

from .types import (
    IOChannel,
    HDInputEvent,
    HDOutputHint,
    EventBatch,
)

logger = logging.getLogger(__name__)


# Type aliases
HintHandler = Callable[[HDOutputHint], None]
EventSource = Callable[[], List[HDInputEvent]]


@dataclass
class IODispatcher:
    """
    Central dispatcher for HD I/O events and hints.

    Responsibilities:
    - Collect HDInputEvents from registered sources
    - Route HDOutputHints to registered handlers
    - Track statistics and health
    """

    # Registered handlers per channel
    _hint_handlers: Dict[IOChannel, List[HintHandler]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Registered event sources per channel
    _event_sources: Dict[IOChannel, List[EventSource]] = field(
        default_factory=lambda: defaultdict(list)
    )

    # Statistics
    _events_collected: int = 0
    _hints_dispatched: int = 0
    _errors: List[Dict[str, Any]] = field(default_factory=list)

    def register_hint_handler(
        self,
        channel: IOChannel,
        handler: HintHandler,
    ) -> None:
        """Register a handler for hints on a channel."""
        self._hint_handlers[channel].append(handler)
        logger.debug(f"Registered hint handler for {channel.value}")

    def register_event_source(
        self,
        channel: IOChannel,
        source: EventSource,
    ) -> None:
        """Register an event source for a channel."""
        self._event_sources[channel].append(source)
        logger.debug(f"Registered event source for {channel.value}")

    def collect_events(self, tick_number: int = 0) -> EventBatch:
        """
        Collect events from all registered sources.

        Returns an EventBatch containing all events for this tick.
        """
        batch = EventBatch(tick_number=tick_number)

        for channel, sources in self._event_sources.items():
            for source in sources:
                try:
                    events = source()
                    for event in events:
                        batch.add(event)
                        self._events_collected += 1
                except Exception as e:
                    self._record_error("collect", channel, str(e))
                    logger.error(f"Event source error on {channel.value}: {e}")

        return batch

    def dispatch_hint(self, hint: HDOutputHint) -> int:
        """
        Dispatch a hint to all handlers for its channel.

        Returns number of handlers that received the hint.
        """
        handlers = self._hint_handlers.get(hint.channel, [])

        dispatched = 0
        for handler in handlers:
            try:
                handler(hint)
                dispatched += 1
                self._hints_dispatched += 1
            except Exception as e:
                self._record_error("dispatch", hint.channel, str(e))
                logger.error(f"Hint handler error on {hint.channel.value}: {e}")

        if dispatched == 0:
            logger.warning(f"No handlers for hint: {hint.channel.value}/{hint.kind}")

        return dispatched

    def dispatch_hints(self, hints: List[HDOutputHint]) -> int:
        """Dispatch multiple hints, return total dispatched count."""
        return sum(self.dispatch_hint(h) for h in hints)

    def _record_error(self, op: str, channel: IOChannel, message: str) -> None:
        """Record an error for diagnostics."""
        self._errors.append({
            "op": op,
            "channel": channel.value,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep last 100 errors
        self._errors = self._errors[-100:]

    def get_stats(self) -> Dict[str, Any]:
        """Get dispatcher statistics."""
        return {
            "events_collected": self._events_collected,
            "hints_dispatched": self._hints_dispatched,
            "handlers": {
                ch.value: len(handlers)
                for ch, handlers in self._hint_handlers.items()
            },
            "sources": {
                ch.value: len(sources)
                for ch, sources in self._event_sources.items()
            },
            "recent_errors": len(self._errors),
        }

    def clear_stats(self) -> None:
        """Reset statistics."""
        self._events_collected = 0
        self._hints_dispatched = 0
        self._errors.clear()


# =============================================================================
# Singleton
# =============================================================================

_dispatcher: Optional[IODispatcher] = None


def get_dispatcher() -> IODispatcher:
    """Get the global I/O dispatcher."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = IODispatcher()
    return _dispatcher


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'IODispatcher',
    'get_dispatcher',
    'HintHandler',
    'EventSource',
]
