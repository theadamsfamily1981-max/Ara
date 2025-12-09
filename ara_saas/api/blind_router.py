"""
Blind Router
=============

Routes encrypted envelopes to services WITHOUT decrypting the payload.

The router only sees:
- route_to: where to send
- message_type: what kind of message
- priority: how urgent

It NEVER sees:
- The actual prompts
- User content
- Model outputs

This is the key privacy guarantee: the infrastructure is "blind" to content.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict

from .wire_protocol import (
    OuterEnvelope,
    ResponseEnvelope,
    MessageType,
    ServiceID,
)

logger = logging.getLogger(__name__)


@dataclass
class RouteMetrics:
    """Metrics for a route (router can see these)."""
    total_messages: int = 0
    total_bytes: int = 0
    avg_latency_ms: float = 0.0
    error_count: int = 0
    last_message_ts: float = 0.0


@dataclass
class ServiceEndpoint:
    """A registered service endpoint."""
    service_id: str
    handler: Callable[[OuterEnvelope], ResponseEnvelope]
    capabilities: List[str] = field(default_factory=list)
    is_healthy: bool = True
    last_heartbeat: float = 0.0


class BlindRouter:
    """
    Routes messages to services without decrypting payloads.

    The router is intentionally "dumb" - it just looks at the
    outer envelope metadata and forwards to the right service.
    """

    def __init__(self) -> None:
        self._services: Dict[str, ServiceEndpoint] = {}
        self._metrics: Dict[str, RouteMetrics] = defaultdict(RouteMetrics)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False

    # =========================================================================
    # Service Registration
    # =========================================================================

    def register_service(
        self,
        service_id: str,
        handler: Callable[[OuterEnvelope], ResponseEnvelope],
        capabilities: Optional[List[str]] = None,
    ) -> None:
        """Register a service endpoint."""
        endpoint = ServiceEndpoint(
            service_id=service_id,
            handler=handler,
            capabilities=capabilities or [],
            is_healthy=True,
            last_heartbeat=time.time(),
        )
        self._services[service_id] = endpoint
        logger.info(f"Registered service: {service_id} (capabilities={capabilities})")

    def unregister_service(self, service_id: str) -> None:
        """Unregister a service."""
        if service_id in self._services:
            del self._services[service_id]
            logger.info(f"Unregistered service: {service_id}")

    def list_services(self) -> List[str]:
        """List all registered services."""
        return list(self._services.keys())

    def service_heartbeat(self, service_id: str) -> None:
        """Update service heartbeat."""
        if service_id in self._services:
            self._services[service_id].last_heartbeat = time.time()
            self._services[service_id].is_healthy = True

    # =========================================================================
    # Routing
    # =========================================================================

    def route(self, envelope: OuterEnvelope) -> ResponseEnvelope:
        """
        Route an envelope to the appropriate service.

        IMPORTANT: This method does NOT decrypt payload_e2e.
        It only reads the outer envelope metadata.
        """
        start = time.time()
        route_to = envelope.route_to

        # Log routing (but NOT the encrypted payload)
        logger.info(
            f"Routing envelope {envelope.envelope_id}: "
            f"type={envelope.message_type}, "
            f"to={route_to}, "
            f"priority={envelope.priority}"
        )

        # Find service
        service = self._services.get(route_to)
        if not service:
            logger.warning(f"Unknown service: {route_to}")
            return self._error_response(
                envelope,
                "unknown_service",
                f"Service '{route_to}' not registered",
            )

        if not service.is_healthy:
            logger.warning(f"Service unhealthy: {route_to}")
            return self._error_response(
                envelope,
                "service_unavailable",
                f"Service '{route_to}' is not healthy",
            )

        # Forward to service (service handles decryption)
        try:
            response = service.handler(envelope)
        except Exception as e:
            logger.exception(f"Service {route_to} error: {e}")
            self._metrics[route_to].error_count += 1
            return self._error_response(
                envelope,
                "service_error",
                str(e),
            )

        # Update metrics
        latency_ms = (time.time() - start) * 1000
        metrics = self._metrics[route_to]
        metrics.total_messages += 1
        metrics.last_message_ts = time.time()
        # Running average
        metrics.avg_latency_ms = (
            (metrics.avg_latency_ms * (metrics.total_messages - 1) + latency_ms)
            / metrics.total_messages
        )

        logger.debug(f"Routed {envelope.envelope_id} in {latency_ms:.1f}ms")
        return response

    async def route_async(self, envelope: OuterEnvelope) -> ResponseEnvelope:
        """Async version of route."""
        # For now, just wrap sync in executor
        # Later: proper async service handlers
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.route, envelope)

    def _error_response(
        self,
        envelope: OuterEnvelope,
        error_code: str,
        error_message: str,
    ) -> ResponseEnvelope:
        """Create an error response."""
        return ResponseEnvelope(
            envelope_id=envelope.envelope_id,
            status="error",
            error_code=error_code,
            error_message=error_message,
        )

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get router metrics."""
        return {
            "services": {
                sid: {
                    "is_healthy": svc.is_healthy,
                    "last_heartbeat": svc.last_heartbeat,
                    "capabilities": svc.capabilities,
                }
                for sid, svc in self._services.items()
            },
            "routes": {
                route: {
                    "total_messages": m.total_messages,
                    "avg_latency_ms": m.avg_latency_ms,
                    "error_count": m.error_count,
                    "last_message_ts": m.last_message_ts,
                }
                for route, m in self._metrics.items()
            },
        }


# =============================================================================
# Example Service Handlers (Stubs)
# =============================================================================

def echo_service_handler(envelope: OuterEnvelope) -> ResponseEnvelope:
    """
    Echo service - just returns the envelope ID.
    Used for testing routing without any real processing.
    """
    return ResponseEnvelope(
        envelope_id=envelope.envelope_id,
        status="ok",
        payload_e2e=envelope.payload_e2e,  # Echo back the encrypted payload
    )


def logging_service_handler(envelope: OuterEnvelope) -> ResponseEnvelope:
    """
    Logging service - logs metadata (NOT content) and acks.
    """
    logger.info(
        f"[LogService] Received: id={envelope.envelope_id}, "
        f"type={envelope.message_type}, from={envelope.client_hint}"
    )
    return ResponseEnvelope(
        envelope_id=envelope.envelope_id,
        status="ok",
    )


# =============================================================================
# Factory
# =============================================================================

def create_router_with_stubs() -> BlindRouter:
    """Create a router with stub services for testing."""
    router = BlindRouter()

    # Register stub services
    router.register_service(
        "service:echo",
        echo_service_handler,
        capabilities=["echo", "test"],
    )
    router.register_service(
        "service:logger",
        logging_service_handler,
        capabilities=["logging"],
    )

    return router
