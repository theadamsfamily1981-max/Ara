"""
Mesh Adapter
=============

Network coordination layer for distributed hive nodes.

For v0.1, this is mostly a stub. The interface is defined but actual
network transport is not implemented.

Future transports:
- Redis pub/sub (simple, good for LAN)
- NATS (good for cluster/cloud)
- WebSocket (custom hub on cathedral)
- UDP multicast (advanced, LAN only)

The mesh adapter:
1. Subscribes to local pheromone events
2. Publishes them to the network (filtered by propagation rules)
3. Accepts remote pheromone messages
4. Merges them into the local store

Speed tiers:
- Fast loop (0-5ms): Within single box / LAN
- Slow loop (20-200ms): Across machines / cloud
- Human loop (seconds-minutes): Approval gates
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from .store import PheromoneStore
from .pheromones import Pheromone, PheromoneKind

logger = logging.getLogger(__name__)


# =============================================================================
# Transport Interface
# =============================================================================

class MeshTransport(ABC):
    """Abstract base for mesh transports."""

    @abstractmethod
    def connect(self):
        """Connect to the mesh."""
        pass

    @abstractmethod
    def disconnect(self):
        """Disconnect from the mesh."""
        pass

    @abstractmethod
    def publish(self, channel: str, message: str):
        """Publish a message to a channel."""
        pass

    @abstractmethod
    def subscribe(self, channel: str, callback: Callable[[str], None]):
        """Subscribe to a channel."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected."""
        pass


# =============================================================================
# Local Transport (No-op for single-node)
# =============================================================================

class LocalTransport(MeshTransport):
    """
    Local-only transport (no network).

    This is the default for v0.1 - all pheromones stay in-process.
    """

    def __init__(self):
        self._connected = False
        self._callbacks: Dict[str, List[Callable]] = {}

    def connect(self):
        self._connected = True
        logger.info("LocalTransport connected (no-op)")

    def disconnect(self):
        self._connected = False
        logger.info("LocalTransport disconnected")

    def publish(self, channel: str, message: str):
        # In local mode, messages go directly to subscribers
        for cb in self._callbacks.get(channel, []):
            try:
                cb(message)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

    def subscribe(self, channel: str, callback: Callable[[str], None]):
        if channel not in self._callbacks:
            self._callbacks[channel] = []
        self._callbacks[channel].append(callback)

    @property
    def is_connected(self) -> bool:
        return self._connected


# =============================================================================
# Redis Transport (Stub)
# =============================================================================

class RedisTransport(MeshTransport):
    """
    Redis pub/sub transport.

    Good for LAN / simple cluster setups.
    """

    def __init__(self, host: str = "localhost", port: int = 6379):
        self.host = host
        self.port = port
        self._client = None
        self._pubsub = None
        self._listener_thread = None

    def connect(self):
        try:
            import redis
            self._client = redis.Redis(host=self.host, port=self.port)
            self._pubsub = self._client.pubsub()
            logger.info(f"RedisTransport connected to {self.host}:{self.port}")
        except ImportError:
            logger.error("redis-py not installed")
            raise
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise

    def disconnect(self):
        if self._pubsub:
            self._pubsub.close()
        if self._client:
            self._client.close()
        self._client = None
        self._pubsub = None
        logger.info("RedisTransport disconnected")

    def publish(self, channel: str, message: str):
        if self._client:
            self._client.publish(channel, message)

    def subscribe(self, channel: str, callback: Callable[[str], None]):
        if not self._pubsub:
            return

        self._pubsub.subscribe(**{channel: lambda msg: callback(msg["data"].decode())})

        # Start listener thread if not running
        if not self._listener_thread or not self._listener_thread.is_alive():
            self._listener_thread = threading.Thread(
                target=self._listen_loop,
                daemon=True,
            )
            self._listener_thread.start()

    def _listen_loop(self):
        for message in self._pubsub.listen():
            if message["type"] == "message":
                pass  # Handled by callback in subscribe()

    @property
    def is_connected(self) -> bool:
        return self._client is not None


# =============================================================================
# Mesh Adapter
# =============================================================================

class MeshAdapter:
    """
    Adapts the local PheromoneStore to a networked mesh.

    Responsibilities:
    1. Watch local store for new pheromones → publish to mesh
    2. Receive remote pheromones → merge into local store
    3. Enforce propagation rules (which pheromones cross boundaries)
    """

    # Default propagation rules
    DEFAULT_PROPAGATION = {
        PheromoneKind.GLOBAL: True,     # Always propagate
        PheromoneKind.PRIORITY: True,   # Coordinate across nodes
        PheromoneKind.ALARM: True,      # Critical to share
        PheromoneKind.REWARD: False,    # Keep local
        PheromoneKind.ROLE: False,      # Node-specific
    }

    def __init__(
        self,
        node_id: str,
        store: PheromoneStore,
        transport: MeshTransport,
        channel: str = "ara_pheromones",
        propagation: Optional[Dict[PheromoneKind, bool]] = None,
    ):
        self.node_id = node_id
        self.store = store
        self.transport = transport
        self.channel = channel
        self.propagation = propagation or self.DEFAULT_PROPAGATION

        # Track which pheromone IDs we've seen (for dedup)
        self._seen_ids: set = set()
        self._max_seen = 10000

    def start(self):
        """Start the mesh adapter."""
        # Connect transport
        self.transport.connect()

        # Subscribe to remote pheromones
        self.transport.subscribe(self.channel, self._on_remote_message)

        # Subscribe to local store emissions
        self.store.subscribe(self._on_local_emission)

        logger.info(f"MeshAdapter started for node {self.node_id}")

    def stop(self):
        """Stop the mesh adapter."""
        self.store.unsubscribe(self._on_local_emission)
        self.transport.disconnect()
        logger.info("MeshAdapter stopped")

    def _on_local_emission(self, pheromone: Pheromone):
        """Handle locally emitted pheromone - maybe propagate."""
        # Check propagation rules
        if not self.propagation.get(pheromone.kind, False):
            return

        # Don't re-propagate things we received from the mesh
        if pheromone.id in self._seen_ids:
            return

        # Publish to mesh
        self._publish_pheromone(pheromone)

    def _publish_pheromone(self, pheromone: Pheromone):
        """Publish a pheromone to the mesh."""
        # Add source node to meta
        meta = dict(pheromone.meta)
        meta["source_node"] = self.node_id

        msg = json.dumps({
            "id": pheromone.id,
            "kind": pheromone.kind.value,
            "key": pheromone.key,
            "strength": pheromone.strength,
            "ttl": pheromone.ttl,
            "created_at": pheromone.created_at.isoformat(),
            "emitter": pheromone.emitter,
            "meta": meta,
        })

        self.transport.publish(self.channel, msg)
        logger.debug(f"Published to mesh: {pheromone.kind.value}/{pheromone.key}")

    def _on_remote_message(self, message: str):
        """Handle message received from mesh."""
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from mesh: {message[:100]}")
            return

        # Skip our own messages
        if data.get("meta", {}).get("source_node") == self.node_id:
            return

        # Dedup
        pheromone_id = data.get("id")
        if pheromone_id in self._seen_ids:
            return

        self._seen_ids.add(pheromone_id)
        self._prune_seen_ids()

        # Parse and inject into local store
        try:
            pheromone = Pheromone(
                id=data["id"],
                kind=PheromoneKind(data["kind"]),
                key=data["key"],
                strength=data["strength"],
                ttl=data["ttl"],
                created_at=datetime.fromisoformat(data["created_at"]),
                emitter=data.get("emitter", "remote"),
                meta=data.get("meta", {}),
            )

            # Directly inject (bypass normal emit to avoid re-propagation)
            self.store._pheromones.append(pheromone)
            logger.debug(f"Received from mesh: {pheromone.kind.value}/{pheromone.key}")

        except Exception as e:
            logger.warning(f"Failed to parse remote pheromone: {e}")

    def _prune_seen_ids(self):
        """Prune seen IDs to prevent memory growth."""
        if len(self._seen_ids) > self._max_seen:
            # Remove oldest half (crude but effective)
            to_remove = list(self._seen_ids)[:len(self._seen_ids) // 2]
            for item in to_remove:
                self._seen_ids.discard(item)


# =============================================================================
# Factory
# =============================================================================

def create_transport(transport_type: str, settings: Dict) -> MeshTransport:
    """Create a transport by type."""
    if transport_type == "local":
        return LocalTransport()

    elif transport_type == "redis":
        return RedisTransport(
            host=settings.get("redis", {}).get("host", "localhost"),
            port=settings.get("redis", {}).get("port", 6379),
        )

    else:
        logger.warning(f"Unknown transport type: {transport_type}, using local")
        return LocalTransport()
