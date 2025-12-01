"""
Real-time WebSocket streaming for TF-A-N visualization.

Streams topology and attention metrics to the Topo-Attention Glass dashboard.
"""

import asyncio
import json
import time
import random
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logging.warning("websockets not available - install with: pip install websockets")


@dataclass
class MetricsSnapshot:
    """Single snapshot of training metrics."""
    ts: float
    step: int
    sparsity: float  # Fraction of tokens kept (1 - drop_rate)
    kept_idx: List[int]  # Indices of kept landmarks
    attn_block: List[List[float]]  # Attention matrix sample [head_sample, seq, seq]
    pd: List[Dict[str, float]]  # Persistence diagram points [{"b": birth, "d": death}]
    epr_cv: Optional[float] = None
    fdt_lr_delta: Optional[float] = None
    spike_rate: Optional[float] = None  # For SNN mode
    vfe: Optional[float] = None  # VFE metric


class VizStream:
    """
    WebSocket server for streaming TF-A-N metrics to visualization dashboard.

    Usage:
        stream = VizStream(host="0.0.0.0", port=8765)
        stream.run()  # Blocks

    Or async:
        await stream.serve()
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        update_interval: float = 0.5,
        demo_mode: bool = False
    ):
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets package required: pip install websockets")

        self.host = host
        self.port = port
        self.update_interval = update_interval
        self.demo_mode = demo_mode

        # Connected clients
        self.clients = set()

        # Current metrics (can be updated from training loop)
        self.current_metrics: Optional[MetricsSnapshot] = None

        # Metrics history (last N snapshots)
        self.history: List[MetricsSnapshot] = []
        self.max_history = 1000

        self.logger = logging.getLogger(__name__)

    async def register(self, websocket):
        """Register a new client connection."""
        self.clients.add(websocket)
        self.logger.info(f"Client connected. Total clients: {len(self.clients)}")

        # Send history to new client
        if self.history:
            try:
                history_payload = {
                    "type": "history",
                    "snapshots": [asdict(s) for s in self.history[-100:]]  # Last 100
                }
                await websocket.send(json.dumps(history_payload))
            except Exception as e:
                self.logger.error(f"Failed to send history: {e}")

    async def unregister(self, websocket):
        """Unregister a client connection."""
        self.clients.discard(websocket)
        self.logger.info(f"Client disconnected. Total clients: {len(self.clients)}")

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )

    async def _handler(self, websocket, path):
        """Handle individual client connection."""
        await self.register(websocket)
        try:
            async for message in websocket:
                # Echo or handle client messages if needed
                data = json.loads(message)
                if data.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong", "ts": time.time()}))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

    async def _emitter_loop(self):
        """Periodically emit metrics to clients."""
        while True:
            try:
                # Get metrics (demo or real)
                if self.demo_mode or self.current_metrics is None:
                    metrics = self._generate_demo_metrics()
                else:
                    metrics = self.current_metrics

                # Add to history
                self.history.append(metrics)
                if len(self.history) > self.max_history:
                    self.history.pop(0)

                # Broadcast to clients
                payload = {
                    "type": "update",
                    "snapshot": asdict(metrics)
                }
                await self.broadcast(json.dumps(payload))

            except Exception as e:
                self.logger.error(f"Emitter error: {e}")

            await asyncio.sleep(self.update_interval)

    def _generate_demo_metrics(self) -> MetricsSnapshot:
        """Generate synthetic demo metrics for testing."""
        step = int(time.time()) % 100000
        n_heads = 8
        seq_len = 64

        # Simulate attention sparsity pattern
        sparsity = 0.67 + random.gauss(0, 0.05)
        sparsity = max(0.5, min(0.9, sparsity))

        # Kept landmark indices
        n_kept = int(seq_len * (1 - sparsity))
        kept_idx = sorted(random.sample(range(seq_len), n_kept))

        # Sample attention matrix (one head)
        attn_block = []
        for i in range(seq_len):
            row = []
            for j in range(seq_len):
                if j <= i:  # Causal
                    # Higher attention near diagonal
                    dist = abs(i - j)
                    val = random.gauss(1.0 / (1 + dist * 0.1), 0.1)
                    val = max(0, min(1, val))
                else:
                    val = 0.0
                row.append(val)
            # Normalize row
            total = sum(row) or 1.0
            attn_block.append([v / total for v in row])

        # Persistence diagram points
        n_features = 64
        pd_points = []
        for i in range(n_features):
            # H0 features (connected components) - born early, die early/late
            if i < n_features // 2:
                birth = random.uniform(0, 0.05)
                death = random.uniform(birth + 0.01, 0.3)
            # H1 features (loops) - born later, die later
            else:
                birth = random.uniform(0.1, 0.4)
                death = random.uniform(birth + 0.05, 0.8)
            pd_points.append({"b": birth, "d": death})

        return MetricsSnapshot(
            ts=time.time(),
            step=step,
            sparsity=sparsity,
            kept_idx=kept_idx,
            attn_block=attn_block,
            pd=pd_points,
            epr_cv=random.uniform(0.12, 0.18),
            fdt_lr_delta=random.gauss(0, 0.0001),
            spike_rate=random.uniform(0.1, 0.25),
            vfe=random.uniform(0.5, 2.5)
        )

    def update_metrics(self, metrics: MetricsSnapshot):
        """Update current metrics (called from training loop)."""
        self.current_metrics = metrics

    async def serve(self):
        """Start WebSocket server (async)."""
        self.logger.info(f"Starting VizStream on ws://{self.host}:{self.port}")

        # Start emitter loop
        emitter_task = asyncio.create_task(self._emitter_loop())

        # Start WebSocket server
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()  # Run forever

    def run(self):
        """Start WebSocket server (blocking)."""
        try:
            asyncio.run(self.serve())
        except KeyboardInterrupt:
            self.logger.info("VizStream stopped by user")


if __name__ == "__main__":
    # Demo mode
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("Starting VizStream in demo mode on ws://localhost:8765")
    print("Connect dashboard or use: websocat ws://localhost:8765")
    print("Press Ctrl+C to stop\n")

    stream = VizStream(demo_mode=True)
    stream.run()
