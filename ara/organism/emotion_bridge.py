#!/usr/bin/env python3
"""
Emotion Bridge - UART/Serial to WebSocket Daemon
=================================================

Central daemon that:
1. Reads EMO/HPV/EMO_STORE/EMO_RECALL/EMO_DREAM lines from UART/serial
2. Parses JSON payloads
3. Broadcasts to WebSocket clients (cockpit, dashboards)
4. Triggers emotional TTS

Wire protocol (from FPGA/HPS):
    EMO {"emotion":"RAGE","strength":0.91,"valence":-0.78,"arousal":0.88,
         "dominance":0.73,"sparsity":0.14,"homeo_dev":0.22,
         "tags":["route_flap","stressed"]}

    HPV {"id":134,"anomaly_score":0.87,"class":"ANOMALY","tag":"route_flap"}

    EMO_STORE {"index":134,"emotion":"RAGE","strength":0.91}
    EMO_RECALL {"index":134,"emotion":"RAGE","sim":0.94,"strength":0.88}
    EMO_DREAM {"index":42,"sim":0.93,"strength":0.80}

Usage:
    python emotion_bridge.py --port /dev/ttyUSB0 --baud 115200
    python emotion_bridge.py --stdin  # For testing with stdin
"""

from __future__ import annotations
import argparse
import asyncio
import json
import sys
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Any, Callable

logger = logging.getLogger(__name__)

# Optional imports
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    serial = None

try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    websockets = None


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class EmotionState:
    """Emotion state from FPGA/organism."""
    emotion: str
    strength: float
    valence: float
    arousal: float
    dominance: float
    sparsity: float
    homeo_dev: float
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class HPVState:
    """Hypervector classification from FPGA."""
    id: int
    anomaly_score: float
    classification: str
    tag: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MemoryEvent:
    """Memory store/recall/dream event."""
    event_type: str  # "store", "recall", "dream"
    index: int
    emotion: str = ""
    similarity: float = 0.0
    strength: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Serial Reader
# ============================================================================

class SerialReader:
    """Reads lines from serial port."""

    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self.ser: Optional[Any] = None

    def open(self) -> bool:
        """Open serial connection."""
        if not HAS_SERIAL:
            logger.error("pyserial not installed")
            return False

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=0.2,
                write_timeout=0.2,
            )
            logger.info(f"Serial opened: {self.port} @ {self.baud}")
            return True
        except Exception as e:
            logger.error(f"Failed to open serial: {e}")
            return False

    def close(self) -> None:
        """Close serial connection."""
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    async def read_loop(self, callback: Callable[[str], None]) -> None:
        """Read lines and call callback."""
        if self.ser is None:
            return

        buffer = b""
        while True:
            try:
                chunk = self.ser.read(256)
                if not chunk:
                    await asyncio.sleep(0.01)
                    continue

                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_str = line.decode("utf-8", errors="ignore").strip()
                    if line_str:
                        await callback(line_str)

            except Exception as e:
                logger.error(f"Serial read error: {e}")
                await asyncio.sleep(1.0)


class StdinReader:
    """Reads lines from stdin (for testing)."""

    async def read_loop(self, callback: Callable[[str], None]) -> None:
        """Read lines from stdin."""
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            line = await reader.readline()
            if not line:
                break
            line_str = line.decode("utf-8", errors="ignore").strip()
            if line_str:
                await callback(line_str)


# ============================================================================
# WebSocket Broadcaster
# ============================================================================

class WebSocketBroadcaster:
    """Broadcasts messages to connected WebSocket clients."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: set = set()
        self._server = None

    async def start(self) -> None:
        """Start WebSocket server."""
        if not HAS_WEBSOCKETS:
            logger.warning("websockets not installed, broadcasting disabled")
            return

        self._server = await websockets.serve(
            self._handler, self.host, self.port
        )
        logger.info(f"WebSocket server started: ws://{self.host}:{self.port}")

    async def stop(self) -> None:
        """Stop WebSocket server."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handler(self, websocket, path) -> None:
        """Handle WebSocket connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                # We don't expect messages from clients, but handle gracefully
                pass
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected: {websocket.remote_address}")

    async def broadcast(self, msg_type: str, data: dict) -> None:
        """Broadcast message to all clients."""
        if not self.clients:
            return

        message = json.dumps({"type": msg_type, "data": data})
        dead = []

        for ws in list(self.clients):
            try:
                await ws.send(message)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.clients.discard(ws)


# ============================================================================
# Line Parser
# ============================================================================

def parse_line(line: str) -> Optional[tuple]:
    """
    Parse a protocol line.

    Returns:
        (message_type, parsed_data) or None if parsing failed
    """
    try:
        # EMO {...}
        if line.startswith("EMO "):
            payload = line[4:].strip()
            obj = json.loads(payload)
            state = EmotionState(
                emotion=obj.get("emotion", "UNKNOWN"),
                strength=float(obj.get("strength", 0.0)),
                valence=float(obj.get("valence", 0.0)),
                arousal=float(obj.get("arousal", 0.0)),
                dominance=float(obj.get("dominance", 0.0)),
                sparsity=float(obj.get("sparsity", 0.0)),
                homeo_dev=float(obj.get("homeo_dev", 0.0)),
                tags=list(obj.get("tags", [])),
            )
            return ("emotion", state)

        # HPV {...}
        if line.startswith("HPV "):
            payload = line[4:].strip()
            obj = json.loads(payload)
            state = HPVState(
                id=int(obj.get("id", 0)),
                anomaly_score=float(obj.get("anomaly_score", 0.0)),
                classification=obj.get("class", "UNKNOWN"),
                tag=obj.get("tag", ""),
            )
            return ("hpv", state)

        # EMO_STORE {...}
        if line.startswith("EMO_STORE "):
            payload = line[10:].strip()
            obj = json.loads(payload)
            event = MemoryEvent(
                event_type="store",
                index=int(obj.get("index", -1)),
                emotion=obj.get("emotion", ""),
                strength=float(obj.get("strength", 0.0)),
            )
            return ("memory", event)

        # EMO_RECALL {...}
        if line.startswith("EMO_RECALL "):
            payload = line[11:].strip()
            obj = json.loads(payload)
            event = MemoryEvent(
                event_type="recall",
                index=int(obj.get("index", -1)),
                emotion=obj.get("emotion", ""),
                similarity=float(obj.get("sim", 0.0)),
                strength=float(obj.get("strength", 0.0)),
            )
            return ("memory", event)

        # EMO_DREAM {...}
        if line.startswith("EMO_DREAM "):
            payload = line[10:].strip()
            obj = json.loads(payload)
            event = MemoryEvent(
                event_type="dream",
                index=int(obj.get("index", -1)),
                similarity=float(obj.get("sim", 0.0)),
                strength=float(obj.get("strength", 0.0)),
            )
            return ("memory", event)

    except Exception as e:
        logger.warning(f"Parse error: {e} for line: {line[:100]}")

    return None


# ============================================================================
# Emotion Bridge
# ============================================================================

class EmotionBridge:
    """
    Main bridge daemon.

    Connects serial input to WebSocket output and TTS.
    """

    def __init__(
        self,
        serial_port: str = "/dev/ttyUSB0",
        serial_baud: int = 115200,
        ws_host: str = "127.0.0.1",
        ws_port: int = 8765,
        use_stdin: bool = False,
        tts_callback: Optional[Callable] = None,
    ):
        self.use_stdin = use_stdin

        if use_stdin:
            self.reader = StdinReader()
        else:
            self.reader = SerialReader(serial_port, serial_baud)

        self.broadcaster = WebSocketBroadcaster(ws_host, ws_port)
        self.tts_callback = tts_callback

        self._running = False
        self._last_emotion: Optional[EmotionState] = None

    async def start(self) -> None:
        """Start the bridge."""
        # Open serial if needed
        if not self.use_stdin:
            if not self.reader.open():
                logger.error("Failed to open serial port")
                return

        # Start WebSocket server
        await self.broadcaster.start()

        self._running = True
        logger.info("Emotion bridge started")

    async def stop(self) -> None:
        """Stop the bridge."""
        self._running = False

        if hasattr(self.reader, 'close'):
            self.reader.close()

        await self.broadcaster.stop()
        logger.info("Emotion bridge stopped")

    async def _handle_line(self, line: str) -> None:
        """Handle a parsed line."""
        result = parse_line(line)
        if result is None:
            return

        msg_type, data = result

        if msg_type == "emotion":
            state: EmotionState = data
            self._last_emotion = state

            # Broadcast to cockpit
            await self.broadcaster.broadcast("emotion", state.to_dict())

            # Trigger TTS
            if self.tts_callback is not None:
                asyncio.create_task(self.tts_callback(state))

            logger.debug(f"Emotion: {state.emotion} (s={state.strength:.2f})")

        elif msg_type == "hpv":
            hpv: HPVState = data
            await self.broadcaster.broadcast("hpv", hpv.to_dict())
            logger.debug(f"HPV: {hpv.classification} (score={hpv.anomaly_score:.2f})")

        elif msg_type == "memory":
            event: MemoryEvent = data
            await self.broadcaster.broadcast("memory", event.to_dict())
            logger.debug(f"Memory: {event.event_type} idx={event.index}")

    async def run(self) -> None:
        """Run the bridge loop."""
        await self.start()

        try:
            await self.reader.read_loop(self._handle_line)
        except KeyboardInterrupt:
            pass
        finally:
            await self.stop()


# ============================================================================
# CLI
# ============================================================================

async def main_async(args):
    """Async main entry point."""
    # Optional: import TTS callback
    tts_callback = None
    if args.tts:
        try:
            from ara.organism.emotion_tts import speak_from_state
            tts_callback = speak_from_state
            logger.info("TTS enabled")
        except ImportError:
            logger.warning("emotion_tts not available, TTS disabled")

    bridge = EmotionBridge(
        serial_port=args.port,
        serial_baud=args.baud,
        ws_host=args.ws_host,
        ws_port=args.ws_port,
        use_stdin=args.stdin,
        tts_callback=tts_callback,
    )

    await bridge.run()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Emotion Bridge Daemon")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--ws-host", default="127.0.0.1", help="WebSocket host")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--tts", action="store_true", help="Enable TTS")
    parser.add_argument("--debug", action="store_true", help="Debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nBridge stopped.")


def demo():
    """Run demo with synthetic data."""
    print("=" * 60)
    print("Emotion Bridge Demo")
    print("=" * 60)

    # Test line parsing
    test_lines = [
        'EMO {"emotion":"calm","strength":0.6,"valence":0.5,"arousal":-0.3,'
        '"dominance":0.4,"sparsity":0.85,"homeo_dev":0.02,"tags":["stable"]}',
        'HPV {"id":42,"anomaly_score":0.15,"class":"NORMAL","tag":"baseline"}',
        'EMO_STORE {"index":100,"emotion":"joy","strength":0.8}',
        'EMO_RECALL {"index":100,"emotion":"joy","sim":0.92,"strength":0.8}',
        'EMO_DREAM {"index":50,"sim":0.88,"strength":0.7}',
    ]

    print("\n--- Parsing test lines ---\n")
    for line in test_lines:
        result = parse_line(line)
        if result:
            msg_type, data = result
            print(f"Type: {msg_type}")
            print(f"Data: {data.to_dict()}")
            print()

    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        main()
