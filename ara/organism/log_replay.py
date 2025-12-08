#!/usr/bin/env python3
"""
Log Replay Tool - Synthetic EMO Event Generator
================================================

Generates realistic EMO/HPV/memory events for testing the organism pipeline
without FPGA hardware.

Modes:
1. Random walk: Generates drifting VAD states
2. Scenario replay: Runs predefined emotional narratives
3. File replay: Replays captured log files

Output goes to stdout (pipe to emotion_bridge --stdin) or directly to WebSocket.

Usage:
    # Pipe to emotion_bridge
    python log_replay.py --mode random | python emotion_bridge.py --stdin --tts

    # Direct WebSocket output
    python log_replay.py --mode scenario --ws ws://127.0.0.1:8765

    # Replay a log file
    python log_replay.py --mode file --input captured.log
"""

from __future__ import annotations
import argparse
import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Iterator
from enum import Enum
import math

# Optional WebSocket
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False


# ============================================================================
# Event Types
# ============================================================================

@dataclass
class EmotionEvent:
    """Full emotion state event."""
    emotion: str
    strength: float
    valence: float
    arousal: float
    dominance: float
    sparsity: float
    homeo_dev: float
    tags: List[str]

    def to_line(self) -> str:
        return f"EMO {json.dumps(asdict(self))}"


@dataclass
class HPVEvent:
    """HPV classification event."""
    id: int
    anomaly_score: float
    cls: str
    tag: str

    def to_line(self) -> str:
        return f'HPV {{"id":{self.id},"anomaly_score":{self.anomaly_score:.3f},"class":"{self.cls}","tag":"{self.tag}"}}'


@dataclass
class MemoryStoreEvent:
    """Memory store event."""
    index: int
    emotion: str
    strength: float

    def to_line(self) -> str:
        return f'EMO_STORE {{"index":{self.index},"emotion":"{self.emotion}","strength":{self.strength:.3f}}}'


@dataclass
class MemoryRecallEvent:
    """Memory recall event."""
    index: int
    emotion: str
    sim: float
    strength: float

    def to_line(self) -> str:
        return f'EMO_RECALL {{"index":{self.index},"emotion":"{self.emotion}","sim":{self.sim:.3f},"strength":{self.strength:.3f}}}'


@dataclass
class MemoryDreamEvent:
    """Memory dream event."""
    index: int
    sim: float
    strength: float

    def to_line(self) -> str:
        return f'EMO_DREAM {{"index":{self.index},"sim":{self.sim:.3f},"strength":{self.strength:.3f}}}'


# ============================================================================
# Emotion Mapping
# ============================================================================

# VAD coordinates for emotions (valence, arousal, dominance)
EMOTION_VAD = {
    "joy": (0.9, 0.6, 0.5),
    "excitement": (0.7, 0.9, 0.4),
    "serenity": (0.8, -0.4, 0.6),
    "calm": (0.5, -0.6, 0.4),
    "trust": (0.6, 0.2, 0.7),
    "fear": (-0.7, 0.8, -0.7),
    "anger": (-0.6, 0.7, 0.5),
    "rage": (-0.8, 0.95, 0.6),
    "anxiety": (-0.5, 0.7, -0.4),
    "sadness": (-0.6, -0.4, -0.3),
    "boredom": (-0.3, -0.8, -0.2),
    "contempt": (0.0, 0.2, 0.8),
    "vigilance": (0.1, 0.8, 0.6),
    "surprise": (0.1, 0.9, -0.3),
    "overwhelmed": (-0.6, 0.9, -0.8),
    "neutral": (0.0, 0.0, 0.0),
}

TAG_POOL = [
    "baseline", "stable", "route_flap", "route_flap_burst",
    "anomaly_burst", "memory_recall", "novel_pattern",
    "fabric_stressed", "recovering", "high_load",
    "low_activity", "burst_detected", "threshold_exceeded",
]


def vad_to_emotion(v: float, a: float, d: float) -> str:
    """Find closest emotion to VAD coordinates."""
    best_emotion = "neutral"
    best_dist = float("inf")

    for emotion, (ev, ea, ed) in EMOTION_VAD.items():
        dist = math.sqrt((v - ev)**2 + (a - ea)**2 + (d - ed)**2)
        if dist < best_dist:
            best_dist = dist
            best_emotion = emotion

    return best_emotion


# ============================================================================
# Generators
# ============================================================================

class RandomWalkGenerator:
    """Generate random-walk VAD states."""

    def __init__(
        self,
        interval: float = 0.5,
        drift_rate: float = 0.1,
        memory_prob: float = 0.1,
        dream_prob: float = 0.02,
    ):
        self.interval = interval
        self.drift_rate = drift_rate
        self.memory_prob = memory_prob
        self.dream_prob = dream_prob

        # Current VAD state
        self.valence = 0.0
        self.arousal = 0.0
        self.dominance = 0.0
        self.sparsity = 0.8
        self.homeo_dev = 0.05

        # Memory state
        self.memory_index = 0
        self.stored_emotions: List[str] = []

    def _drift(self, value: float, target: float = 0.0) -> float:
        """Drift value with random walk + mean reversion."""
        noise = random.gauss(0, self.drift_rate)
        reversion = (target - value) * 0.05
        new_val = value + noise + reversion
        return max(-1.0, min(1.0, new_val))

    def generate(self) -> Iterator[str]:
        """Generate events forever."""
        step = 0

        while True:
            # Drift VAD
            self.valence = self._drift(self.valence)
            self.arousal = self._drift(self.arousal)
            self.dominance = self._drift(self.dominance)
            self.sparsity = self._drift(self.sparsity, 0.8)
            self.homeo_dev = abs(self._drift(self.homeo_dev, 0.05))

            # Determine emotion
            emotion = vad_to_emotion(self.valence, self.arousal, self.dominance)
            strength = 0.3 + random.random() * 0.6

            # Pick tags
            num_tags = random.randint(0, 3)
            tags = random.sample(TAG_POOL, num_tags)

            # Emit emotion event
            event = EmotionEvent(
                emotion=emotion,
                strength=strength,
                valence=self.valence,
                arousal=self.arousal,
                dominance=self.dominance,
                sparsity=self.sparsity,
                homeo_dev=self.homeo_dev,
                tags=tags,
            )
            yield event.to_line()

            # Maybe emit memory store (on strong emotions)
            if strength > 0.7 and random.random() < self.memory_prob:
                self.stored_emotions.append(emotion)
                store_event = MemoryStoreEvent(
                    index=self.memory_index,
                    emotion=emotion,
                    strength=strength,
                )
                self.memory_index += 1
                yield store_event.to_line()

            # Maybe emit memory recall
            if self.stored_emotions and random.random() < self.memory_prob:
                idx = random.randint(0, len(self.stored_emotions) - 1)
                recall_event = MemoryRecallEvent(
                    index=idx,
                    emotion=self.stored_emotions[idx],
                    sim=0.65 + random.random() * 0.3,
                    strength=0.5 + random.random() * 0.4,
                )
                yield recall_event.to_line()

            # Maybe emit dream (less frequent)
            if self.stored_emotions and random.random() < self.dream_prob:
                idx = random.randint(0, len(self.stored_emotions) - 1)
                dream_event = MemoryDreamEvent(
                    index=idx,
                    sim=0.7 + random.random() * 0.25,
                    strength=0.6 + random.random() * 0.3,
                )
                yield dream_event.to_line()

            # Occasional anomaly classification
            if step % 10 == 0:
                is_anomaly = strength > 0.8 and self.valence < -0.3
                hpv_event = HPVEvent(
                    id=step,
                    anomaly_score=0.8 if is_anomaly else 0.2,
                    cls="ANOMALY" if is_anomaly else "NORMAL",
                    tag=tags[0] if tags else "baseline",
                )
                yield hpv_event.to_line()

            step += 1
            time.sleep(self.interval)


class ScenarioGenerator:
    """Generate predefined emotional narratives."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval

    def _scenario_calm_to_rage(self) -> List[str]:
        """Calm operation escalating to rage."""
        events = []

        # Start calm
        for i in range(5):
            v = 0.5 - i * 0.1
            a = -0.3 + i * 0.15
            d = 0.4 - i * 0.05
            emotion = vad_to_emotion(v, a, d)
            events.append(EmotionEvent(
                emotion=emotion,
                strength=0.4 + i * 0.1,
                valence=v, arousal=a, dominance=d,
                sparsity=0.8 - i * 0.05,
                homeo_dev=0.02 + i * 0.03,
                tags=["stable"] if i < 2 else ["route_flap"],
            ).to_line())

        # Escalate to rage
        for i in range(5):
            v = -0.3 - i * 0.1
            a = 0.5 + i * 0.1
            d = 0.3 + i * 0.08
            emotion = vad_to_emotion(v, a, d)
            events.append(EmotionEvent(
                emotion=emotion,
                strength=0.7 + i * 0.06,
                valence=v, arousal=a, dominance=d,
                sparsity=0.5 - i * 0.05,
                homeo_dev=0.15 + i * 0.05,
                tags=["route_flap_burst", "fabric_stressed"],
            ).to_line())

        # Store the rage
        events.append(MemoryStoreEvent(
            index=0, emotion="rage", strength=0.95
        ).to_line())

        # Cool down
        for i in range(5):
            v = -0.6 + i * 0.2
            a = 0.9 - i * 0.2
            d = 0.7 - i * 0.1
            emotion = vad_to_emotion(v, a, d)
            events.append(EmotionEvent(
                emotion=emotion,
                strength=0.8 - i * 0.1,
                valence=v, arousal=a, dominance=d,
                sparsity=0.4 + i * 0.1,
                homeo_dev=0.3 - i * 0.05,
                tags=["recovering"],
            ).to_line())

        return events

    def _scenario_memory_flashback(self) -> List[str]:
        """Calm state triggering a memory recall."""
        events = []

        # Normal operation
        for i in range(3):
            events.append(EmotionEvent(
                emotion="calm",
                strength=0.5,
                valence=0.4, arousal=-0.3, dominance=0.5,
                sparsity=0.85,
                homeo_dev=0.02,
                tags=["stable"],
            ).to_line())

        # Memory recall triggered
        events.append(MemoryRecallEvent(
            index=0, emotion="rage", sim=0.92, strength=0.88
        ).to_line())

        # Emotional response to recall
        for i in range(3):
            events.append(EmotionEvent(
                emotion="anxiety",
                strength=0.6 + i * 0.1,
                valence=-0.3 - i * 0.1,
                arousal=0.4 + i * 0.1,
                dominance=-0.2,
                sparsity=0.7,
                homeo_dev=0.1,
                tags=["memory_recall", "route_flap"],
            ).to_line())

        # Recovery
        for i in range(3):
            events.append(EmotionEvent(
                emotion="vigilance",
                strength=0.5 - i * 0.1,
                valence=0.0 + i * 0.15,
                arousal=0.5 - i * 0.15,
                dominance=0.3 + i * 0.1,
                sparsity=0.8,
                homeo_dev=0.05,
                tags=["recovering"],
            ).to_line())

        return events

    def _scenario_dream_cycle(self) -> List[str]:
        """Low-activity dreaming state."""
        events = []

        # Quiet baseline
        for i in range(3):
            events.append(EmotionEvent(
                emotion="boredom",
                strength=0.3,
                valence=-0.2, arousal=-0.6, dominance=-0.1,
                sparsity=0.95,
                homeo_dev=0.01,
                tags=["low_activity"],
            ).to_line())

        # Dreams start
        for i in range(4):
            events.append(MemoryDreamEvent(
                index=i * 10,
                sim=0.75 + random.random() * 0.2,
                strength=0.6 + random.random() * 0.3,
            ).to_line())

            events.append(EmotionEvent(
                emotion="surprise" if i % 2 == 0 else "calm",
                strength=0.4,
                valence=0.0, arousal=-0.4, dominance=0.0,
                sparsity=0.9,
                homeo_dev=0.02,
                tags=["dreaming"],
            ).to_line())

        return events

    def generate(self) -> Iterator[str]:
        """Generate scenario events."""
        scenarios = [
            ("Calm to Rage", self._scenario_calm_to_rage()),
            ("Memory Flashback", self._scenario_memory_flashback()),
            ("Dream Cycle", self._scenario_dream_cycle()),
        ]

        for name, events in scenarios:
            print(f"# === Scenario: {name} ===", file=sys.stderr)
            for event in events:
                yield event
                time.sleep(self.interval)

            # Pause between scenarios
            time.sleep(2.0)


class FileReplayGenerator:
    """Replay events from a log file."""

    def __init__(self, filepath: str, interval: float = 0.5):
        self.filepath = filepath
        self.interval = interval

    def generate(self) -> Iterator[str]:
        """Replay file contents."""
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    yield line
                    time.sleep(self.interval)


# ============================================================================
# Output Modes
# ============================================================================

def output_stdout(events: Iterator[str]) -> None:
    """Output to stdout."""
    for event in events:
        print(event)
        sys.stdout.flush()


async def output_websocket(events: Iterator[str], url: str) -> None:
    """Output directly to WebSocket."""
    if not HAS_WEBSOCKETS:
        print("websockets not installed", file=sys.stderr)
        return

    async with websockets.connect(url) as ws:
        print(f"Connected to {url}", file=sys.stderr)

        for event in events:
            # Parse and reformat for WebSocket
            if event.startswith("EMO "):
                payload = json.loads(event[4:])
                msg = json.dumps({"type": "emotion", "data": payload})
            elif event.startswith("HPV "):
                payload = json.loads(event[4:])
                msg = json.dumps({"type": "hpv", "data": payload})
            elif event.startswith("EMO_STORE "):
                payload = json.loads(event[10:])
                payload["event_type"] = "store"
                msg = json.dumps({"type": "memory", "data": payload})
            elif event.startswith("EMO_RECALL "):
                payload = json.loads(event[11:])
                payload["event_type"] = "recall"
                msg = json.dumps({"type": "memory", "data": payload})
            elif event.startswith("EMO_DREAM "):
                payload = json.loads(event[10:])
                payload["event_type"] = "dream"
                msg = json.dumps({"type": "memory", "data": payload})
            else:
                continue

            await ws.send(msg)
            print(f"Sent: {event[:60]}...", file=sys.stderr)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Log Replay Tool")
    parser.add_argument("--mode", choices=["random", "scenario", "file"],
                        default="random", help="Generation mode")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Interval between events (seconds)")
    parser.add_argument("--input", help="Input file for file mode")
    parser.add_argument("--ws", help="WebSocket URL for direct output")
    parser.add_argument("--count", type=int, default=0,
                        help="Max events (0 = infinite for random)")

    args = parser.parse_args()

    # Create generator
    if args.mode == "random":
        gen = RandomWalkGenerator(interval=args.interval)
        events = gen.generate()
    elif args.mode == "scenario":
        gen = ScenarioGenerator(interval=args.interval)
        events = gen.generate()
    elif args.mode == "file":
        if not args.input:
            print("--input required for file mode", file=sys.stderr)
            sys.exit(1)
        gen = FileReplayGenerator(args.input, interval=args.interval)
        events = gen.generate()

    # Limit count if specified
    if args.count > 0:
        def limited(gen, n):
            for i, item in enumerate(gen):
                if i >= n:
                    break
                yield item
        events = limited(events, args.count)

    # Output
    try:
        if args.ws:
            asyncio.run(output_websocket(events, args.ws))
        else:
            output_stdout(events)
    except KeyboardInterrupt:
        print("\nReplay stopped.", file=sys.stderr)


if __name__ == "__main__":
    main()
