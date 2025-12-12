"""
Organism Runtime - Hardware-Ready Harness
==========================================

Iteration 46: First shipping organism that can run on hardware.

This runtime:
1. Loads trained model parameters (or uses defaults)
2. Runs SNN inference with HV projection
3. Applies reflexive probe for concept tagging
4. Detects emotions via VAD mind
5. Emits state over UART for voice synthesis
6. Injects feedback into next timestep

Hardware split:
- FPGA: CorrSpike-HDC correlation + basic LIF (future)
- Host (this): Full VAD affect + reflexive probe + voice
"""

from __future__ import annotations
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Callable, Any
from pathlib import Path
import numpy as np

from ara.organism.vad_mind import VADEmotionalMind, VADState, EmotionArchetype
from ara.organism.reflexive_probe import TinyReflexiveProbe, ConceptMatch

# Voice bridge (optional)
try:
    from ara.organism.voice_bridge import VoiceBridge, get_voice_bridge
    VOICE_BRIDGE_AVAILABLE = True
except ImportError:
    VOICE_BRIDGE_AVAILABLE = False
    VoiceBridge = None

# Optional: PyTorch for full SNN
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

# Optional: Serial for UART
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    serial = None

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class OrganismConfig:
    """Configuration for the organism runtime."""
    # Network dimensions
    num_inputs: int = 784
    num_hidden: int = 256
    num_outputs: int = 2
    num_steps: int = 10

    # Hypervector dimensions
    hv_dim: int = 1024
    status_dim: int = 64

    # LIF parameters
    beta_hidden: float = 0.9
    beta_output: float = 0.9
    v_thr_hidden: float = 1.0
    v_thr_output: float = 1.0
    lambda_inh: float = 0.1
    gamma_inh: float = 0.05

    # Homeostasis
    target_hidden_rate: float = 0.15
    target_output_rate: float = 0.10

    # UART
    uart_port: str = "/dev/ttyUSB0"
    uart_baud: int = 115200
    uart_enabled: bool = False

    # Feedback
    feedback_strength: float = 0.3

    # Voice synthesis
    voice_enabled: bool = False
    voice_min_interval: float = 2.0

    @classmethod
    def from_json(cls, path: str) -> 'OrganismConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================================
# NumPy-based SNN (no PyTorch required)
# ============================================================================

class NumpyLIFNet:
    """
    Pure NumPy LIF network for runtime inference.

    This is a lightweight implementation that doesn't require PyTorch.
    Can be used as fallback or for testing.
    """

    def __init__(self, cfg: OrganismConfig):
        self.cfg = cfg
        rng = np.random.default_rng(42)

        # Initialize weights (normally would load from file)
        self.fc1_weight = rng.standard_normal((cfg.num_hidden, cfg.num_inputs)).astype(np.float32) * 0.1
        self.fc2_weight = rng.standard_normal((cfg.num_outputs, cfg.num_hidden)).astype(np.float32) * 0.1

        # HV projections
        self.hv_proj = rng.standard_normal((cfg.hv_dim, cfg.num_hidden)).astype(np.float32) * 0.1
        self.status_proj = rng.standard_normal((cfg.status_dim, 4)).astype(np.float32) * 0.1

        # Thresholds
        self.v_thr1 = np.ones(cfg.num_hidden, dtype=np.float32) * cfg.v_thr_hidden
        self.v_thr2 = np.ones(cfg.num_outputs, dtype=np.float32) * cfg.v_thr_output

        # State
        self.mem1: Optional[np.ndarray] = None
        self.mem2: Optional[np.ndarray] = None
        self.inh1: Optional[np.ndarray] = None
        self.inh2: Optional[np.ndarray] = None

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset membrane and inhibition state."""
        cfg = self.cfg
        self.mem1 = np.zeros((batch_size, cfg.num_hidden), dtype=np.float32)
        self.mem2 = np.zeros((batch_size, cfg.num_outputs), dtype=np.float32)
        self.inh1 = np.zeros((batch_size, cfg.num_hidden), dtype=np.float32)
        self.inh2 = np.zeros((batch_size, cfg.num_outputs), dtype=np.float32)

    def forward(
        self,
        x: np.ndarray,
        feedback_hv: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """
        Forward pass through the network.

        Args:
            x: [B, num_inputs] input telemetry
            feedback_hv: [B, hv_dim+status_dim] optional feedback

        Returns:
            hv_full: [B, hv_dim+status_dim] output hypervector
            logits: [B, num_outputs] classification logits
            hidden_rate: average hidden spike rate
            homeo_dev: homeostasis deviation
            sparsity_ratio: output sparsity
        """
        cfg = self.cfg
        B = x.shape[0]

        if self.mem1 is None or self.mem1.shape[0] != B:
            self.reset_state(B)

        # Inject feedback
        if feedback_hv is not None:
            fb_portion = feedback_hv[:, :cfg.num_inputs]
            x = x + cfg.feedback_strength * fb_portion

        # Accumulate spikes
        hidden_spikes = np.zeros((cfg.num_steps, B, cfg.num_hidden), dtype=np.float32)
        output_spikes = np.zeros((cfg.num_steps, B, cfg.num_outputs), dtype=np.float32)

        for t in range(cfg.num_steps):
            # Hidden layer
            cur1 = x @ self.fc1_weight.T
            self.mem1 = cfg.beta_hidden * self.mem1 + cur1

            # Inhibition
            self.inh1 = cfg.lambda_inh * self.inh1
            self.mem1 = self.mem1 - cfg.gamma_inh * self.inh1

            # Spike
            spk1 = (self.mem1 >= self.v_thr1).astype(np.float32)
            self.mem1 = np.where(spk1 > 0, 0.0, self.mem1)
            self.inh1 = self.inh1 + spk1
            hidden_spikes[t] = spk1

            # Output layer
            cur2 = spk1 @ self.fc2_weight.T
            self.mem2 = cfg.beta_output * self.mem2 + cur2

            self.inh2 = cfg.lambda_inh * self.inh2
            self.mem2 = self.mem2 - cfg.gamma_inh * self.inh2

            spk2 = (self.mem2 >= self.v_thr2).astype(np.float32)
            self.mem2 = np.where(spk2 > 0, 0.0, self.mem2)
            self.inh2 = self.inh2 + spk2
            output_spikes[t] = spk2

        # Compute rates
        hidden_rate = hidden_spikes.mean()
        output_rate = output_spikes.mean()
        sparsity_ratio = 1.0 - output_rate

        # Homeostasis deviation
        homeo_dev = abs(hidden_rate - cfg.target_hidden_rate)

        # HV projection
        hidden_avg = hidden_spikes.mean(axis=0)  # [B, H]
        hv_raw = hidden_avg @ self.hv_proj.T  # [B, hv_dim]
        hv_bin = np.sign(hv_raw)
        hv_bin[hv_bin == 0] = 1.0

        # Status HV
        stats = np.array([[hidden_rate, output_rate, homeo_dev, sparsity_ratio]])
        status_raw = stats @ self.status_proj.T
        status_hv = np.sign(status_raw)
        status_hv[status_hv == 0] = 1.0
        status_hv = np.tile(status_hv, (B, 1))

        # Combine
        hv_full = np.concatenate([hv_bin, status_hv], axis=1)

        # Logits from spike counts
        logits = output_spikes.sum(axis=0)

        return hv_full, logits, float(hidden_rate), float(homeo_dev), float(sparsity_ratio)


# ============================================================================
# UART Communication
# ============================================================================

class UARTEmitter:
    """Emit organism state over UART."""

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200):
        self.port = port
        self.baud = baud
        self.ser: Optional[Any] = None

    def open(self) -> bool:
        """Open UART connection."""
        if not HAS_SERIAL:
            logger.warning("pyserial not installed, UART disabled")
            return False

        try:
            self.ser = serial.Serial(self.port, baudrate=self.baud, timeout=0.1)
            logger.info(f"UART opened: {self.port} @ {self.baud}")
            return True
        except Exception as e:
            logger.warning(f"UART unavailable: {e}")
            return False

    def close(self) -> None:
        """Close UART connection."""
        if self.ser is not None:
            self.ser.close()
            self.ser = None

    def emit_emotion(self, state: VADState) -> None:
        """Emit emotional state over UART."""
        if self.ser is None:
            return

        # Protocol: EMO:<archetype>:<strength>:<dominance>:<valence>:<arousal>\n
        line = (
            f"EMO:{state.archetype.value}:"
            f"{state.strength:.2f}:{state.dominance:.2f}:"
            f"{state.valence:.2f}:{state.arousal:.2f}\n"
        )
        self.ser.write(line.encode("utf-8"))

    def emit_concept(self, match: ConceptMatch) -> None:
        """Emit concept match over UART."""
        if self.ser is None:
            return

        # Protocol: TAG:<tag>:<similarity>:<novel>\n
        line = f"TAG:{match.tag}:{match.similarity:.2f}:{1 if match.is_novel else 0}\n"
        self.ser.write(line.encode("utf-8"))

    def emit_classification(self, label: str, confidence: float) -> None:
        """Emit classification result over UART."""
        if self.ser is None:
            return

        line = f"CLS:{label}:{confidence:.2f}\n"
        self.ser.write(line.encode("utf-8"))


# ============================================================================
# Main Organism Runtime
# ============================================================================

@dataclass
class OrganismState:
    """Current state of the organism."""
    step: int = 0
    hidden_rate: float = 0.0
    homeo_dev: float = 0.0
    sparsity_ratio: float = 0.0
    classification: str = "UNKNOWN"
    confidence: float = 0.0
    emotion: VADState = field(default_factory=lambda: VADState(0, 0, 0))
    concept_tag: str = ""
    is_novel: bool = False


class OrganismRuntime:
    """
    Main organism runtime harness.

    Usage:
        runtime = OrganismRuntime(config)
        runtime.start()

        # In your main loop:
        while True:
            telemetry = get_telemetry()
            state = runtime.step(telemetry)
            print(f"Emotion: {state.emotion.archetype.value}")
    """

    def __init__(self, cfg: Optional[OrganismConfig] = None):
        self.cfg = cfg or OrganismConfig()

        # Initialize components
        self.net = NumpyLIFNet(self.cfg)
        self.probe = TinyReflexiveProbe(
            hv_dim=self.cfg.hv_dim,
            status_dim=self.cfg.status_dim,
            feedback_strength=self.cfg.feedback_strength,
        )
        self.mind = VADEmotionalMind(hv_dim=self.cfg.hv_dim + self.cfg.status_dim)
        self.uart = UARTEmitter(self.cfg.uart_port, self.cfg.uart_baud)

        # Voice bridge (optional)
        self.voice_bridge: Optional[VoiceBridge] = None
        if self.cfg.voice_enabled and VOICE_BRIDGE_AVAILABLE:
            from ara.organism.voice_bridge import VoiceBridgeConfig
            voice_cfg = VoiceBridgeConfig(
                min_speak_interval=self.cfg.voice_min_interval,
            )
            self.voice_bridge = VoiceBridge(organism=self, config=voice_cfg)

        # State
        self.state = OrganismState()
        self.feedback_hv: Optional[np.ndarray] = None
        self._running = False

    def start(self) -> None:
        """Initialize the organism."""
        self.net.reset_state()

        if self.cfg.uart_enabled:
            self.uart.open()

        # Start voice bridge if enabled
        if self.voice_bridge:
            self.voice_bridge.start()
            logger.info("Voice bridge started")

        self._running = True
        logger.info("Organism runtime started")

    def stop(self) -> None:
        """Stop the organism."""
        self._running = False

        # Stop voice bridge
        if self.voice_bridge:
            self.voice_bridge.stop()

        self.uart.close()
        logger.info("Organism runtime stopped")

    def step(self, telemetry: np.ndarray) -> OrganismState:
        """
        Process one timestep of telemetry.

        Args:
            telemetry: [1, num_inputs] or [num_inputs] input vector

        Returns:
            Current organism state
        """
        # Ensure batch dimension
        if telemetry.ndim == 1:
            telemetry = telemetry.reshape(1, -1)

        # Pad/truncate to expected size
        if telemetry.shape[1] != self.cfg.num_inputs:
            padded = np.zeros((1, self.cfg.num_inputs), dtype=np.float32)
            n = min(telemetry.shape[1], self.cfg.num_inputs)
            padded[0, :n] = telemetry[0, :n]
            telemetry = padded

        # Forward through SNN
        hv_full, logits, hidden_rate, homeo_dev, sparsity_ratio = self.net.forward(
            telemetry, self.feedback_hv
        )

        # Classification
        pred = int(logits[0].argmax())
        confidence = float(logits[0].max() / (logits[0].sum() + 1e-6))
        label = "NORMAL" if pred == 0 else "ANOMALY"

        # Compute inhibition level (for VAD)
        inhibition_level = float(np.linalg.norm(logits) / (logits.size + 1e-6))

        # Compute early_exit_gap (confidence margin)
        sorted_logits = np.sort(logits[0])[::-1]
        if len(sorted_logits) >= 2:
            early_exit_gap = float(sorted_logits[0] - sorted_logits[1])
        else:
            early_exit_gap = float(sorted_logits[0]) if len(sorted_logits) > 0 else 0.0

        # Emotional state
        emo_state = self.mind.detect(
            hidden_rate, homeo_dev, early_exit_gap, inhibition_level, sparsity_ratio
        )
        emo_hv = self.mind.get_emotion_hv(emo_state)

        # Combine HV with emotion
        combined_hv = hv_full + 0.1 * emo_hv.reshape(1, -1)

        # Reflexive probe
        feedback_batch, matches = self.probe.process_batch(combined_hv)
        self.feedback_hv = feedback_batch
        match = matches[0]

        # Update state
        self.state = OrganismState(
            step=self.state.step + 1,
            hidden_rate=hidden_rate,
            homeo_dev=homeo_dev,
            sparsity_ratio=sparsity_ratio,
            classification=label,
            confidence=confidence,
            emotion=emo_state,
            concept_tag=match.tag,
            is_novel=match.is_novel,
        )

        # UART emission
        if self.cfg.uart_enabled:
            self.uart.emit_emotion(emo_state)
            self.uart.emit_concept(match)
            self.uart.emit_classification(label, confidence)

        return self.state

    def get_stats(self) -> Dict:
        """Get organism statistics."""
        return {
            "step": self.state.step,
            "probe": self.probe.get_codebook_summary(),
            "emotion": self.state.emotion.to_dict(),
        }


# ============================================================================
# Telemetry Sources
# ============================================================================

def random_telemetry(num_inputs: int = 784) -> np.ndarray:
    """Generate random telemetry (for testing)."""
    return np.random.randn(1, num_inputs).astype(np.float32)


def system_telemetry(num_inputs: int = 784) -> np.ndarray:
    """
    Generate telemetry from system metrics.

    Currently a placeholder - wire to real psutil/nvidia-smi/etc.
    """
    try:
        import psutil
        cpu = psutil.cpu_percent() / 100.0
        mem = psutil.virtual_memory().percent / 100.0

        # Simple encoding: fill first few dims with metrics
        vec = np.zeros(num_inputs, dtype=np.float32)
        vec[0] = cpu
        vec[1] = mem
        vec[2] = np.random.randn()  # Placeholder for network latency
        vec[3] = np.random.randn()  # Placeholder for GPU load

        # Add noise to rest
        vec[4:] = np.random.randn(num_inputs - 4) * 0.1

        return vec.reshape(1, -1)
    except ImportError:
        return random_telemetry(num_inputs)


# ============================================================================
# Main Loop
# ============================================================================

def run_organism_loop(
    config: Optional[OrganismConfig] = None,
    telemetry_fn: Optional[Callable[[], np.ndarray]] = None,
    max_steps: int = 0,
    print_interval: int = 20,
) -> None:
    """
    Run the organism in a continuous loop.

    Args:
        config: Organism configuration
        telemetry_fn: Function that returns telemetry vector
        max_steps: Max steps (0 = infinite)
        print_interval: Steps between status prints
    """
    cfg = config or OrganismConfig()

    if telemetry_fn is None:
        telemetry_fn = lambda: random_telemetry(cfg.num_inputs)

    runtime = OrganismRuntime(cfg)
    runtime.start()

    print("=" * 60)
    print("Organism Runtime")
    print("=" * 60)
    print(f"Inputs: {cfg.num_inputs}, Hidden: {cfg.num_hidden}")
    print(f"HV dim: {cfg.hv_dim}, Status dim: {cfg.status_dim}")
    print(f"UART: {'enabled' if cfg.uart_enabled else 'disabled'}")
    print("=" * 60)

    try:
        step = 0
        while max_steps == 0 or step < max_steps:
            telemetry = telemetry_fn()
            state = runtime.step(telemetry)

            if step % print_interval == 0 or state.is_novel:
                print(f"\n[Step {state.step:5d}] {state.classification}")
                print(f"  Rate: {state.hidden_rate:.3f}, "
                      f"Dev: {state.homeo_dev:.3f}, "
                      f"Sparse: {state.sparsity_ratio:.3f}")
                print(f"  Emotion: {state.emotion.archetype.value} "
                      f"(V={state.emotion.valence:+.2f}, "
                      f"A={state.emotion.arousal:+.2f}, "
                      f"D={state.emotion.dominance:+.2f})")
                print(f"  Concept: {state.concept_tag}"
                      f" {'[NOVEL]' if state.is_novel else ''}")

            step += 1
            time.sleep(0.01)  # 100 Hz max

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        runtime.stop()
        stats = runtime.get_stats()
        print("\n" + "=" * 60)
        print("Final Statistics")
        print("=" * 60)
        print(f"Total steps: {stats['step']}")
        print(f"Concepts learned: {stats['probe']['num_concepts']}")
        print(f"Novelty rate: {stats['probe']['novelty_rate']:.2%}")
        print("=" * 60)


def demo():
    """Run a quick demo of the organism."""
    print("Running Organism Runtime Demo (100 steps)...")
    run_organism_loop(max_steps=100, print_interval=10)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
