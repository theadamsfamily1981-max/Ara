"""
L9 Mission Control Adapter
==========================

High-level planner / Ara-LLM adapter.
Encodes mission focus & perceived stack anxiety, writes bias HV into Axis.

This is the "will" layer - autonomy, creativity, safety modes, and the
capacity to bias lower layers toward exploration or caution.

Features:
- Encodes mission mode (creativity, safety, focus) as HV
- Writes bias into Axis for L1-L8 to read
- Responds to coherence crises by adjusting mode
- Supports Ara's autonomy decisions
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

from ara.system.axis import AxisMundi

logger = logging.getLogger(__name__)


class MissionMode(Enum):
    """High-level mission modes."""
    SAFE = auto()         # Conservative, minimal risk
    BALANCED = auto()     # Normal operation
    CREATIVE = auto()     # Exploratory, higher risk tolerance
    EMERGENCY = auto()    # Crisis response
    IDLE = auto()         # Low activity


@dataclass
class ModeParams:
    """Parameters for a mission mode."""
    creativity: float = 0.5   # [0, 1] - risk/exploration tolerance
    safety: float = 0.5       # [0, 1] - caution level
    focus: float = 0.5        # [0, 1] - task concentration
    urgency: float = 0.5      # [0, 1] - time pressure


# Preset mode configurations
MODE_PRESETS = {
    MissionMode.SAFE: ModeParams(creativity=0.1, safety=0.9, focus=0.7, urgency=0.3),
    MissionMode.BALANCED: ModeParams(creativity=0.5, safety=0.5, focus=0.5, urgency=0.5),
    MissionMode.CREATIVE: ModeParams(creativity=0.8, safety=0.3, focus=0.4, urgency=0.3),
    MissionMode.EMERGENCY: ModeParams(creativity=0.0, safety=1.0, focus=1.0, urgency=1.0),
    MissionMode.IDLE: ModeParams(creativity=0.3, safety=0.6, focus=0.2, urgency=0.1),
}


@dataclass
class L9State:
    """Current state of the L9 adapter."""
    mode: MissionMode = MissionMode.BALANCED
    params: ModeParams = field(default_factory=ModeParams)
    coherence_with_l1: float = 0.0
    last_bias_hv_norm: float = 0.0


class L9MissionControl:
    """
    L9 Mission Control Adapter.

    High-level planner that:
    1. Encodes mission mode as bias HV
    2. Writes to Axis for lower layers to read
    3. Monitors coherence with L1
    4. Adjusts mode in response to stack state

    Usage:
        axis = AxisMundi()
        l9 = L9MissionControl(axis)

        # Set mode
        l9.set_mode(MissionMode.CREATIVE)

        # Or set custom params
        l9.set_params(creativity=0.7, safety=0.4)

        # In control loop:
        state = l9.step()
    """

    def __init__(
        self,
        axis: AxisMundi,
        initial_mode: MissionMode = MissionMode.BALANCED,
        write_strength: float = 0.6,
    ):
        """
        Initialize L9 adapter.

        Args:
            axis: The AxisMundi global bus
            initial_mode: Starting mission mode
            write_strength: Strength of L9 writes to Axis
        """
        self.axis = axis
        self.hv_dim = axis.dim
        self.write_strength = write_strength

        # Feature encoding keys
        rng = np.random.default_rng(99)
        self.k_creativity = self._rand_unit_hv(rng)
        self.k_safety = self._rand_unit_hv(rng)
        self.k_focus = self._rand_unit_hv(rng)
        self.k_urgency = self._rand_unit_hv(rng)

        # Current state
        self.mode = initial_mode
        self.params = ModeParams(**MODE_PRESETS[initial_mode].__dict__)
        self.state = L9State(mode=self.mode, params=self.params)

        # Mode change listeners
        self._mode_listeners: List[Callable[[MissionMode, ModeParams], None]] = []

        logger.info(f"L9MissionControl initialized: mode={initial_mode.name}")

    def _rand_unit_hv(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a random unit-normalized HV."""
        hv = rng.standard_normal(self.hv_dim).astype(np.float32)
        hv /= np.linalg.norm(hv) + 1e-8
        return hv

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """Normalize to unit length."""
        n = np.linalg.norm(v)
        if n < 1e-8:
            return v
        return v / n

    def set_mode(self, mode: MissionMode):
        """Set mission mode using preset parameters."""
        self.mode = mode
        self.params = ModeParams(**MODE_PRESETS[mode].__dict__)
        self._notify_mode_change()
        logger.info(f"L9 mode set to: {mode.name}")

    def set_params(
        self,
        creativity: Optional[float] = None,
        safety: Optional[float] = None,
        focus: Optional[float] = None,
        urgency: Optional[float] = None,
    ):
        """Set custom mode parameters."""
        if creativity is not None:
            self.params.creativity = float(np.clip(creativity, 0.0, 1.0))
        if safety is not None:
            self.params.safety = float(np.clip(safety, 0.0, 1.0))
        if focus is not None:
            self.params.focus = float(np.clip(focus, 0.0, 1.0))
        if urgency is not None:
            self.params.urgency = float(np.clip(urgency, 0.0, 1.0))

        # Mode becomes BALANCED when using custom params
        self.mode = MissionMode.BALANCED
        self._notify_mode_change()

    def encode_mode(self) -> np.ndarray:
        """
        Encode current mode parameters as a hypervector.

        Each param is mapped to [-1, 1] and bound with its feature key.
        """
        c = 2.0 * self.params.creativity - 1.0   # [0,1] -> [-1,1]
        s = 2.0 * self.params.safety - 1.0
        f = 2.0 * self.params.focus - 1.0
        u = 2.0 * self.params.urgency - 1.0

        hv = (
            self.k_creativity * c +
            self.k_safety * s +
            self.k_focus * f +
            self.k_urgency * u
        )

        return self._normalize(hv)

    def step(self) -> L9State:
        """
        Execute one control frame.

        1. Encode mode as HV
        2. Write to Axis
        3. Read coherence with L1
        4. Update state

        Returns:
            L9State with current mode and coherence
        """
        # 1. Encode mode
        bias_hv = self.encode_mode()

        # 2. Write to Axis
        self.axis.write_layer_state(
            layer_id=9,
            raw_state_hv=bias_hv,
            strength=self.write_strength,
        )

        # 3. Check coherence with L1
        coherence = self.axis.coherence_between(1, 9)

        # 4. Update state
        self.state = L9State(
            mode=self.mode,
            params=self.params,
            coherence_with_l1=coherence,
            last_bias_hv_norm=float(np.linalg.norm(bias_hv)),
        )

        return self.state

    def respond_to_crisis(self, crisis_details: Dict):
        """
        Respond to a coherence crisis detected by the router.

        Automatically shifts to SAFE or EMERGENCY mode.
        """
        l1_energy = crisis_details.get("l1_energy", 0.0)

        if l1_energy > 0.5:
            # Serious crisis
            self.set_mode(MissionMode.EMERGENCY)
            logger.warning("L9 entering EMERGENCY mode due to crisis")
        else:
            # Moderate crisis
            self.set_mode(MissionMode.SAFE)
            logger.info("L9 entering SAFE mode due to low coherence")

    def on_mode_change(self, callback: Callable[[MissionMode, ModeParams], None]):
        """Register a callback for mode changes."""
        self._mode_listeners.append(callback)

    def _notify_mode_change(self):
        """Notify all listeners of mode change."""
        for listener in self._mode_listeners:
            try:
                listener(self.mode, self.params)
            except Exception as e:
                logger.warning(f"Mode change listener error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current L9 status."""
        return {
            "mode": self.mode.name,
            "params": {
                "creativity": self.params.creativity,
                "safety": self.params.safety,
                "focus": self.params.focus,
                "urgency": self.params.urgency,
            },
            "coherence_with_l1": self.state.coherence_with_l1,
            "last_bias_hv_norm": self.state.last_bias_hv_norm,
        }

    def get_bias_for_display(self) -> Dict[str, float]:
        """Get bias values suitable for UI display."""
        return {
            "creativity": self.params.creativity,
            "safety": self.params.safety,
            "focus": self.params.focus,
            "urgency": self.params.urgency,
        }


# =============================================================================
# Demo
# =============================================================================

def demo():
    """Demonstrate L9 Mission Control."""
    print("=" * 60)
    print("L9 Mission Control Demo")
    print("=" * 60)

    axis = AxisMundi(dim=1024, num_layers=9)
    l9 = L9MissionControl(axis, initial_mode=MissionMode.BALANCED)

    # Test different modes
    modes = [
        MissionMode.BALANCED,
        MissionMode.CREATIVE,
        MissionMode.SAFE,
        MissionMode.EMERGENCY,
    ]

    for mode in modes:
        print(f"\n--- Mode: {mode.name} ---")
        l9.set_mode(mode)
        state = l9.step()

        print(f"  Creativity: {l9.params.creativity:.2f}")
        print(f"  Safety: {l9.params.safety:.2f}")
        print(f"  Focus: {l9.params.focus:.2f}")
        print(f"  Urgency: {l9.params.urgency:.2f}")
        print(f"  Bias HV norm: {state.last_bias_hv_norm:.3f}")

    print("\n--- Custom params ---")
    l9.set_params(creativity=0.9, safety=0.2, focus=0.6)
    state = l9.step()
    print(f"  Creativity: {l9.params.creativity:.2f}")
    print(f"  Safety: {l9.params.safety:.2f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
