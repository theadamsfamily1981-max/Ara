# neuromod_inference.py
"""
Inference-time Neuromodulation Shim

Maps (appraisal, drives) -> decoding parameters, mirroring TFAN's
training-time emotion controller, but for live inference.

This makes Ara's emotional state actually influence how she responds:
- High arousal -> higher temperature (more creative/varied)
- Negative valence -> lower temperature (more conservative)
- Fatigue/instability -> shorter outputs
- Good mood + system health -> more willing to use tools
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class NeuromodConfig:
    # Base decoding settings
    base_temperature: float = 0.7
    min_temperature: float = 0.3
    max_temperature: float = 1.3

    base_top_p: float = 0.9
    min_top_p: float = 0.6
    max_top_p: float = 1.0

    base_max_tokens: int = 512
    min_max_tokens: int = 64
    max_max_tokens: int = 2048

    # How much to smooth multipliers (0 = no smoothing, 1 = frozen)
    smoothing: float = 0.8

    # How strongly to react to affect/homeostat
    arousal_temp_gain: float = 0.4   # how much arousal can raise temp
    neg_valence_temp_drop: float = 0.3
    fatigue_token_drop: float = 0.5  # max 50% fewer tokens at full fatigue
    instability_token_drop: float = 0.4

    # Tool risk range: 0 = ultra conservative, 1 = very exploratory
    min_tool_risk: float = 0.1
    max_tool_risk: float = 0.9


@dataclass
class NeuromodState:
    """Keeps smoothed multipliers between steps."""
    temp_mult: float = 1.0
    top_p_mult: float = 1.0
    tokens_mult: float = 1.0
    tool_risk: float = 0.5  # 0..1


@dataclass
class DecodingParams:
    temperature: float
    top_p: float
    max_tokens: int
    tool_risk: float  # pass into AEPO / tool-gating if you want


class Neuromodulator:
    """
    Inference-time neuromodulation shim.

    Maps (appraisal, drives) -> decoding parameters, roughly mirroring the
    TFAN emotion controller, but for inference instead of training.

    Expected inputs:

    Appraisal (from AppraisalEngine):
        appraisal.valence     # float in [-1, 1]
        appraisal.arousal     # float in [0, 1]
        appraisal.dominance   # float in [0, 1]
        appraisal.instability # float in [0, 1]  (0 = stable, 1 = very unstable)

    Drives (from HomeostaticCore / DriveState):
        drives.energy     # 0..1; 1 = very fatigued
        drives.integrity  # 0..1; 1 = system unhappy / error-y
        drives.novelty    # 0..1; 1 = hungry for new stuff
        drives.safety     # 0..1; 1 = feeling unsafe
    """

    def __init__(self, config: NeuromodConfig | None = None):
        self.cfg = config or NeuromodConfig()
        self.state = NeuromodState()

    def _ema(self, prev: float, new: float) -> float:
        """Exponential moving average for smoothing."""
        s = self.cfg.smoothing
        return s * prev + (1.0 - s) * new

    def update(
        self,
        appraisal: Any,   # AppraisalEngine output
        drives: Any,      # HomeostaticCore / DriveState output
    ) -> DecodingParams:
        """
        Compute new decoding params given current affect + drives.
        Call once per turn before generating a response.
        """

        cfg = self.cfg

        # -------- Temperature: arousal up -> temp up; neg valence -> temp down --------
        arousal = float(getattr(appraisal, "arousal", 0.0))  # 0..1
        valence = float(getattr(appraisal, "valence", 0.0))  # -1..1

        # Base arousal contribution: 1.0 +/- arousal_temp_gain * arousal
        # arousal=0 -> 1.0, arousal=1 -> 1 + gain
        arousal_factor = 1.0 + cfg.arousal_temp_gain * (arousal - 0.5) * 2.0

        # Negative valence pulls temp down
        neg_v = max(-valence, 0.0)  # 0..1 for negative moods
        valence_factor = 1.0 - cfg.neg_valence_temp_drop * neg_v

        raw_temp_mult = arousal_factor * valence_factor

        # -------- Tokens: fatigue/instability -> shorter outputs --------
        energy = float(getattr(drives, "energy", 0.0))         # 0..1, 1 = tired
        instability = float(getattr(appraisal, "instability", 0.0))  # 0..1

        fatigue_drop = cfg.fatigue_token_drop * energy
        instability_drop = cfg.instability_token_drop * instability

        raw_tokens_mult = 1.0 - max(0.0, fatigue_drop + instability_drop)
        raw_tokens_mult = max(0.3, raw_tokens_mult)  # never below 30%

        # -------- Top-p: slightly widen/narrow with arousal --------
        raw_top_p_mult = 1.0 + 0.2 * (arousal - 0.5) * 2.0  # +/-20%

        # -------- Tool risk: high valence + moderate arousal -> more willing --------
        integrity = float(getattr(drives, "integrity", 0.0))  # 0..1, 1 = unhappy
        safety = float(getattr(drives, "safety", 0.0))        # 0..1, 1 = feels unsafe

        # Simple heuristic: like tools when mood good, system intact, not unsafe
        mood_score = 0.5 * (valence + 1.0) + 0.5 * (1.0 - instability)
        mood_score = max(0.0, min(1.0, mood_score))

        integrity_penalty = 0.5 * integrity
        safety_penalty = 0.5 * safety

        raw_tool_risk = mood_score * (1.0 - integrity_penalty - safety_penalty)
        raw_tool_risk = max(0.0, min(1.0, raw_tool_risk))

        # -------- Smooth and clamp --------
        self.state.temp_mult = self._ema(self.state.temp_mult, raw_temp_mult)
        self.state.tokens_mult = self._ema(self.state.tokens_mult, raw_tokens_mult)
        self.state.top_p_mult = self._ema(self.state.top_p_mult, raw_top_p_mult)
        self.state.tool_risk = self._ema(self.state.tool_risk, raw_tool_risk)

        temperature = cfg.base_temperature * self.state.temp_mult
        temperature = max(cfg.min_temperature, min(cfg.max_temperature, temperature))

        top_p = cfg.base_top_p * self.state.top_p_mult
        top_p = max(cfg.min_top_p, min(cfg.max_top_p, top_p))

        max_tokens = int(cfg.base_max_tokens * self.state.tokens_mult)
        max_tokens = max(cfg.min_max_tokens, min(cfg.max_max_tokens, max_tokens))

        tool_risk = cfg.min_tool_risk + self.state.tool_risk * (
            cfg.max_tool_risk - cfg.min_tool_risk
        )

        return DecodingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tool_risk=tool_risk,
        )

    def get_state_summary(self) -> Dict[str, float]:
        """Get current internal state for debugging/logging."""
        return {
            "temp_mult": self.state.temp_mult,
            "tokens_mult": self.state.tokens_mult,
            "top_p_mult": self.state.top_p_mult,
            "tool_risk": self.state.tool_risk,
        }
