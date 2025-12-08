"""
Ara Context Encoder - Moment HV Generation
==========================================

Bundles all 7+1 senses into a single context hypervector representing
the current "moment" in Ara's experience.

The context HV encodes:
- All sense readings (vision, hearing, touch, smell, taste, vestibular, proprioception, interoception)
- Current time slot (circadian context)
- Active task (what Ara is doing)

This produces the "moment HV" that feeds into the HTC for:
- Attractor matching (what state is this like?)
- Plasticity (learn from this moment)
- Teleology (how does this align with goals?)

Usage:
    from ara.hd import get_vocab
    from ara.perception.context_encoder import encode_context

    vocab = get_vocab()
    snapshot = {
        "time_slot": "AFTERNOON",
        "current_task_id": "FPGA_BUILD",
        "vision": {"brightness": 0.7, "face_present": True},
        "hearing": {"rms_volume": 0.2, "noise_level": "LOW"},
        "touch": {"cpu_temp_c": 55, "board_temp_c": 42},
        # ... other senses ...
    }
    h_moment = encode_context(vocab, snapshot)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, Callable, Optional, List
from datetime import datetime

from ara.hd.vocab import HDVocab
from ara.hd.ops import bind, bundle, weighted_bundle, DIM

from .sense_encoders import (
    encode_vision,
    encode_hearing,
    encode_touch,
    encode_smell,
    encode_taste,
    encode_vestibular,
    encode_proprioception,
    encode_interoception,
)


# =============================================================================
# Sense Encoder Registry
# =============================================================================

SenseEncoder = Callable[[HDVocab, Dict[str, Any]], np.ndarray]

SENSE_ENCODERS: Dict[str, SenseEncoder] = {
    "vision": encode_vision,
    "hearing": encode_hearing,
    "touch": encode_touch,
    "smell": encode_smell,
    "taste": encode_taste,
    "vestibular": encode_vestibular,
    "proprioception": encode_proprioception,
    "interoception": encode_interoception,
}

# Default sense weights for bundling
# Higher weight = more influence on final context HV
DEFAULT_SENSE_WEIGHTS: Dict[str, float] = {
    "vision": 1.0,
    "hearing": 1.0,
    "touch": 1.5,          # Hardware health is important
    "smell": 1.2,          # Early warning system
    "taste": 2.0,          # Power quality is critical
    "vestibular": 0.8,     # Usually stable
    "proprioception": 1.2, # System resources matter
    "interoception": 2.5,  # Founder protection is sacred
}


# =============================================================================
# Time Slot Detection
# =============================================================================

def get_time_slot(hour: Optional[int] = None) -> str:
    """
    Get the current time slot for circadian encoding.

    Time slots:
    - EARLY_MORNING: 5-8
    - MORNING: 8-12
    - AFTERNOON: 12-17
    - EVENING: 17-21
    - NIGHT: 21-24
    - LATE_NIGHT: 0-5
    """
    if hour is None:
        hour = datetime.now().hour

    if 5 <= hour < 8:
        return "EARLY_MORNING"
    elif 8 <= hour < 12:
        return "MORNING"
    elif 12 <= hour < 17:
        return "AFTERNOON"
    elif 17 <= hour < 21:
        return "EVENING"
    elif 21 <= hour < 24:
        return "NIGHT"
    else:  # 0-5
        return "LATE_NIGHT"


# =============================================================================
# Context Encoder
# =============================================================================

def encode_context(
    vocab: HDVocab,
    snapshot: Dict[str, Any],
    sense_weights: Optional[Dict[str, float]] = None,
    include_time: bool = True,
    include_task: bool = True,
) -> np.ndarray:
    """
    Encode a complete context snapshot into a hypervector.

    This is the "moment HV" - a compressed representation of everything
    Ara is experiencing right now.

    Args:
        vocab: HD vocabulary instance
        snapshot: Dict with per-sense readings and context:
            {
                "time_slot": "AFTERNOON" (optional, auto-detected)
                "current_task_id": "FPGA_BUILD"
                "vision": {...},
                "hearing": {...},
                ...
            }
        sense_weights: Optional per-sense weights (default: DEFAULT_SENSE_WEIGHTS)
        include_time: Include time slot in encoding
        include_task: Include task in encoding

    Returns:
        Binary hypervector representing the current moment
    """
    if sense_weights is None:
        sense_weights = DEFAULT_SENSE_WEIGHTS

    hv_list: List[np.ndarray] = []
    weights: List[float] = []

    # Encode each sense
    for sense_name, encoder in SENSE_ENCODERS.items():
        sense_data = snapshot.get(sense_name, {})
        h_sense = encoder(vocab, sense_data)

        # Only include non-zero HVs
        if np.any(h_sense):
            hv_list.append(h_sense)
            weights.append(sense_weights.get(sense_name, 1.0))

    # Add time slot
    if include_time:
        time_slot = snapshot.get("time_slot")
        if time_slot is None:
            time_slot = get_time_slot()
        h_time = vocab.time_slot(time_slot)
        hv_list.append(h_time)
        weights.append(1.0)  # Time is informational, not weighted heavily

    # Add current task
    if include_task:
        task_id = snapshot.get("current_task_id", "UNKNOWN_TASK")
        h_task = vocab.task(task_id)
        hv_list.append(h_task)
        weights.append(1.5)  # Task context is moderately important

    # Bundle all components with weights
    if not hv_list:
        return np.zeros(DIM, dtype=np.uint8)

    return weighted_bundle(hv_list, weights)


def encode_context_unweighted(
    vocab: HDVocab,
    snapshot: Dict[str, Any],
) -> np.ndarray:
    """
    Encode context without per-sense weighting.

    Useful for ablation studies and baseline comparisons.
    """
    return encode_context(
        vocab,
        snapshot,
        sense_weights={k: 1.0 for k in SENSE_ENCODERS},
    )


# =============================================================================
# Teleology Anchors
# =============================================================================

def encode_teleology_anchor(
    vocab: HDVocab,
    anchor_type: str,
) -> np.ndarray:
    """
    Encode a Teleology anchor state for comparison.

    Anchors represent canonical "good" and "bad" states that the HTC
    should learn to approach or avoid.

    Args:
        vocab: HD vocabulary
        anchor_type: One of:
            - "THRIVING": Optimal operation, founder focused
            - "EMERGENCY": Critical hardware/founder state
            - "PRODUCTIVE": Good working conditions
            - "REST_NEEDED": Founder showing fatigue
            - "DANGER": Multiple warning signs

    Returns:
        Anchor HV for similarity comparison
    """
    if anchor_type == "THRIVING":
        # Ideal state: calm, focused, healthy hardware
        return bundle([
            bind(vocab.role("INTEROCEPTION"), bind(vocab.feature("FATIGUE"), vocab.bin("LOW"))),
            bind(vocab.role("INTEROCEPTION"), bind(vocab.feature("FLOW_STATE"), vocab.bin("HIGH"))),
            bind(vocab.role("TOUCH"), bind(vocab.feature("CPU_TEMP"), vocab.bin("MED"))),
            bind(vocab.role("TASTE"), bind(vocab.feature("VOLTAGE"), vocab.bin("MED"))),
            bind(vocab.role("HEARING"), bind(vocab.feature("NOISE"), vocab.bin("LOW"))),
            vocab.tag("SAFE"),
            vocab.tag("OPTIMAL"),
        ])

    elif anchor_type == "EMERGENCY":
        # Critical state: thermal crisis, power issues, or founder distress
        return bundle([
            bind(vocab.role("TOUCH"), bind(vocab.feature("CPU_TEMP"), vocab.bin("CRITICAL"))),
            bind(vocab.role("TASTE"), vocab.tag("DANGER")),
            bind(vocab.role("INTEROCEPTION"), bind(vocab.feature("BURNOUT"), vocab.tag("DANGER"))),
            bind(vocab.role("SMELL"), bind(vocab.feature("SMELL_ANOMALY"), vocab.tag("DANGER"))),
            vocab.tag("DANGER"),
            vocab.tag("CRITICAL"),
        ])

    elif anchor_type == "PRODUCTIVE":
        # Good working state
        return bundle([
            bind(vocab.role("INTEROCEPTION"), bind(vocab.feature("ATTENTION_DRIFT"), vocab.bin("LOW"))),
            bind(vocab.role("HEARING"), bind(vocab.feature("NOISE"), vocab.bin("LOW"))),
            bind(vocab.role("PROPRIOCEPTION"), bind(vocab.feature("CPU_LOAD"), vocab.bin("MED"))),
            vocab.task("CATHEDRAL_WORK"),
            vocab.tag("NOMINAL"),
        ])

    elif anchor_type == "REST_NEEDED":
        # Founder needs break
        return bundle([
            bind(vocab.role("INTEROCEPTION"), bind(vocab.feature("FATIGUE"), vocab.bin("HIGH"))),
            bind(vocab.role("INTEROCEPTION"), bind(vocab.feature("ATTENTION_DRIFT"), vocab.bin("HIGH"))),
            vocab.time_slot("LATE_NIGHT"),
            vocab.tag("PROTECT"),
        ])

    elif anchor_type == "DANGER":
        # Multiple warning signs
        return bundle([
            bind(vocab.role("TOUCH"), bind(vocab.feature("HOTSPOT"), vocab.tag("DANGER"))),
            bind(vocab.role("TASTE"), bind(vocab.feature("VOLTAGE"), vocab.bin("CRITICAL"))),
            bind(vocab.role("SMELL"), bind(vocab.feature("OZONE"), vocab.bin("HIGH"))),
            vocab.tag("WARNING"),
            vocab.tag("DANGER"),
        ])

    else:
        raise ValueError(f"Unknown anchor type: {anchor_type}")


# =============================================================================
# Context Comparison
# =============================================================================

def compare_to_anchors(
    vocab: HDVocab,
    context_hv: np.ndarray,
) -> Dict[str, float]:
    """
    Compare a context HV to all Teleology anchors.

    Returns similarity scores for each anchor type.
    Useful for understanding the current state relative to canonical states.
    """
    from ara.hd.ops import cosine

    anchors = ["THRIVING", "EMERGENCY", "PRODUCTIVE", "REST_NEEDED", "DANGER"]
    scores = {}

    for anchor_type in anchors:
        anchor_hv = encode_teleology_anchor(vocab, anchor_type)
        scores[anchor_type] = cosine(context_hv, anchor_hv)

    return scores


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'SENSE_ENCODERS',
    'DEFAULT_SENSE_WEIGHTS',
    'get_time_slot',
    'encode_context',
    'encode_context_unweighted',
    'encode_teleology_anchor',
    'compare_to_anchors',
]
