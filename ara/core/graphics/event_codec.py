"""
Ara Event Codec - UI Events to Hypervectors
============================================

Encodes UI interaction events (gaze, hover, click) into hypervectors
that can bias the HTC's context encoding.

Event HV format:
    H_ui = ROLE_UI ⊕ H_PANEL ⊕ H_ACTION ⊕ H_DWELL

This allows the soul to learn from operator attention patterns:
- Which panels they focus on during crises
- What they click when stressed
- How long they dwell on certain views
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np

from ara.hd.ops import bind, bundle, random_hv_from_string, DIM
from ara.hd.vocab import get_vocab


# =============================================================================
# Panel and Action HV Caches
# =============================================================================

_PANEL_HVS: Dict[str, np.ndarray] = {}
_ACTION_HVS: Dict[str, np.ndarray] = {}


def _get_panel_hv(panel_id: str) -> np.ndarray:
    """Get or create HV for a panel."""
    if panel_id not in _PANEL_HVS:
        _PANEL_HVS[panel_id] = random_hv_from_string(f"PANEL:{panel_id}")
    return _PANEL_HVS[panel_id]


def _get_action_hv(action: str) -> np.ndarray:
    """Get or create HV for an action."""
    if action not in _ACTION_HVS:
        _ACTION_HVS[action] = random_hv_from_string(f"UI_ACTION:{action}")
    return _ACTION_HVS[action]


# =============================================================================
# Dwell Time Binning
# =============================================================================

DWELL_BINS = ["MINIMAL", "LOW", "MED", "HIGH", "EXTREME"]


def bin_dwell_time(seconds: float) -> str:
    """Bin dwell time into discrete levels."""
    if seconds < 0.5:
        return "MINIMAL"
    if seconds < 2.0:
        return "LOW"
    if seconds < 5.0:
        return "MED"
    if seconds < 15.0:
        return "HIGH"
    return "EXTREME"


# =============================================================================
# Encoding Functions
# =============================================================================

def encode_ui_interaction(
    panel_id: str,
    action: str,
    dwell_bin: str = "MINIMAL",
    target_element: Optional[str] = None,
) -> np.ndarray:
    """
    Encode a UI interaction into a hypervector.

    H_ui = ROLE_UI ⊕ H_PANEL ⊕ H_ACTION ⊕ H_DWELL [⊕ H_TARGET]

    Args:
        panel_id: Panel identifier (e.g., "network", "soul_map", "thermal")
        action: Action type (e.g., "click", "hover", "gaze", "scroll")
        dwell_bin: Discretized dwell time ("MINIMAL", "LOW", "MED", "HIGH", "EXTREME")
        target_element: Optional specific element within panel

    Returns:
        Hypervector encoding the interaction
    """
    vocab = get_vocab()

    # Get component HVs
    h_panel = _get_panel_hv(panel_id)
    h_action = _get_action_hv(action)
    h_dwell = vocab.bin(dwell_bin)

    # Bind components with feature markers
    components = [
        bind(vocab.feature("PANEL"), h_panel),
        bind(vocab.feature("ACTION"), h_action),
        bind(vocab.feature("DWELL"), h_dwell),
    ]

    # Add target if present
    if target_element:
        h_target = random_hv_from_string(f"UI_TARGET:{target_element}")
        components.append(bind(vocab.feature("TARGET"), h_target))

    # Bundle and bind with UI role
    h_ui = bundle(components)
    h_bound = bind(vocab.role("UI"), h_ui)

    return h_bound


def encode_gaze_fixation(
    panel_id: str,
    x: float,
    y: float,
    duration_ms: float,
) -> np.ndarray:
    """
    Encode a gaze fixation event.

    Args:
        panel_id: Panel where gaze is focused
        x, y: Normalized gaze position [0, 1]
        duration_ms: Fixation duration in milliseconds

    Returns:
        Hypervector encoding the gaze event
    """
    vocab = get_vocab()

    h_panel = _get_panel_hv(panel_id)

    # Bin duration
    if duration_ms < 100:
        duration_bin = "MINIMAL"
    elif duration_ms < 300:
        duration_bin = "LOW"
    elif duration_ms < 700:
        duration_bin = "MED"
    elif duration_ms < 1500:
        duration_bin = "HIGH"
    else:
        duration_bin = "EXTREME"

    h_duration = vocab.bin(duration_bin)

    # Bin position into quadrant
    quadrant = _position_to_quadrant(x, y)
    h_position = random_hv_from_string(f"GAZE_POS:{quadrant}")

    components = [
        bind(vocab.feature("PANEL"), h_panel),
        bind(vocab.feature("GAZE_DURATION"), h_duration),
        bind(vocab.feature("GAZE_POS"), h_position),
    ]

    h_gaze = bundle(components)
    h_bound = bind(vocab.role("VISION"), h_gaze)

    return h_bound


def _position_to_quadrant(x: float, y: float) -> str:
    """Map normalized position to quadrant name."""
    col = "LEFT" if x < 0.33 else "CENTER" if x < 0.66 else "RIGHT"
    row = "TOP" if y < 0.33 else "MIDDLE" if y < 0.66 else "BOTTOM"
    return f"{row}_{col}"


def decode_ui_interaction(
    hv: np.ndarray,
    candidate_panels: list[str],
    candidate_actions: list[str],
) -> Dict[str, Any]:
    """
    Attempt to decode a UI interaction HV.

    Uses similarity matching against known panels and actions.

    Args:
        hv: Hypervector to decode
        candidate_panels: List of possible panel IDs
        candidate_actions: List of possible actions

    Returns:
        Dict with best-match panel, action, and confidence scores
    """
    from ara.hd.ops import cosine

    vocab = get_vocab()

    # Unbind UI role
    h_unbound = bind(vocab.role("UI"), hv)

    # Find best panel match
    best_panel = None
    best_panel_sim = -1.0
    for panel in candidate_panels:
        h_panel = _get_panel_hv(panel)
        h_panel_bound = bind(vocab.feature("PANEL"), h_panel)
        sim = cosine(h_unbound, h_panel_bound)
        if sim > best_panel_sim:
            best_panel_sim = sim
            best_panel = panel

    # Find best action match
    best_action = None
    best_action_sim = -1.0
    for action in candidate_actions:
        h_action = _get_action_hv(action)
        h_action_bound = bind(vocab.feature("ACTION"), h_action)
        sim = cosine(h_unbound, h_action_bound)
        if sim > best_action_sim:
            best_action_sim = sim
            best_action = action

    return {
        "panel": best_panel,
        "panel_confidence": best_panel_sim,
        "action": best_action,
        "action_confidence": best_action_sim,
    }


# =============================================================================
# Attention Bias
# =============================================================================

def compute_attention_bias(
    recent_ui_hvs: list[np.ndarray],
    decay: float = 0.9,
) -> np.ndarray:
    """
    Compute an attention bias HV from recent UI interactions.

    This can be added to context HVs to bias perception toward
    what the operator is paying attention to.

    Args:
        recent_ui_hvs: List of recent UI interaction HVs
        decay: Exponential decay factor for older events

    Returns:
        Weighted bundle of recent interactions
    """
    if not recent_ui_hvs:
        return np.zeros(DIM, dtype=np.uint8)

    # Compute weights with exponential decay
    n = len(recent_ui_hvs)
    weights = [decay ** (n - 1 - i) for i in range(n)]

    return bundle(recent_ui_hvs, weights=weights)


__all__ = [
    'encode_ui_interaction',
    'encode_gaze_fixation',
    'decode_ui_interaction',
    'compute_attention_bias',
    'bin_dwell_time',
    'DWELL_BINS',
]
