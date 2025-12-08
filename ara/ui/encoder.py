"""
Ara UI Encoder - Interaction to HV Encoding
===========================================

Encodes UI interactions into hypervectors for the HTC.

Every UI event becomes:
    H_ui = ROLE_UI ⊕ H_PANEL ⊕ H_ACTION ⊕ H_DWELL ⊕ [H_TARGET]

This allows the soul to learn from your interaction patterns:
- Which panels you focus on during crises
- What you click when stressed
- How long you dwell on certain views

The HTC learns to weight information you pay attention to.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List

from ara.io.types import HDInputEvent, IOChannel, HV
from ara.hd.ops import bind, bundle, DIM
from ara.hd.vocab import get_vocab


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class UIEventData:
    """UI interaction event data."""
    panel_id: str                      # "network", "soul_map", "thermal", etc.
    action: str                        # "click", "open", "close", "hover", "scroll"
    dwell_seconds: float = 0.0         # How long focused on this element
    target_node: Optional[str] = None  # If clicking on a specific node
    target_flow: Optional[str] = None  # If clicking on a specific flow
    target_widget: Optional[str] = None  # Specific widget within panel
    x: float = 0.0                     # Normalized position [0, 1]
    y: float = 0.0
    event_id: Optional[str] = None


@dataclass
class GazeData:
    """Gaze/attention tracking data."""
    panel_id: str                      # Where gaze is focused
    x: float = 0.5                     # Normalized gaze position [0, 1]
    y: float = 0.5
    fixation_duration_ms: float = 0.0
    saccade_count: int = 0             # Eye movements in window
    target_element: Optional[str] = None


# =============================================================================
# Binning Functions
# =============================================================================

def bin_dwell(seconds: float) -> str:
    """Bin dwell time into discrete levels."""
    if seconds < 0.5:
        return "MINIMAL"
    if seconds < 2.0:
        return "LOW"
    if seconds < 5.0:
        return "MED"
    if seconds < 15.0:
        return "HIGH"
    return "EXTREME"  # Very long focus


def bin_fixation(duration_ms: float) -> str:
    """Bin gaze fixation duration."""
    if duration_ms < 100:
        return "MINIMAL"
    if duration_ms < 300:
        return "LOW"
    if duration_ms < 700:
        return "MED"
    if duration_ms < 1500:
        return "HIGH"
    return "EXTREME"


def bin_position(x: float, y: float) -> str:
    """Bin screen position into regions."""
    # Divide screen into 9 regions
    col = "LEFT" if x < 0.33 else "CENTER" if x < 0.66 else "RIGHT"
    row = "TOP" if y < 0.33 else "MIDDLE" if y < 0.66 else "BOTTOM"
    return f"{row}_{col}"


# =============================================================================
# Encoding Functions
# =============================================================================

def encode_ui_event(event: UIEventData) -> HDInputEvent:
    """
    Encode a UI event into an HDInputEvent.

    H_ui = ROLE_UI ⊕ H_PANEL ⊕ H_ACTION ⊕ H_DWELL ⊕ [H_TARGET]
    """
    vocab = get_vocab()

    # Get component HVs
    h_panel = vocab.custom("panel", event.panel_id)
    h_action = vocab.custom("action", event.action)
    h_dwell = vocab.bin(bin_dwell(event.dwell_seconds))
    h_position = vocab.custom("position", bin_position(event.x, event.y))

    # Bind components with feature markers
    components = [
        bind(vocab.feature("PANEL"), h_panel),
        bind(vocab.feature("ACTION"), h_action),
        bind(vocab.feature("DWELL"), h_dwell),
        bind(vocab.feature("POSITION"), h_position),
    ]

    # Add target if present
    if event.target_node:
        h_target = vocab.custom("node", event.target_node)
        components.append(bind(vocab.feature("TARGET_NODE"), h_target))

    if event.target_flow:
        h_target = vocab.custom("flow", event.target_flow)
        components.append(bind(vocab.feature("TARGET_FLOW"), h_target))

    if event.target_widget:
        h_widget = vocab.custom("widget", event.target_widget)
        components.append(bind(vocab.feature("WIDGET"), h_widget))

    # Bundle and bind with UI role
    h_ui = bundle(components)
    h_bound = bind(vocab.role("UI"), h_ui)

    # Priority based on action type
    priority = 0.5
    if event.action in ["click", "select"]:
        priority = 0.8
    elif event.action in ["open", "close"]:
        priority = 0.7
    elif event.dwell_seconds > 5.0:
        priority = 0.6  # Long dwell is significant

    return HDInputEvent(
        channel=IOChannel.UI,
        role="ROLE_UI_EVENT",
        meta={
            "panel_id": event.panel_id,
            "action": event.action,
            "dwell_seconds": event.dwell_seconds,
            "target_node": event.target_node,
            "target_flow": event.target_flow,
            "target_widget": event.target_widget,
            "position": {"x": event.x, "y": event.y},
        },
        hv=h_bound,
        priority=priority,
        source_id=event.event_id,
    )


def encode_gaze_event(gaze: GazeData) -> HDInputEvent:
    """
    Encode a gaze/attention event into an HDInputEvent.

    H_gaze = ROLE_UI_GAZE ⊕ H_PANEL ⊕ H_FIXATION ⊕ H_POSITION
    """
    vocab = get_vocab()

    # Get component HVs
    h_panel = vocab.custom("panel", gaze.panel_id)
    h_fixation = vocab.bin(bin_fixation(gaze.fixation_duration_ms))
    h_position = vocab.custom("position", bin_position(gaze.x, gaze.y))

    components = [
        bind(vocab.feature("PANEL"), h_panel),
        bind(vocab.feature("FIXATION"), h_fixation),
        bind(vocab.feature("GAZE_POS"), h_position),
    ]

    if gaze.target_element:
        h_target = vocab.custom("element", gaze.target_element)
        components.append(bind(vocab.feature("TARGET"), h_target))

    # Bundle and bind
    h_gaze = bundle(components)
    h_bound = bind(vocab.role("VISION"), h_gaze)  # Visual attention

    # Priority based on fixation
    priority = 0.3  # Gaze is lower priority than explicit clicks
    if gaze.fixation_duration_ms > 500:
        priority = 0.5
    if gaze.fixation_duration_ms > 1000:
        priority = 0.6

    return HDInputEvent(
        channel=IOChannel.UI,
        role="ROLE_UI_GAZE",
        meta={
            "panel_id": gaze.panel_id,
            "fixation_ms": gaze.fixation_duration_ms,
            "position": {"x": gaze.x, "y": gaze.y},
            "target": gaze.target_element,
        },
        hv=h_bound,
        priority=priority,
    )


def encode_ui_batch(events: List[UIEventData]) -> List[HDInputEvent]:
    """Encode a batch of UI events."""
    return [encode_ui_event(e) for e in events]


# =============================================================================
# Panel Priority Mapping
# =============================================================================

# Which panels are important during which contexts
PANEL_CONTEXT_WEIGHTS: Dict[str, Dict[str, float]] = {
    "thermal_emergency": {
        "thermal": 2.0,
        "soul_map": 1.5,
        "network": 0.5,
    },
    "network_crisis": {
        "network": 2.0,
        "flows": 1.5,
        "thermal": 0.5,
    },
    "deep_work": {
        "soul_map": 1.5,
        "task": 1.0,
        "network": 0.3,
    },
    "founder_fatigue": {
        "status": 2.0,
        "soul_map": 1.0,
        "thermal": 0.5,
    },
}


def get_panel_weight(panel_id: str, context: str) -> float:
    """Get contextual weight for a panel."""
    context_weights = PANEL_CONTEXT_WEIGHTS.get(context, {})
    return context_weights.get(panel_id, 1.0)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'UIEventData',
    'GazeData',
    'encode_ui_event',
    'encode_gaze_event',
    'encode_ui_batch',
    'bin_dwell',
    'bin_fixation',
    'bin_position',
    'PANEL_CONTEXT_WEIGHTS',
    'get_panel_weight',
]
