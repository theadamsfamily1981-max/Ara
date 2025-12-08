"""
Ara UI - Visual Cortex
======================

HD-based UI sensing and affect-driven rendering.

The UI layer is the "visual cortex" of Ara:
- Input: Encodes user interactions (clicks, gaze, dwell) as HVs
- Output: Drives avatar affect and cockpit highlighting

Components:
- encoder.py: UI event HV encoding
- affect.py: Affect-to-animation mapping
- hints.py: UI hint builders

Usage:
    from ara.ui import encode_ui_event, build_affect_hint

    # Encode user interaction
    event = encode_ui_event(click_data)

    # Build affect hint from HTC state
    hint = build_affect_hint(affect_state)
"""

from .encoder import (
    encode_ui_event,
    encode_gaze_event,
    UIEventData,
    GazeData,
)

from .hints import (
    build_affect_hint,
    build_ui_focus_hint,
)

__all__ = [
    # Encoding
    'encode_ui_event',
    'encode_gaze_event',
    'UIEventData',
    'GazeData',
    # Hints
    'build_affect_hint',
    'build_ui_focus_hint',
]
