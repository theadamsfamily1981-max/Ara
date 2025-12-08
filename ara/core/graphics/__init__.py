"""
Ara Core Graphics - Visual Cortex Logic
=======================================

Pure HD logic for graphics and UI:
- affect.py: Compute affect from resonance/reward history
- palette.py: Map affect to shader parameters
- memory_palace.py: Project attractors to 3D visualization
- event_codec.py: Encode UI events as HVs

No GPU code here - just the math that drives rendering.
"""

from .affect import affect_from_history, AffectState
from .palette import shader_params_from_affect, ShaderParams
from .memory_palace import project_attractors
from .event_codec import encode_ui_interaction

__all__ = [
    'affect_from_history',
    'AffectState',
    'shader_params_from_affect',
    'ShaderParams',
    'project_attractors',
    'encode_ui_interaction',
]
