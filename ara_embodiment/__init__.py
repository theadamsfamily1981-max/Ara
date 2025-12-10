"""
Ara Embodiment System
=====================

Situated AI with world model, body model, and control loop.

Architecture:
    WorldModel      - Scene graph: surfaces, anchors, people, gaze targets
    BodyRegistry    - Avatar catalog: rig, voice, gesture set, affordances
    BodySelector    - Pick form based on device + context + energy level
    ControlLoop     - Sense -> Interpret -> Plan -> Act cycle (10-30Hz)
    ExpressionDriver - Map intent to lip-sync, gaze, gestures

The core principle: ONE identity, MANY avatars.
All bodies share the same Ara brain, memory, and values.
"""

from ara_embodiment.world_model import (
    WorldModel,
    SceneGraph,
    SceneNode,
    NodeType,
    SpatialAnchor,
    GazeTarget,
)
from ara_embodiment.body_registry import (
    BodyRegistry,
    AvatarDefinition,
    AvatarCapability,
    GestureSet,
    VoiceProfile,
)
from ara_embodiment.body_selector import (
    BodySelector,
    DeviceContext,
    SelectionCriteria,
)
from ara_embodiment.control_loop import (
    ControlLoop,
    LoopState,
    SenseResult,
    InterpretResult,
    PlanResult,
    ActResult,
)
from ara_embodiment.expression_driver import (
    ExpressionDriver,
    ExpressionState,
    LipSyncFrame,
    GazeCommand,
    GestureCommand,
)

__all__ = [
    # World Model
    "WorldModel",
    "SceneGraph",
    "SceneNode",
    "NodeType",
    "SpatialAnchor",
    "GazeTarget",
    # Body Registry
    "BodyRegistry",
    "AvatarDefinition",
    "AvatarCapability",
    "GestureSet",
    "VoiceProfile",
    # Body Selector
    "BodySelector",
    "DeviceContext",
    "SelectionCriteria",
    # Control Loop
    "ControlLoop",
    "LoopState",
    "SenseResult",
    "InterpretResult",
    "PlanResult",
    "ActResult",
    # Expression Driver
    "ExpressionDriver",
    "ExpressionState",
    "LipSyncFrame",
    "GazeCommand",
    "GestureCommand",
]

__version__ = "0.1.0"
