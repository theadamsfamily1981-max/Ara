"""
Ara MDP - Markov Decision Process Schema
=========================================

The complete multimodal control loop as a formal MDP:

    State → Plan → Execute → Observe → Reward → Learn → State'

Key exports:
- EmotionControl: Shared emotion vector across modalities
- State, Inputs, Plan, Outputs, Rewards: MDP components
- Episode: Memory record of one interaction
- MDPTransition: Complete atomic tick

Usage:
    from ara_core.mdp import (
        State, Inputs, Plan, PlanBlock, BlockType,
        Outputs, Rewards, Episode, EmotionControl,
        create_transition
    )
"""

from .schema import (
    # Emotion
    EmotionControl,

    # Inputs
    UserSignals,
    HardwareState,
    MemoryContext,
    ExternalContext,
    Inputs,

    # State
    UserState,
    AraState,
    State,

    # Plan (Actions)
    BlockType,
    PlanBlock,
    Plan,

    # Outputs
    VoiceOutput,
    VisualOutput,
    UIOutput,
    HardwareMetrics,
    Outputs,

    # Rewards
    Rewards,

    # Memory
    Episode,

    # Transition
    MDPTransition,
    create_transition,
)

__all__ = [
    "EmotionControl",
    "UserSignals", "HardwareState", "MemoryContext", "ExternalContext", "Inputs",
    "UserState", "AraState", "State",
    "BlockType", "PlanBlock", "Plan",
    "VoiceOutput", "VisualOutput", "UIOutput", "HardwareMetrics", "Outputs",
    "Rewards",
    "Episode",
    "MDPTransition", "create_transition",
]
