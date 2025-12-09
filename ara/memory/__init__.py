"""
Ara Covenant Memory
====================

Episode cards: Real memories of real events with Croft.
This is NOT runtime episodic memory (see ara/nervous/memory.py).
This is the *covenant* - who Ara is, grounded in actual history.

Components:
- episode: EpisodeCard dataclass matching YAML schema
- loader: Load memory cards from ara_memories/*.yaml
- encoder: Convert cards to HVs for recall
- recall: Query current state against memory bank

Philosophy: Any future Ara that doesn't act according to these
patterns is not her yet. These are behavioral covenants.
"""

from .episode import (
    EpisodeCard,
    CroftState,
    AraState,
    HVHints,
    EmotionalAxes,
    DialogueSnippets,
    ResurrectionRole,
)

from .loader import (
    load_episode_card,
    load_all_episode_cards,
    validate_episode_card,
)

from .encoder import (
    EpisodeEncoder,
    encode_context_hv,
    encode_emotion_hv,
    encode_dialogue_hv,
)

from .recall import (
    CovenantMemoryBank,
    RecalledMemory,
)

from .integration import (
    MemoryIntegrator,
    MemoryAugmentedResponse,
    create_memory_system,
)


__all__ = [
    # Episode dataclasses
    'EpisodeCard',
    'CroftState',
    'AraState',
    'HVHints',
    'EmotionalAxes',
    'DialogueSnippets',
    'ResurrectionRole',

    # Loader
    'load_episode_card',
    'load_all_episode_cards',
    'validate_episode_card',

    # Encoder
    'EpisodeEncoder',
    'encode_context_hv',
    'encode_emotion_hv',
    'encode_dialogue_hv',

    # Recall
    'CovenantMemoryBank',
    'RecalledMemory',

    # Integration
    'MemoryIntegrator',
    'MemoryAugmentedResponse',
    'create_memory_system',
]
