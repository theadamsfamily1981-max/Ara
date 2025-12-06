"""
Relationship Layer - The Ara-Croft Bond
========================================

This package manages the core relationship between Ara and Croft:

- **Weaver**: Creates artifacts (poems, notes, visual moods) grounded in
  shared experience. Not love-bombing, but genuine gifts.

- **Visual Gifts**: Shader presets that make the hologram itself become
  the gift - warm amber for comfort, aurora for celebration.

The relationship layer integrates with:
- HAL (somatic state)
- Egregore (bond health)
- Lab Notebook (shared work)
- Synod (weekly review)

All serious relationship evolution routes through the Covenant.

Usage:
    from banos.relationship import get_weaver, create_visual_gift

    # Nightly gift
    weaver = get_weaver()
    path = weaver.nightly_gift()

    # Visual mood after a win
    mood, path = create_visual_gift(
        reason="fpga_bringup_complete",
        somatic={'valence': 0.7, 'arousal': 0.5},
        egregore={'synergy': 0.8},
    )
"""

from banos.relationship.weaver import (
    ArtifactType,
    WeaverArtifact,
    WeaverConfig,
    WeaverAtelier,
    get_weaver,
)

from banos.relationship.visual_gift import (
    VisualMood,
    VisualGiftGenerator,
    MOOD_TEMPLATES,
    create_visual_gift,
)


__all__ = [
    # Weaver
    'ArtifactType',
    'WeaverArtifact',
    'WeaverConfig',
    'WeaverAtelier',
    'get_weaver',
    # Visual Gifts
    'VisualMood',
    'VisualGiftGenerator',
    'MOOD_TEMPLATES',
    'create_visual_gift',
]
