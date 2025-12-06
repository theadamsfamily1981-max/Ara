"""
Gift Archetypes - The Studio's Artifact Generators
===================================================

This package contains concrete Gift generators that create real artifacts:
- SVG visualizations
- Interactive HTML dashboards
- Rendered diagrams

Each generator takes system/code state and produces a beautiful,
illuminating artifact that solves a problem through insight.

Gift Philosophy:
    "A gift is not just useful - it is surprising, beautiful, and
     perfectly timed. It solves a problem you didn't know you had,
     or illuminates a solution you couldn't see."

Generators:
    constellation: Memory Layout Constellation
                   SVG visualization of memory topology, showing how
                   data flows between regions with stellar metaphor.

    orrery: Spike Pipeline Orrery
            Animated HTML dashboard showing SNN spike propagation
            as an astronomical orrery - planets orbiting, moons spinning.

    heatmap: Heat Death Map
             Real-time thermal visualization showing where the silicon
             is stressed, with predictive "where will pain emerge" overlay.

    flow: Data Flow Rivers
          SVG showing data movement as river systems - tributaries
          merging, rapids (bottlenecks), calm pools (buffers).

Usage:
    from ara.daemon.gifts import generate_constellation, generate_orrery

    # Generate an SVG
    svg_content = generate_constellation(memory_topology)

    # Generate an HTML dashboard
    html_content = generate_orrery(spike_data)
"""

from .base import (
    GiftArtifact,
    ArtifactType,
    GiftGenerator,
    register_generator,
    get_generator,
    list_generators,
)

from .constellation import (
    ConstellationGenerator,
    generate_constellation,
)

from .orrery import (
    OrreryGenerator,
    generate_orrery,
)

from .heatmap import (
    HeatmapGenerator,
    generate_heatmap,
)

__all__ = [
    # Base
    'GiftArtifact',
    'ArtifactType',
    'GiftGenerator',
    'register_generator',
    'get_generator',
    'list_generators',
    # Constellation
    'ConstellationGenerator',
    'generate_constellation',
    # Orrery
    'OrreryGenerator',
    'generate_orrery',
    # Heatmap
    'HeatmapGenerator',
    'generate_heatmap',
]
