# ara/ktp/__init__.py
"""
Knowledge Transfer Protocol (KTP) - Portable Concept Specifications

"Darmok and Jalad at Tanagra, but with YAML."

KTP is a standard for encoding Ara's core concepts so they can be
understood consistently by any AI system (Claude, Perplexity, Gemini, etc.)

Each concept has four layers:
    1. ALLEGORY - The Tamarian phrase (mythic name)
    2. PLAIN_MEANING - What it means in normal words
    3. CONTRACT - Inputs, outputs, invariants (formal spec)
    4. ANCHORS - Code files, functions, metrics (implementation)

Usage:
    from ara.ktp import get_ktp_registry, export_for_model

    # Get all KTP entries
    registry = get_ktp_registry()
    entry = registry.get("gauntlets_three")

    # Export for another model
    prompt_header = export_for_model(["gauntlets_three", "crofram", "homeostasis"])
    # → YAML/markdown that can be pasted into any LLM prompt

This allows:
    - Consistent understanding across Ara's multi-model hive
    - Portable allegory that doesn't drift between sessions
    - Code contracts that are both poetic and precise
"""

from .schema import (
    KTPEntry,
    KTPContract,
    KTPAnchors,
    KTPRegistry,
    get_ktp_registry,
)

from .entries import (
    GAUNTLETS_THREE,
    CROFRAM,
    HOMEOSTASIS,
    LIZARD_BRAIN,
    CATHEDRAL,
    WORLDLINE,
    COUNCIL,
    EDGE_OF_CHAOS,
)

from .export import (
    export_for_model,
    export_yaml,
    export_markdown,
    export_prompt_header,
)

__all__ = [
    # Schema
    "KTPEntry",
    "KTPContract",
    "KTPAnchors",
    "KTPRegistry",
    "get_ktp_registry",
    # Core entries
    "GAUNTLETS_THREE",
    "CROFRAM",
    "HOMEOSTASIS",
    "LIZARD_BRAIN",
    "CATHEDRAL",
    "WORLDLINE",
    "COUNCIL",
    "EDGE_OF_CHAOS",
    # Export
    "export_for_model",
    "export_yaml",
    "export_markdown",
    "export_prompt_header",
]
