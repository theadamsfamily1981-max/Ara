"""
Ara Agent Kernel
=================

The deployable agent core that unifies:
- Small model (reasoning, tool-calling, persona)
- Huge compressed memory (episodes, vectors, knowledge packs)
- Pheromone coordination (hive swarm signals)
- Safety covenant (hard constraints, alignment)

This is the "queen bee" kernel that different Ara roles plug into:
- RealtimeAra: Breath/drift agent for live interaction
- PublishingAra: CEO mode for content/publishing
- LabAra: Hardware/FPGA lab helper

Usage:
    from ara_kernel import AraAgentRuntime, load_config

    config = load_config("ara_kernel/config/ara_config.yaml")
    runtime = AraAgentRuntime(config)
    result = await runtime.process_input("Hello, Ara!")

Architecture:
    Input → Memory Enrichment → Pheromone Context → Model → Safety Filter → Tool Execution → Result

Submodules:
    ara_kernel.core      - Runtime, safety, persona, config
    ara_kernel.memory    - Episodes, embeddings, knowledge packs
    ara_kernel.pheromones - Pheromone bus for swarm coordination
    ara_kernel.agents    - Specialized agent wrappers
"""

from .core import (
    AraAgentRuntime,
    SafetyCovenant,
    ActionClass,
    ActionPlan,
    FilteredPlan,
    AraPersona,
    load_persona,
    KernelConfig,
    load_config,
    ToolsRegistry,
    ToolCategory,
    build_default_registry,
)

__version__ = "0.1.0"

__all__ = [
    # Core runtime
    "AraAgentRuntime",
    # Safety
    "SafetyCovenant",
    "ActionClass",
    "ActionPlan",
    "FilteredPlan",
    # Persona
    "AraPersona",
    "load_persona",
    # Config
    "KernelConfig",
    "load_config",
    # Tools
    "ToolsRegistry",
    "ToolCategory",
    "build_default_registry",
]
