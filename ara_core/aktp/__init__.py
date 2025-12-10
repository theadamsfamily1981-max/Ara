"""
A-KTP: Allegory-based Knowledge Transfer Protocol
===================================================

NAIKT-2025 - Neurosymbolic framework for ethical inter-agent
knowledge transfer using allegorical narratives.

Components:
- AGM (AllegoryGenerator): Maps problems to universal story structures
- ACT (AdversarialConstraintTransfer): Refines constraints through 5-bot debate
- DRSE (DynamicRewardShaper): Shapes MORL rewards based on constraints
- AKTPProtocol: Meta-loop orchestrator (AGM→ACT→DRSE)

Key features:
- Zero-shot transfer via structure-mapping allegories
- Ethical convergence through adversarial debate
- Progress Toward Ethical (PTE) tracking: 0.2 → 5.4 over 3 cycles
- Hypothetical/bias flagging for unverified domains

Usage:
    from ara_core.aktp import run_aktp

    result = run_aktp(
        problem="Should we migrate to microservices?",
        domain="infrastructure"
    )

    print(f"Recommendation: {result.final_recommendation}")
    print(f"Converged: {result.converged} (PTE: {result.total_pte})")
"""

from .allegory import (
    AllegoryArchetype,
    StructuralMapping,
    Allegory,
    AllegoryGenerator,
    generate_allegory,
    get_template_allegory,
    ALLEGORY_TEMPLATES,
)

from .bots import (
    BotPersonality,
    BotConfig,
    BotResponse,
    DebateRound,
    DebateBot,
    DebatePanel,
    BOT_CONFIGS,
    create_debate_panel,
    quick_debate,
)

from .constraints import (
    ConstraintType,
    Constraint,
    ConstraintSet,
    AdversarialConstraintTransfer,
    refine_constraints,
)

from .rewards import (
    RewardShapingConfig,
    ShapedReward,
    PolicyUpdate,
    DynamicRewardShaper,
    shape_rewards,
)

from .protocol import (
    CycleResult,
    ProtocolResult,
    AKTPProtocol,
    run_aktp,
    transfer_knowledge,
)


__all__ = [
    # Allegory
    "AllegoryArchetype",
    "StructuralMapping",
    "Allegory",
    "AllegoryGenerator",
    "generate_allegory",
    "get_template_allegory",
    "ALLEGORY_TEMPLATES",

    # Bots
    "BotPersonality",
    "BotConfig",
    "BotResponse",
    "DebateRound",
    "DebateBot",
    "DebatePanel",
    "BOT_CONFIGS",
    "create_debate_panel",
    "quick_debate",

    # Constraints
    "ConstraintType",
    "Constraint",
    "ConstraintSet",
    "AdversarialConstraintTransfer",
    "refine_constraints",

    # Rewards
    "RewardShapingConfig",
    "ShapedReward",
    "PolicyUpdate",
    "DynamicRewardShaper",
    "shape_rewards",

    # Protocol
    "CycleResult",
    "ProtocolResult",
    "AKTPProtocol",
    "run_aktp",
    "transfer_knowledge",
]
