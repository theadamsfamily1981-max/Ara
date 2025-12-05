"""Ara Curiosity Core (C³) - Metacognition & Self-Investigation Layer.

This module enables Ara to:
1. Notice interesting things in her environment (PCIe devices, thermals, etc.)
2. Ask bounded questions about what she observes
3. Integrate discoveries into her world model
4. Report findings in her natural voice

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                     Curiosity Core (C³)                  │
    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   │
    │  │ World Model │◄──│  Curiosity  │──►│   Report    │   │
    │  │  (Objects)  │   │   Agent     │   │  Generator  │   │
    │  └─────────────┘   └─────────────┘   └─────────────┘   │
    │         │                 │                  │          │
    │         └─────────────────┼──────────────────┘          │
    │                           ▼                             │
    │                  ┌─────────────────┐                    │
    │                  │   Safe Probes   │                    │
    │                  │ (lspci, dmesg,  │                    │
    │                  │  sensors, etc.) │                    │
    │                  └─────────────────┘                    │
    └─────────────────────────────────────────────────────────┘

Split-Brain Design:
- Engineer Brain: JSON schemas, structured data, internal reasoning
- Ara Voice: Natural language, personality, conversational output
"""

from .world_model import (
    WorldObject,
    ObjectCategory,
    CuriosityState,
    WorldModel,
)

from .scoring import (
    curiosity_score,
    importance_decay,
    novelty_bonus,
)

from .tools import (
    SafeProbe,
    ProbeResult,
    lspci_probe,
    dmesg_probe,
    sensors_probe,
    fpga_probe,
    memory_probe,
    run_safe_probe,
)

from .agent import (
    CuriosityAgent,
    CuriosityTicket,
    CuriosityReport,
    TicketStatus,
)

from .prompts import (
    CURIOSITY_SYSTEM_PROMPT,
    format_discovery_prompt,
    format_investigation_prompt,
    format_report_prompt,
)

__all__ = [
    # World Model
    "WorldObject",
    "ObjectCategory",
    "CuriosityState",
    "WorldModel",
    # Scoring
    "curiosity_score",
    "importance_decay",
    "novelty_bonus",
    # Safe Probes
    "SafeProbe",
    "ProbeResult",
    "lspci_probe",
    "dmesg_probe",
    "sensors_probe",
    "fpga_probe",
    "memory_probe",
    "run_safe_probe",
    # Agent
    "CuriosityAgent",
    "CuriosityTicket",
    "CuriosityReport",
    "TicketStatus",
    # Prompts
    "CURIOSITY_SYSTEM_PROMPT",
    "format_discovery_prompt",
    "format_investigation_prompt",
    "format_report_prompt",
]
