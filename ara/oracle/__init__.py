# ara/oracle/__init__.py
"""
The Oracle of Delphi: Pythia Triad Architecture

Three specialized Oracles that deliberate together:

- ORACLE ALPHA (The Visionary): Long-horizon imagination (1000+ steps)
  Uses QHDC particle ensemble on GPU 1, massively parallel futures

- ORACLE BETA (The Analyst): Real-time inference (<1ms latency)
  Uses TRC reversible dynamics on GPU 2 + BittWare FPGA

- ORACLE GAMMA (The Arbiter): NIB covenant enforcement
  Uses Forest Kitten FPGA for hardware-locked safety circuits

Hardware allocation:
- GPU 1 (3090 #1): Oracle Alpha + world model
- GPU 2 (3090 #2): Oracle Beta + inference
- BittWare A10P FPGA: HDC acceleration for Beta
- Forest Kitten FPGA: Safety circuits for Gamma
- Threadripper cores 0-21: Alpha particle management
- Threadripper cores 22-42: Beta FPGA communication
- Threadripper cores 43-63: Gamma governance
"""

from .alpha_visionary import (
    OracleAlpha,
    MassiveParticleEnsemble,
    Prophecy,
)

from .beta_analyst import (
    OracleBeta,
    BittWareFPGAInterface,
    FastWorldModel,
    NVMeExperienceReplay,
)

from .gamma_arbiter import (
    OracleGamma,
    ForestKittenSafetyCore,
    CovenantViolation,
)

from .pythia_triad import (
    PythiaTriad,
    OracleConsensus,
)

__all__ = [
    # Alpha (Visionary)
    "OracleAlpha",
    "MassiveParticleEnsemble",
    "Prophecy",
    # Beta (Analyst)
    "OracleBeta",
    "BittWareFPGAInterface",
    "FastWorldModel",
    "NVMeExperienceReplay",
    # Gamma (Arbiter)
    "OracleGamma",
    "ForestKittenSafetyCore",
    "CovenantViolation",
    # Orchestrator
    "PythiaTriad",
    "OracleConsensus",
]
