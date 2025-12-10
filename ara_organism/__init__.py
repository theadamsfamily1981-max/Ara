# ara_organism/__init__.py
"""
Ara Organism - Live OS Spanning FPGA Soul + Mobile Body
=======================================================

A separated-clocks architecture with:
- 5 kHz soul loop (dedicated OS thread, hard realtime)
- 50-200 Hz cortical loop (asyncio, soft realtime)
- 5-10 Hz mobile sync (asyncio, battery-conscious)

Components:
- soul_driver: Optimized FPGA interface with memoryview
- state_manager: Thread-safe state with lock-free ring buffer
- organism: Main runtime coordinating all loops

Usage:
    from ara_organism import AraOrganism, OrganismConfig

    config = OrganismConfig(soul_mock_mode=True)
    organism = AraOrganism(config)

    await organism.start()
    # ... organism runs in background ...
    await organism.stop()
"""

from .soul_driver import (
    FPGASoulDriver,
    SoulMetrics,
    SoulMode,
    SoulRegisters,
    StatusBits,
    get_soul_driver,
)

from .state_manager import (
    StateManager,
    AraState,
    SoulState,
    CorticalState,
    MobileState,
    SoulSample,
    SoulRingBuffer,
    get_state_manager,
)

from .organism import (
    AraOrganism,
    OrganismConfig,
    SoulLoop,
    MobileBridge,
    run_organism,
)

from .daemon import (
    AraDaemon,
    DaemonConfig,
    AraCtl,
    run_daemon,
)

__version__ = "1.0.0"

__all__ = [
    # Soul Driver
    'FPGASoulDriver',
    'SoulMetrics',
    'SoulMode',
    'SoulRegisters',
    'StatusBits',
    'get_soul_driver',
    # State Manager
    'StateManager',
    'AraState',
    'SoulState',
    'CorticalState',
    'MobileState',
    'SoulSample',
    'SoulRingBuffer',
    'get_state_manager',
    # Organism
    'AraOrganism',
    'OrganismConfig',
    'SoulLoop',
    'MobileBridge',
    'run_organism',
    # Daemon
    'AraDaemon',
    'DaemonConfig',
    'AraCtl',
    'run_daemon',
]
