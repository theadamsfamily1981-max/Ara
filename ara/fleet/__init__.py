"""
Fleet Topology - The Skeleton of the Cathedral
===============================================

Hardware archetypes that make the organism robust:

    WATCHER (Medic)
        Out-of-band safety controller. Can kill power, force reboots.
        Stays alive when everything else is hard-locked.

    ARCHIVIST
        NAS with snapshots. Memory that survives catastrophic experiments.
        Read-heavy, writes via well-defined backup pipelines.

    BRAINSTEM
        Tiny always-on orchestrator. Holds global view of fleet.
        Never runs heavy training or wild experiments.

    SHIELD
        Network tap/observer. Reads metrics from Juniper.
        Never pushes config without human-signed approval.

    POWER_SPINE
        UPS + Smart PDU. Turns random power failures into
        predictable, survivable events.

    INTERN_CLUSTER
        Sacrificial VM host. All wild experiments go here first.
        Fast to rebuild, expendable.

    CRYPTO_KEY
        Hardware security key. Gates ultra-critical actions.
        Requires physical tap for signed tokens.

    SENSOR_HUB
        Environmental sensors. Richer context → fewer dumb decisions.

These wrap the cathedral in:
- A skeleton (structure)
- An immune system (safety)
- A black box recorder (memory)
"""

from .topology import (
    FleetRole,
    AuthLevel,
    NodeCapability,
    FleetNode,
    FleetTopology,
    create_default_topology,
)
from .watcher import (
    WatcherState,
    PowerAction,
    WatcherEvent,
    Watcher,
    create_watcher,
)
from .brainstem import (
    JobState,
    ScheduledJob,
    Brainstem,
    create_brainstem,
)
from .safety import (
    ActionSeverity,
    SignedAction,
    ApprovalRequest,
    CryptoGatekeeper,
    SafetyPolicy,
    create_safety_policy,
)
from .power import (
    PowerState,
    OutletState,
    PowerEvent,
    SmartPDU,
    UPS,
    PowerSpine,
    create_power_spine,
)

__all__ = [
    # Topology
    'FleetRole',
    'AuthLevel',
    'NodeCapability',
    'FleetNode',
    'FleetTopology',
    'create_default_topology',
    # Watcher
    'WatcherState',
    'PowerAction',
    'WatcherEvent',
    'Watcher',
    'create_watcher',
    # Brainstem
    'JobState',
    'ScheduledJob',
    'Brainstem',
    'create_brainstem',
    # Safety
    'ActionSeverity',
    'SignedAction',
    'ApprovalRequest',
    'CryptoGatekeeper',
    'SafetyPolicy',
    'create_safety_policy',
    # Power
    'PowerState',
    'OutletState',
    'PowerEvent',
    'SmartPDU',
    'UPS',
    'PowerSpine',
    'create_power_spine',
]
