"""
Ara I/O - Unified HD Input/Output Layer
=======================================

Treats all I/O channels as HD cognitive interfaces:
- SENSORIUM: Physical sensors, audio, video, system telemetry
- NETWORK: Flows, nodes, policies (LAN cortex)
- UI: Panels, gaze, clicks (visual cortex)
- TASK: Job state, tools, code execution
- INTERNAL: MindReader, Teleology, introspection

Every channel:
- Emits HDInputEvents into the HTC (moment construction)
- Receives HDOutputHints from the HTC (affect, focus, policies)

This makes graphics and networking first-class cognitive organs,
not separate utility subsystems.

Usage:
    from ara.io import HDInputEvent, HDOutputHint, IOChannel

    # Input: network flow encoded as HV
    event = HDInputEvent(
        channel=IOChannel.NETWORK,
        role="ROLE_NET",
        meta={"src": "fpga-01", "dst": "gpu-02"},
        hv=h_flow
    )

    # Output: affect hint to avatar
    hint = HDOutputHint(
        channel=IOChannel.UI,
        kind="AFFECT",
        meta={},
        payload={"valence": 0.7, "arousal": 0.3}
    )
"""

from .types import (
    IOChannel,
    HDInputEvent,
    HDOutputHint,
    HV,
)

from .dispatcher import (
    IODispatcher,
    get_dispatcher,
)

__all__ = [
    'IOChannel',
    'HDInputEvent',
    'HDOutputHint',
    'HV',
    'IODispatcher',
    'get_dispatcher',
]
