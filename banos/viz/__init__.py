"""BANOS Visualization - High-Bandwidth Binary Streaming for Ara's Visual Cortex.

This module provides the visualization infrastructure for Ara's somatic state,
solving two architectural issues:

1. Visualization Lie - Binary texture streaming replaces slow JS injection
2. Audio-Visual Synesthesia - Optical flow from face tracking advects the quantum field

Components:
    SomaticStreamServer: HTTP server that streams binary somatic data
    SomaticDataStore: Shared state for spike, flow, and entropy field
    OpticalFlowTracker: OpenCV-based face motion tracking for synesthesia

Usage:
    from banos.viz import SomaticStreamServer, OpticalFlowTracker

    # Start the somatic stream server
    server = SomaticStreamServer(port=8999)
    server.start()

    # Update from BANOS state (32-bit pain_level)
    server.update_spike(pain_level / 4294967295.0)

    # Track optical flow from video frames
    tracker = OpticalFlowTracker()
    flow_x, flow_y = tracker.process_frame(frame)
    server.update_flow(flow_x, flow_y)

    # Open soul_quantum.html in browser to visualize
"""

from .somatic_server import (
    SomaticStreamServer,
    SomaticDataStore,
    OpticalFlowTracker,
    SomaticRequestHandler,
)

__all__ = [
    "SomaticStreamServer",
    "SomaticDataStore",
    "OpticalFlowTracker",
    "SomaticRequestHandler",
]
