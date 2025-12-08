"""
Ara Network Encoder - Flow and Node HV Encoding
===============================================

Encodes network flows and node states into hypervectors for the HTC.

Every flow becomes:
    H_flow = ROLE_NET ⊕ H_SRC_NODE ⊕ H_DST_NODE ⊕ H_SERVICE ⊕ H_LATENCY_BIN ⊕ H_RATE_BIN

Every node state becomes:
    H_node = ROLE_NET_NODE ⊕ H_NODE_ID ⊕ H_CPU_BIN ⊕ H_GPU_BIN ⊕ H_TEMP_BIN ⊕ H_ERR_BIN

These HVs are fed into the HTC as part of the moment construction,
allowing the soul to learn network patterns.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from ara.io.types import HDInputEvent, IOChannel, HV
from ara.hd.ops import bind, bundle, DIM
from ara.hd.vocab import get_vocab


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FlowData:
    """Network flow data for HV encoding."""
    src_node: str
    dst_node: str
    service: str
    protocol: str = "tcp"
    src_port: int = 0
    dst_port: int = 0
    bytes_per_sec: float = 0.0
    packets_per_sec: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    flow_id: Optional[str] = None


@dataclass
class NodeState:
    """Node state data for HV encoding."""
    node_id: str
    node_type: str = "generic"  # "fpga", "gpu", "cpu", "storage"
    cpu_load: float = 0.0       # 0-1
    gpu_load: float = 0.0       # 0-1
    fpga_load: float = 0.0      # 0-1
    memory_used: float = 0.0    # 0-1
    max_temp: float = 40.0      # Celsius
    power_watts: float = 0.0
    error_rate: float = 0.0
    uptime_hours: float = 0.0


# =============================================================================
# Binning Functions
# =============================================================================

def bin_latency(latency_ms: float) -> str:
    """Bin latency into discrete levels."""
    if latency_ms < 1:
        return "MINIMAL"
    if latency_ms < 10:
        return "LOW"
    if latency_ms < 50:
        return "MED"
    if latency_ms < 200:
        return "HIGH"
    return "CRITICAL"


def bin_rate(mbps: float) -> str:
    """Bin throughput rate."""
    if mbps < 1:
        return "MINIMAL"
    if mbps < 100:
        return "LOW"
    if mbps < 1000:
        return "MED"
    if mbps < 10000:
        return "HIGH"
    return "EXTREME"


def bin_error_rate(err: float) -> str:
    """Bin error rate (0-1)."""
    if err < 0.001:
        return "ZERO"
    if err < 0.01:
        return "LOW"
    if err < 0.05:
        return "MED"
    if err < 0.1:
        return "HIGH"
    return "CRITICAL"


def bin_load(load: float) -> str:
    """Bin CPU/GPU/FPGA load (0-1)."""
    if load < 0.1:
        return "MINIMAL"
    if load < 0.3:
        return "LOW"
    if load < 0.6:
        return "MED"
    if load < 0.85:
        return "HIGH"
    return "CRITICAL"


def bin_temp(temp_c: float) -> str:
    """Bin temperature (Celsius)."""
    if temp_c < 40:
        return "LOW"
    if temp_c < 55:
        return "MED"
    if temp_c < 70:
        return "HIGH"
    if temp_c < 85:
        return "CRITICAL"
    return "EXTREME"


# =============================================================================
# Encoding Functions
# =============================================================================

def encode_flow(flow: FlowData) -> HDInputEvent:
    """
    Encode a network flow into an HDInputEvent.

    H_flow = ROLE_NET ⊕ H_SRC ⊕ H_DST ⊕ H_SERVICE ⊕ H_LATENCY ⊕ H_RATE ⊕ H_ERR
    """
    vocab = get_vocab()

    # Get component HVs
    h_src = vocab.custom("node", flow.src_node)
    h_dst = vocab.custom("node", flow.dst_node)
    h_service = vocab.custom("service", flow.service)
    h_protocol = vocab.custom("protocol", flow.protocol)

    # Bin continuous values
    h_latency = vocab.bin(bin_latency(flow.latency_ms))
    h_rate = vocab.bin(bin_rate(flow.bytes_per_sec / 1e6))  # Convert to Mbps
    h_err = vocab.bin(bin_error_rate(flow.error_rate))

    # Bind components with role markers
    components = [
        bind(vocab.feature("SRC_NODE"), h_src),
        bind(vocab.feature("DST_NODE"), h_dst),
        bind(vocab.feature("SERVICE"), h_service),
        bind(vocab.feature("PROTOCOL"), h_protocol),
        bind(vocab.feature("LATENCY"), h_latency),
        bind(vocab.feature("RATE"), h_rate),
        bind(vocab.feature("ERR_RATE"), h_err),
    ]

    # Bundle and bind with network role
    h_flow = bundle(components)
    h_bound = bind(vocab.role("NETWORK"), h_flow)

    # Compute priority based on characteristics
    priority = 0.5
    if bin_latency(flow.latency_ms) in ["HIGH", "CRITICAL"]:
        priority += 0.2
    if bin_error_rate(flow.error_rate) in ["HIGH", "CRITICAL"]:
        priority += 0.3

    return HDInputEvent(
        channel=IOChannel.NETWORK,
        role="ROLE_NET_FLOW",
        meta={
            "src_node": flow.src_node,
            "dst_node": flow.dst_node,
            "service": flow.service,
            "latency_ms": flow.latency_ms,
            "bytes_per_sec": flow.bytes_per_sec,
            "error_rate": flow.error_rate,
            "flow_id": flow.flow_id,
        },
        hv=h_bound,
        priority=min(1.0, priority),
        source_id=flow.flow_id,
    )


def encode_node_state(state: NodeState) -> HDInputEvent:
    """
    Encode a node state into an HDInputEvent.

    H_node = ROLE_NET_NODE ⊕ H_ID ⊕ H_TYPE ⊕ H_CPU ⊕ H_GPU ⊕ H_TEMP ⊕ H_ERR
    """
    vocab = get_vocab()

    # Get component HVs
    h_node_id = vocab.custom("node", state.node_id)
    h_node_type = vocab.custom("node_type", state.node_type)

    # Bin continuous values
    h_cpu = vocab.bin(bin_load(state.cpu_load))
    h_gpu = vocab.bin(bin_load(state.gpu_load))
    h_fpga = vocab.bin(bin_load(state.fpga_load))
    h_mem = vocab.bin(bin_load(state.memory_used))
    h_temp = vocab.bin(bin_temp(state.max_temp))
    h_err = vocab.bin(bin_error_rate(state.error_rate))

    # Bind components with feature markers
    components = [
        bind(vocab.feature("NODE_ID"), h_node_id),
        bind(vocab.feature("NODE_TYPE"), h_node_type),
        bind(vocab.feature("CPU_LOAD"), h_cpu),
        bind(vocab.feature("GPU_UTIL"), h_gpu),
        bind(vocab.feature("FPGA_LOAD"), h_fpga),
        bind(vocab.feature("MEMORY_USED"), h_mem),
        bind(vocab.feature("TEMP"), h_temp),
        bind(vocab.feature("ERR_RATE"), h_err),
    ]

    # Bundle and bind with node role
    h_node = bundle(components)
    h_bound = bind(vocab.role("PROPRIOCEPTION"), h_node)  # Self-monitoring

    # Compute priority based on health
    priority = 0.5
    if bin_temp(state.max_temp) in ["HIGH", "CRITICAL", "EXTREME"]:
        priority += 0.3
    if bin_error_rate(state.error_rate) in ["HIGH", "CRITICAL"]:
        priority += 0.2

    return HDInputEvent(
        channel=IOChannel.NETWORK,
        role="ROLE_NET_NODE",
        meta={
            "node_id": state.node_id,
            "node_type": state.node_type,
            "cpu_load": state.cpu_load,
            "gpu_load": state.gpu_load,
            "fpga_load": state.fpga_load,
            "max_temp": state.max_temp,
            "error_rate": state.error_rate,
        },
        hv=h_bound,
        priority=min(1.0, priority),
        source_id=state.node_id,
    )


def encode_flow_batch(flows: List[FlowData]) -> List[HDInputEvent]:
    """Encode a batch of flows."""
    return [encode_flow(f) for f in flows]


def encode_node_batch(states: List[NodeState]) -> List[HDInputEvent]:
    """Encode a batch of node states."""
    return [encode_node_state(s) for s in states]


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'FlowData',
    'NodeState',
    'encode_flow',
    'encode_node_state',
    'encode_flow_batch',
    'encode_node_batch',
    # Binning functions
    'bin_latency',
    'bin_rate',
    'bin_error_rate',
    'bin_load',
    'bin_temp',
]
