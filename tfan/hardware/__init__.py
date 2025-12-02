"""
TF-A-N Hardware Module

Hardware-aware generative cognition components:
- Autosynth: Hardware generative loop for FPGA acceleration
"""

from .autosynth import (
    BottleneckType,
    BottleneckSeverity,
    ProposalStatus,
    PerformanceMetrics,
    Bottleneck,
    BottleneckDetector,
    HLSProposal,
    HLSTemplates,
    KernelProposer,
    ProposalVerifier,
    DeploymentManager,
    AutosynthController,
    create_autosynth_controller,
    analyze_bottlenecks,
    propose_kernel
)

__all__ = [
    "BottleneckType",
    "BottleneckSeverity",
    "ProposalStatus",
    "PerformanceMetrics",
    "Bottleneck",
    "BottleneckDetector",
    "HLSProposal",
    "HLSTemplates",
    "KernelProposer",
    "ProposalVerifier",
    "DeploymentManager",
    "AutosynthController",
    "create_autosynth_controller",
    "analyze_bottlenecks",
    "propose_kernel"
]
