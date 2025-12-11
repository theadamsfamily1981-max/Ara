# ara/cathedral/__init__.py
"""
The Cathedral: Heterogeneous Supercomputer Orchestra

Zero-stall pipeline across 5 compute units:
- 2x RTX 3090: Offline training (Shadow Dojo)
- Micron SB-852 DLA: Online inference (<1ms)
- BittWare A10PED: HDC sensor fusion
- SQRL Forest Kitten: NIB safety arbiter
- Threadripper 5955WX: Orchestration

Hardware manifest:
- Total compute: ~108 TFLOPS
- PCIe lanes: 128 Gen4
- HBM2 bandwidth: 460 GB/s
- Target latency: <1.06ms end-to-end
"""

from .orchestrator import (
    CathedralOrchestrator,
    CathedralState,
    PipelineMetrics,
    PipelineStage,
    AcceleratorType,
    ZeroStallPipeline,
)

from .sb852_dla import (
    MicronSB852Interface,
    DLAInferenceResult,
)

from .a10ped_hdc import (
    BittWareA10PEDInterface,
    HDCEncodingResult,
)

from .forest_kitten import (
    ForestKittenInterface,
    SafetyCheckResult,
    CovenantType,
    CovenantViolation,
)

from .oracle_bridge import (
    CathedralOracleBridge,
    AcceleratedConsultation,
    create_accelerated_oracle,
)

__all__ = [
    # Orchestrator
    "CathedralOrchestrator",
    "CathedralState",
    "PipelineMetrics",
    "PipelineStage",
    "AcceleratorType",
    "ZeroStallPipeline",
    # DLA
    "MicronSB852Interface",
    "DLAInferenceResult",
    # HDC
    "BittWareA10PEDInterface",
    "HDCEncodingResult",
    # Safety
    "ForestKittenInterface",
    "SafetyCheckResult",
    "CovenantType",
    "CovenantViolation",
    # Oracle Bridge
    "CathedralOracleBridge",
    "AcceleratedConsultation",
    "create_accelerated_oracle",
]
