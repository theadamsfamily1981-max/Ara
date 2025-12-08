"""
Binary Neural Network Package
==============================

1-bit neurons: ridiculously wide, cheap correlator layers.

Modules:
    core: XNOR+popcount primitives, BinaryDense layer
    frontend: BinaryFrontEnd for SNN/T-FAN integration
    memory: Binary associative memory (Hamming distance search)

Quick Start:
    from ara.neuro.binary import BinaryFrontEnd, BinaryMemory

    # Create front-end
    frontend = BinaryFrontEnd(input_dim=1024, output_dim=512)

    # Process input
    code, sketch = frontend(x)

    # Store in associative memory
    memory = BinaryMemory(code_dim=512)
    memory.store(code, label="state_42")

    # Later: recall similar patterns
    matches = memory.query(new_code, k=5)
"""

from ara.neuro.binary.core import (
    pack_bits,
    xnor_popcount,
    pack_bits_numpy,
    unpack_bits_numpy,
    xnor_popcount_numpy,
    TORCH_AVAILABLE,
)

from ara.neuro.binary.memory import (
    BinaryMemory,
    MoodCodeMemory,
    EpisodicIndex,
    MemoryEntry,
    QueryResult,
)

from ara.neuro.binary.frontend import (
    FrontEndConfig,
    BinaryFrontEndNumpy,
)

from ara.neuro.binary.holographic import (
    HolographicConfig,
    HolographicProcessor,
    get_holographic_processor,
)

from ara.neuro.binary.ensemble import (
    LaneRole,
    EnsembleConfig,
    EnsembleJob,
    EnsembleResult,
    EnsembleChoir,
    FieldMonitor,
    get_ensemble_choir,
)

__all__ = [
    # Core primitives
    'pack_bits',
    'xnor_popcount',
    'pack_bits_numpy',
    'unpack_bits_numpy',
    'xnor_popcount_numpy',
    'TORCH_AVAILABLE',

    # Memory
    'BinaryMemory',
    'MoodCodeMemory',
    'EpisodicIndex',
    'MemoryEntry',
    'QueryResult',

    # Frontend (numpy fallback)
    'FrontEndConfig',
    'BinaryFrontEndNumpy',

    # Holographic (HDC + SNN unified)
    'HolographicConfig',
    'HolographicProcessor',
    'get_holographic_processor',

    # Ensemble Choir (Iteration 31)
    'LaneRole',
    'EnsembleConfig',
    'EnsembleJob',
    'EnsembleResult',
    'EnsembleChoir',
    'FieldMonitor',
    'get_ensemble_choir',
]

# Conditionally export torch classes
if TORCH_AVAILABLE:
    from ara.neuro.binary.core import (
        pack_bits_torch,
        xnor_popcount_torch,
        BinaryDense,
        BinaryConv2d,
    )
    from ara.neuro.binary.frontend import (
        BinaryFrontEnd,
        BinaryEncoder,
        TopologicalSketch,
        SensorySketcher,
        SignActivation,
    )

    __all__.extend([
        'pack_bits_torch',
        'xnor_popcount_torch',
        'BinaryDense',
        'BinaryConv2d',
        'BinaryFrontEnd',
        'BinaryEncoder',
        'TopologicalSketch',
        'SensorySketcher',
        'SignActivation',
    ])
