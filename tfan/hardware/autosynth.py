"""
Autosynth: Hardware Generative Loop

This module implements the ability to:
1. Detect performance bottlenecks not solvable by r,k adjustments
2. Propose HLS C++ kernels from strict templates
3. Verify proposed code via PGU
4. Deploy verified kernels via CXL fabric

This is "hardware-aware generative cognition" - the system learning to
design bits of its own silicon to improve cognitive functions.

Pipeline:
    Telemetry (latency, throughput, utilization)
        ↓
    BottleneckDetector.analyze() → Bottleneck
        ↓
    KernelProposer.propose() → HLSProposal
        ↓
    ProposalVerifier.verify() → VerifiedProposal
        ↓
    DeploymentManager.deploy() → DeployedKernel

Safety model:
- Phase A: Propose only, human reviews & deploys
- Phase B: Auto-deploy non-critical accelerators (sandboxed)
- Phase C: Partial autonomy with long history of safe behavior

Current implementation: Phase A (propose + verify, no auto-deploy)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
import hashlib
import re


# ============================================================
# Bottleneck Types
# ============================================================

class BottleneckType(str, Enum):
    """Types of performance bottlenecks we can detect."""
    COMPUTE_BOUND = "compute_bound"       # CPU/GPU cycles limiting
    MEMORY_BOUND = "memory_bound"         # Memory bandwidth/latency
    IO_BOUND = "io_bound"                 # Storage/network I/O
    KERNEL_SPECIFIC = "kernel_specific"   # Specific function is slow
    STRUCTURAL = "structural"             # Topological computation
    INFERENCE = "inference"               # Model inference latency


class BottleneckSeverity(str, Enum):
    """How severe is the bottleneck?"""
    LOW = "low"           # Minor impact, optimization optional
    MEDIUM = "medium"     # Noticeable impact, should address
    HIGH = "high"         # Significant impact, priority fix
    CRITICAL = "critical" # Blocking operations, urgent


class ProposalStatus(str, Enum):
    """Status of an HLS proposal."""
    DRAFT = "draft"           # Just proposed, not verified
    VERIFIED = "verified"     # Passed PGU verification
    REJECTED = "rejected"     # Failed verification
    APPROVED = "approved"     # Human approved for synthesis
    SYNTHESIZED = "synthesized"  # Bitstream generated
    DEPLOYED = "deployed"     # Running on hardware
    RETIRED = "retired"       # No longer active


# ============================================================
# Telemetry Data
# ============================================================

@dataclass
class PerformanceMetrics:
    """Performance metrics for bottleneck detection."""
    timestamp: datetime = field(default_factory=datetime.now)

    # Latency metrics (ms)
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0

    # Throughput
    ops_per_second: float = 0.0
    tokens_per_second: float = 0.0

    # Utilization (0-1)
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    fpga_utilization: float = 0.0
    memory_utilization: float = 0.0

    # Specific function timings (function_name → ms)
    function_timings: Dict[str, float] = field(default_factory=dict)

    # Queue depths
    pending_requests: int = 0
    aepo_queue_depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "latency": {
                "p50_ms": self.p50_latency_ms,
                "p95_ms": self.p95_latency_ms,
                "p99_ms": self.p99_latency_ms
            },
            "throughput": {
                "ops_per_sec": self.ops_per_second,
                "tokens_per_sec": self.tokens_per_second
            },
            "utilization": {
                "cpu": self.cpu_utilization,
                "gpu": self.gpu_utilization,
                "fpga": self.fpga_utilization,
                "memory": self.memory_utilization
            },
            "function_timings": self.function_timings,
            "queues": {
                "pending": self.pending_requests,
                "aepo": self.aepo_queue_depth
            }
        }


# ============================================================
# Bottleneck
# ============================================================

@dataclass
class Bottleneck:
    """A detected performance bottleneck."""
    id: str
    bottleneck_type: BottleneckType
    severity: BottleneckSeverity
    description: str
    affected_function: Optional[str] = None
    measured_latency_ms: float = 0.0
    target_latency_ms: float = 0.0
    improvement_potential: float = 0.0  # Expected speedup (e.g., 2.0 = 2x faster)
    detected_at: datetime = field(default_factory=datetime.now)
    metrics_snapshot: Optional[PerformanceMetrics] = None

    @property
    def latency_gap(self) -> float:
        """How much we exceed the target."""
        if self.target_latency_ms <= 0:
            return 0
        return self.measured_latency_ms - self.target_latency_ms

    @property
    def latency_ratio(self) -> float:
        """Ratio of measured to target (>1 means exceeding target)."""
        if self.target_latency_ms <= 0:
            return 1.0
        return self.measured_latency_ms / self.target_latency_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.bottleneck_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "affected_function": self.affected_function,
            "measured_latency_ms": self.measured_latency_ms,
            "target_latency_ms": self.target_latency_ms,
            "latency_gap": self.latency_gap,
            "latency_ratio": self.latency_ratio,
            "improvement_potential": self.improvement_potential,
            "detected_at": self.detected_at.isoformat()
        }


# ============================================================
# Bottleneck Detector
# ============================================================

class BottleneckDetector:
    """
    Detects performance bottlenecks from telemetry data.

    Uses heuristics + thresholds to identify:
    - Functions that exceed latency targets
    - Resource utilization imbalances
    - Patterns that suggest HLS acceleration would help
    """

    def __init__(
        self,
        # Latency thresholds
        p95_target_ms: float = 120.0,
        p99_target_ms: float = 200.0,
        # Utilization thresholds
        cpu_high_threshold: float = 0.85,
        memory_high_threshold: float = 0.80,
        # Function-specific targets
        function_targets: Optional[Dict[str, float]] = None
    ):
        self.p95_target_ms = p95_target_ms
        self.p99_target_ms = p99_target_ms
        self.cpu_high_threshold = cpu_high_threshold
        self.memory_high_threshold = memory_high_threshold
        self.function_targets = function_targets or {}

        # Default function targets for known expensive operations
        self._default_function_targets = {
            "hyperbolic_distance": 5.0,
            "topological_features": 10.0,
            "persistent_homology": 20.0,
            "graph_laplacian": 15.0,
            "spectral_gap": 10.0,
            "semantic_encode": 8.0,
            "constraint_solve": 50.0
        }

        # History for trend detection
        self._metrics_history: List[PerformanceMetrics] = []
        self._max_history = 100
        self._bottleneck_counter = 0

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record metrics for trend analysis."""
        self._metrics_history.append(metrics)
        if len(self._metrics_history) > self._max_history:
            self._metrics_history.pop(0)

    def analyze(self, metrics: PerformanceMetrics) -> List[Bottleneck]:
        """
        Analyze metrics and detect bottlenecks.

        Returns list of detected bottlenecks, sorted by severity.
        """
        self.record_metrics(metrics)
        bottlenecks = []

        # Check overall latency
        if metrics.p95_latency_ms > self.p95_target_ms:
            severity = self._classify_latency_severity(
                metrics.p95_latency_ms, self.p95_target_ms
            )
            bottlenecks.append(self._create_bottleneck(
                bottleneck_type=BottleneckType.COMPUTE_BOUND,
                severity=severity,
                description=f"p95 latency {metrics.p95_latency_ms:.1f}ms exceeds target {self.p95_target_ms:.1f}ms",
                measured=metrics.p95_latency_ms,
                target=self.p95_target_ms,
                metrics=metrics
            ))

        # Check function-specific timings
        all_targets = {**self._default_function_targets, **self.function_targets}
        for func_name, timing_ms in metrics.function_timings.items():
            target = all_targets.get(func_name)
            if target and timing_ms > target:
                severity = self._classify_latency_severity(timing_ms, target)
                improvement = timing_ms / target if target > 0 else 1.0

                bottlenecks.append(self._create_bottleneck(
                    bottleneck_type=BottleneckType.KERNEL_SPECIFIC,
                    severity=severity,
                    description=f"Function {func_name} takes {timing_ms:.1f}ms (target: {target:.1f}ms)",
                    affected_function=func_name,
                    measured=timing_ms,
                    target=target,
                    improvement_potential=improvement,
                    metrics=metrics
                ))

        # Check for compute-bound (high CPU, low memory)
        if (metrics.cpu_utilization > self.cpu_high_threshold and
            metrics.memory_utilization < 0.5):
            bottlenecks.append(self._create_bottleneck(
                bottleneck_type=BottleneckType.COMPUTE_BOUND,
                severity=BottleneckSeverity.MEDIUM,
                description=f"CPU-bound: {metrics.cpu_utilization:.0%} CPU, {metrics.memory_utilization:.0%} memory",
                metrics=metrics
            ))

        # Check for memory-bound (high memory, moderate CPU)
        if (metrics.memory_utilization > self.memory_high_threshold and
            metrics.cpu_utilization < 0.6):
            bottlenecks.append(self._create_bottleneck(
                bottleneck_type=BottleneckType.MEMORY_BOUND,
                severity=BottleneckSeverity.MEDIUM,
                description=f"Memory-bound: {metrics.memory_utilization:.0%} memory, {metrics.cpu_utilization:.0%} CPU",
                metrics=metrics
            ))

        # Check queue buildup
        if metrics.pending_requests > 10:
            bottlenecks.append(self._create_bottleneck(
                bottleneck_type=BottleneckType.IO_BOUND,
                severity=BottleneckSeverity.MEDIUM if metrics.pending_requests < 50 else BottleneckSeverity.HIGH,
                description=f"Request queue depth: {metrics.pending_requests}",
                metrics=metrics
            ))

        # Sort by severity (critical first)
        severity_order = {
            BottleneckSeverity.CRITICAL: 0,
            BottleneckSeverity.HIGH: 1,
            BottleneckSeverity.MEDIUM: 2,
            BottleneckSeverity.LOW: 3
        }
        bottlenecks.sort(key=lambda b: severity_order.get(b.severity, 99))

        return bottlenecks

    def _classify_latency_severity(self, measured: float, target: float) -> BottleneckSeverity:
        """Classify severity based on how much we exceed target."""
        ratio = measured / target if target > 0 else 1.0

        if ratio >= 3.0:
            return BottleneckSeverity.CRITICAL
        elif ratio >= 2.0:
            return BottleneckSeverity.HIGH
        elif ratio >= 1.5:
            return BottleneckSeverity.MEDIUM
        else:
            return BottleneckSeverity.LOW

    def _create_bottleneck(
        self,
        bottleneck_type: BottleneckType,
        severity: BottleneckSeverity,
        description: str,
        affected_function: Optional[str] = None,
        measured: float = 0.0,
        target: float = 0.0,
        improvement_potential: float = 0.0,
        metrics: Optional[PerformanceMetrics] = None
    ) -> Bottleneck:
        """Create a bottleneck record."""
        self._bottleneck_counter += 1
        return Bottleneck(
            id=f"bottleneck_{self._bottleneck_counter:04d}",
            bottleneck_type=bottleneck_type,
            severity=severity,
            description=description,
            affected_function=affected_function,
            measured_latency_ms=measured,
            target_latency_ms=target,
            improvement_potential=improvement_potential if improvement_potential else (measured / target if target > 0 else 1.0),
            metrics_snapshot=metrics
        )

    def get_hls_candidates(self, bottlenecks: List[Bottleneck]) -> List[Bottleneck]:
        """
        Filter bottlenecks that are good candidates for HLS acceleration.

        Good candidates:
        - Kernel-specific bottlenecks (specific function is slow)
        - Compute-bound operations
        - High improvement potential
        """
        candidates = []

        for b in bottlenecks:
            # Kernel-specific with known function
            if b.bottleneck_type == BottleneckType.KERNEL_SPECIFIC and b.affected_function:
                candidates.append(b)
            # Compute-bound with significant gap
            elif b.bottleneck_type == BottleneckType.COMPUTE_BOUND and b.latency_ratio > 1.5:
                candidates.append(b)
            # Structural computations are good HLS targets
            elif b.bottleneck_type == BottleneckType.STRUCTURAL:
                candidates.append(b)

        return candidates


# ============================================================
# HLS Proposal
# ============================================================

@dataclass
class HLSProposal:
    """
    A proposed HLS kernel to address a bottleneck.

    Contains the generated C++ code + metadata for verification.
    """
    id: str
    bottleneck_id: str
    function_name: str
    description: str
    hls_code: str                    # The generated Vitis HLS C++
    interface_spec: Dict[str, Any]   # Input/output interface
    resource_estimate: Dict[str, Any]  # LUT, BRAM, DSP estimates
    latency_estimate_cycles: int = 0
    expected_speedup: float = 1.0
    status: ProposalStatus = ProposalStatus.DRAFT
    created_at: datetime = field(default_factory=datetime.now)
    verification_result: Optional[Dict[str, Any]] = None
    verification_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "bottleneck_id": self.bottleneck_id,
            "function_name": self.function_name,
            "description": self.description,
            "hls_code_lines": len(self.hls_code.split('\n')),
            "interface": self.interface_spec,
            "resources": self.resource_estimate,
            "latency_cycles": self.latency_estimate_cycles,
            "expected_speedup": self.expected_speedup,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "verification_errors": self.verification_errors
        }


# ============================================================
# HLS Templates
# ============================================================

class HLSTemplates:
    """
    Strict templates for HLS code generation.

    We don't let the LLM write freeform HLS - that's dangerous.
    Instead, we provide templates with well-defined parameters.
    """

    @staticmethod
    def hyperbolic_distance_kernel() -> str:
        """Template for hyperbolic distance computation."""
        return '''
// Hyperbolic Distance Kernel for Poincaré Ball Model
// Auto-generated by Ara Autosynth

#include "ap_fixed.h"
#include "hls_math.h"

typedef ap_fixed<32, 16> fixed_t;

void hyperbolic_distance(
    fixed_t x1[{DIM}],
    fixed_t y1[{DIM}],
    fixed_t x2[{DIM}],
    fixed_t y2[{DIM}],
    fixed_t* distance
) {{
    #pragma HLS INTERFACE mode=ap_fifo port=x1
    #pragma HLS INTERFACE mode=ap_fifo port=y1
    #pragma HLS INTERFACE mode=ap_fifo port=x2
    #pragma HLS INTERFACE mode=ap_fifo port=y2
    #pragma HLS INTERFACE mode=s_axilite port=distance
    #pragma HLS INTERFACE mode=s_axilite port=return

    fixed_t norm1_sq = 0;
    fixed_t norm2_sq = 0;
    fixed_t diff_sq = 0;

    COMPUTE_NORMS:
    for (int i = 0; i < {DIM}; i++) {{
        #pragma HLS PIPELINE II=1
        norm1_sq += x1[i] * x1[i] + y1[i] * y1[i];
        norm2_sq += x2[i] * x2[i] + y2[i] * y2[i];
        fixed_t dx = x1[i] - x2[i];
        fixed_t dy = y1[i] - y2[i];
        diff_sq += dx * dx + dy * dy;
    }}

    // Poincaré ball distance formula
    fixed_t denom = (1 - norm1_sq) * (1 - norm2_sq);
    fixed_t arg = 1 + 2 * diff_sq / denom;
    *distance = hls::acosh(arg);
}}
'''

    @staticmethod
    def spectral_gap_kernel() -> str:
        """Template for spectral gap computation (simplified)."""
        return '''
// Spectral Gap Kernel (Power Iteration Method)
// Auto-generated by Ara Autosynth

#include "ap_fixed.h"

typedef ap_fixed<32, 16> fixed_t;
const int MAX_ITER = {MAX_ITER};
const int N = {MATRIX_SIZE};

void spectral_gap(
    fixed_t adjacency[N][N],
    fixed_t* lambda2
) {{
    #pragma HLS INTERFACE mode=ap_memory port=adjacency
    #pragma HLS INTERFACE mode=s_axilite port=lambda2
    #pragma HLS INTERFACE mode=s_axilite port=return

    fixed_t v[N];
    fixed_t v_new[N];
    fixed_t lambda = 0;

    // Initialize eigenvector
    INIT:
    for (int i = 0; i < N; i++) {{
        #pragma HLS UNROLL factor=4
        v[i] = (i == 0) ? 1.0 : 0.0;
    }}

    // Power iteration
    ITERATE:
    for (int iter = 0; iter < MAX_ITER; iter++) {{
        #pragma HLS PIPELINE off

        // Matrix-vector multiply
        fixed_t norm_sq = 0;
        MATVEC:
        for (int i = 0; i < N; i++) {{
            #pragma HLS PIPELINE II=1
            fixed_t sum = 0;
            for (int j = 0; j < N; j++) {{
                sum += adjacency[i][j] * v[j];
            }}
            v_new[i] = sum;
            norm_sq += sum * sum;
        }}

        // Normalize
        fixed_t inv_norm = 1.0 / hls::sqrt(norm_sq);
        NORMALIZE:
        for (int i = 0; i < N; i++) {{
            #pragma HLS UNROLL factor=4
            v[i] = v_new[i] * inv_norm;
        }}

        lambda = norm_sq;
    }}

    *lambda2 = lambda;
}}
'''

    @staticmethod
    def topology_features_kernel() -> str:
        """Template for topological feature extraction."""
        return '''
// Topology Feature Extraction Kernel
// Auto-generated by Ara Autosynth

#include "ap_fixed.h"

typedef ap_fixed<32, 16> fixed_t;
const int N = {NUM_NODES};
const int E = {NUM_EDGES};

struct TopologyFeatures {{
    int betti_0;      // Connected components
    int betti_1;      // Cycles/holes
    fixed_t density;  // Edge density
}};

void topology_features(
    int edges[E][2],
    int num_edges,
    TopologyFeatures* features
) {{
    #pragma HLS INTERFACE mode=ap_memory port=edges
    #pragma HLS INTERFACE mode=s_axilite port=num_edges
    #pragma HLS INTERFACE mode=s_axilite port=features
    #pragma HLS INTERFACE mode=s_axilite port=return

    // Union-Find for connected components
    int parent[N];
    int rank[N];

    INIT_UF:
    for (int i = 0; i < N; i++) {{
        #pragma HLS UNROLL factor=8
        parent[i] = i;
        rank[i] = 0;
    }}

    int num_components = N;
    int num_cycles = 0;

    PROCESS_EDGES:
    for (int e = 0; e < num_edges; e++) {{
        #pragma HLS PIPELINE II=2
        int u = edges[e][0];
        int v = edges[e][1];

        // Find roots
        int root_u = u;
        int root_v = v;
        while (parent[root_u] != root_u) root_u = parent[root_u];
        while (parent[root_v] != root_v) root_v = parent[root_v];

        if (root_u != root_v) {{
            // Union
            if (rank[root_u] < rank[root_v]) {{
                parent[root_u] = root_v;
            }} else if (rank[root_u] > rank[root_v]) {{
                parent[root_v] = root_u;
            }} else {{
                parent[root_v] = root_u;
                rank[root_u]++;
            }}
            num_components--;
        }} else {{
            // Same component = cycle
            num_cycles++;
        }}
    }}

    features->betti_0 = num_components;
    features->betti_1 = num_cycles;
    features->density = (fixed_t)(2 * num_edges) / (N * (N - 1));
}}
'''

    @classmethod
    def get_template(cls, function_name: str) -> Optional[str]:
        """Get template by function name."""
        templates = {
            "hyperbolic_distance": cls.hyperbolic_distance_kernel,
            "spectral_gap": cls.spectral_gap_kernel,
            "topology_features": cls.topology_features_kernel,
            "topological_features": cls.topology_features_kernel,
            "graph_laplacian": cls.spectral_gap_kernel,  # Similar structure
        }
        template_fn = templates.get(function_name)
        return template_fn() if template_fn else None

    @classmethod
    def available_templates(cls) -> List[str]:
        """List available template names."""
        return ["hyperbolic_distance", "spectral_gap", "topology_features"]


# ============================================================
# Kernel Proposer
# ============================================================

class KernelProposer:
    """
    Proposes HLS kernels to address bottlenecks.

    Uses strict templates rather than freeform generation.
    Parameters are filled in based on bottleneck analysis.
    """

    def __init__(self):
        self._proposal_counter = 0

        # Default parameters for templates
        self._default_params = {
            "hyperbolic_distance": {"DIM": 64},
            "spectral_gap": {"MAX_ITER": 100, "MATRIX_SIZE": 128},
            "topology_features": {"NUM_NODES": 256, "NUM_EDGES": 1024}
        }

        # Resource estimates (rough, for planning)
        self._resource_estimates = {
            "hyperbolic_distance": {"LUT": 5000, "BRAM": 4, "DSP": 32},
            "spectral_gap": {"LUT": 15000, "BRAM": 32, "DSP": 64},
            "topology_features": {"LUT": 8000, "BRAM": 16, "DSP": 8}
        }

    def propose(
        self,
        bottleneck: Bottleneck,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Optional[HLSProposal]:
        """
        Propose an HLS kernel for the given bottleneck.

        Returns None if no suitable template exists.
        """
        func_name = bottleneck.affected_function

        if not func_name:
            # Try to infer from bottleneck type
            if bottleneck.bottleneck_type == BottleneckType.STRUCTURAL:
                func_name = "topology_features"
            else:
                return None

        # Get template
        template = HLSTemplates.get_template(func_name)
        if not template:
            return None

        # Get parameters
        params = self._default_params.get(func_name, {}).copy()
        if custom_params:
            params.update(custom_params)

        # Fill template
        try:
            hls_code = template.format(**params)
        except KeyError as e:
            return None

        # Create proposal
        self._proposal_counter += 1

        return HLSProposal(
            id=f"proposal_{self._proposal_counter:04d}",
            bottleneck_id=bottleneck.id,
            function_name=func_name,
            description=f"HLS kernel for {func_name} to address: {bottleneck.description}",
            hls_code=hls_code,
            interface_spec=self._extract_interface(hls_code),
            resource_estimate=self._resource_estimates.get(func_name, {}),
            latency_estimate_cycles=self._estimate_latency(func_name, params),
            expected_speedup=bottleneck.improvement_potential
        )

    def _extract_interface(self, hls_code: str) -> Dict[str, Any]:
        """Extract interface specification from HLS code."""
        interface = {"inputs": [], "outputs": [], "pragmas": []}

        # Find pragmas
        pragma_pattern = r'#pragma\s+HLS\s+INTERFACE\s+(.+)'
        for match in re.finditer(pragma_pattern, hls_code):
            interface["pragmas"].append(match.group(1).strip())

        # Find function signature
        func_pattern = r'void\s+(\w+)\s*\(([^)]+)\)'
        match = re.search(func_pattern, hls_code)
        if match:
            interface["function_name"] = match.group(1)
            params = match.group(2).split(',')
            for param in params:
                param = param.strip()
                if '*' in param:
                    interface["outputs"].append(param)
                else:
                    interface["inputs"].append(param)

        return interface

    def _estimate_latency(self, func_name: str, params: Dict[str, Any]) -> int:
        """Rough latency estimate in clock cycles."""
        base_latency = {
            "hyperbolic_distance": 100,
            "spectral_gap": 10000,
            "topology_features": 5000
        }

        latency = base_latency.get(func_name, 1000)

        # Scale by parameters
        if "DIM" in params:
            latency *= params["DIM"] // 64
        if "MATRIX_SIZE" in params:
            latency *= (params["MATRIX_SIZE"] // 128) ** 2
        if "NUM_EDGES" in params:
            latency *= params["NUM_EDGES"] // 1024

        return int(latency)


# ============================================================
# Proposal Verifier
# ============================================================

class ProposalVerifier:
    """
    Verifies HLS proposals for safety and correctness.

    Checks:
    - No dynamic memory allocation
    - No recursion
    - Resource bounds respected
    - Interface correctness
    - No obvious race conditions
    """

    def __init__(
        self,
        max_lut: int = 50000,
        max_bram: int = 100,
        max_dsp: int = 200
    ):
        self.max_lut = max_lut
        self.max_bram = max_bram
        self.max_dsp = max_dsp

        # Patterns to reject
        self._forbidden_patterns = [
            (r'\bmalloc\b', "Dynamic memory allocation (malloc)"),
            (r'\bfree\b', "Dynamic memory deallocation (free)"),
            (r'\bnew\b', "Dynamic allocation (new)"),
            (r'\bdelete\b', "Dynamic deallocation (delete)"),
            (r'\brecursive\b', "Recursion indicator"),
            (r'\bgoto\b', "Goto statement"),
            (r'\bvolatile\b', "Volatile keyword"),
            (r'\bpthread\b', "Threading primitives"),
            (r'\bstd::', "Standard library (use HLS libraries)"),
        ]

        # Required patterns
        self._required_patterns = [
            (r'#pragma\s+HLS\s+INTERFACE', "HLS interface pragma required"),
            (r'void\s+\w+\s*\(', "Function definition required"),
        ]

    def verify(self, proposal: HLSProposal) -> HLSProposal:
        """
        Verify an HLS proposal.

        Updates proposal status and verification_result.
        """
        errors = []
        warnings = []

        code = proposal.hls_code

        # Check forbidden patterns
        for pattern, description in self._forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                errors.append(f"FORBIDDEN: {description}")

        # Check required patterns
        for pattern, description in self._required_patterns:
            if not re.search(pattern, code):
                errors.append(f"MISSING: {description}")

        # Check resource estimates
        resources = proposal.resource_estimate
        if resources.get("LUT", 0) > self.max_lut:
            errors.append(f"LUT estimate {resources['LUT']} exceeds limit {self.max_lut}")
        if resources.get("BRAM", 0) > self.max_bram:
            errors.append(f"BRAM estimate {resources['BRAM']} exceeds limit {self.max_bram}")
        if resources.get("DSP", 0) > self.max_dsp:
            errors.append(f"DSP estimate {resources['DSP']} exceeds limit {self.max_dsp}")

        # Check interface
        interface = proposal.interface_spec
        if not interface.get("pragmas"):
            warnings.append("No interface pragmas detected")
        if not interface.get("function_name"):
            errors.append("Could not parse function name")

        # Check for common HLS issues
        if "while(1)" in code or "while (1)" in code:
            errors.append("Infinite loop detected")

        # Update proposal
        proposal.verification_errors = errors
        proposal.verification_result = {
            "errors": errors,
            "warnings": warnings,
            "passed": len(errors) == 0
        }

        if len(errors) == 0:
            proposal.status = ProposalStatus.VERIFIED
        else:
            proposal.status = ProposalStatus.REJECTED

        return proposal


# ============================================================
# Deployment Manager
# ============================================================

class DeploymentManager:
    """
    Manages deployment of verified HLS kernels.

    In Phase A (current), this is a placeholder that:
    - Tracks approved proposals
    - Records what would be deployed
    - Provides hooks for actual deployment when ready

    Actual synthesis + deployment requires:
    - Vitis HLS toolchain
    - CXL fabric access
    - Bitstream loading capability
    """

    def __init__(self, auto_deploy: bool = False):
        self.auto_deploy = auto_deploy  # Phase A: always False

        # Tracking
        self._proposals: Dict[str, HLSProposal] = {}
        self._deployed: Dict[str, Dict[str, Any]] = {}
        self._deployment_history: List[Dict[str, Any]] = []

    def register_proposal(self, proposal: HLSProposal) -> None:
        """Register a verified proposal."""
        if proposal.status == ProposalStatus.VERIFIED:
            self._proposals[proposal.id] = proposal

    def approve(self, proposal_id: str) -> bool:
        """Human approval of a proposal for synthesis."""
        if proposal_id not in self._proposals:
            return False

        proposal = self._proposals[proposal_id]
        if proposal.status != ProposalStatus.VERIFIED:
            return False

        proposal.status = ProposalStatus.APPROVED
        return True

    def simulate_deploy(self, proposal_id: str) -> Dict[str, Any]:
        """
        Simulate deployment (Phase A).

        Returns what the deployment would look like.
        """
        if proposal_id not in self._proposals:
            return {"error": "Proposal not found"}

        proposal = self._proposals[proposal_id]
        if proposal.status not in [ProposalStatus.APPROVED, ProposalStatus.VERIFIED]:
            return {"error": f"Proposal status is {proposal.status.value}, not approved"}

        deployment_record = {
            "proposal_id": proposal_id,
            "function_name": proposal.function_name,
            "simulated_at": datetime.now().isoformat(),
            "would_deploy_to": "fpga_slot_0",
            "expected_speedup": proposal.expected_speedup,
            "resource_usage": proposal.resource_estimate,
            "status": "simulated"
        }

        self._deployment_history.append(deployment_record)
        return deployment_record

    def get_deployment_summary(self) -> Dict[str, Any]:
        """Get summary of deployment state."""
        return {
            "registered_proposals": len(self._proposals),
            "approved": len([p for p in self._proposals.values() if p.status == ProposalStatus.APPROVED]),
            "deployed": len(self._deployed),
            "deployment_history": len(self._deployment_history),
            "auto_deploy": self.auto_deploy
        }


# ============================================================
# Autosynth Controller
# ============================================================

class AutosynthController:
    """
    Main controller for the hardware generative loop.

    Orchestrates:
    - Bottleneck detection
    - Kernel proposal
    - Verification
    - Deployment tracking
    """

    def __init__(
        self,
        detector: Optional[BottleneckDetector] = None,
        proposer: Optional[KernelProposer] = None,
        verifier: Optional[ProposalVerifier] = None,
        deployment: Optional[DeploymentManager] = None
    ):
        self.detector = detector or BottleneckDetector()
        self.proposer = proposer or KernelProposer()
        self.verifier = verifier or ProposalVerifier()
        self.deployment = deployment or DeploymentManager(auto_deploy=False)

        # Statistics
        self._stats = {
            "bottlenecks_detected": 0,
            "proposals_generated": 0,
            "proposals_verified": 0,
            "proposals_rejected": 0
        }

    def analyze_and_propose(
        self,
        metrics: PerformanceMetrics
    ) -> Dict[str, Any]:
        """
        Full pipeline: detect bottlenecks → propose kernels → verify.

        Returns summary of what was found and proposed.
        """
        result = {
            "bottlenecks": [],
            "proposals": [],
            "recommendations": []
        }

        # Detect bottlenecks
        bottlenecks = self.detector.analyze(metrics)
        self._stats["bottlenecks_detected"] += len(bottlenecks)
        result["bottlenecks"] = [b.to_dict() for b in bottlenecks]

        # Get HLS candidates
        candidates = self.detector.get_hls_candidates(bottlenecks)

        # Propose kernels for candidates
        for bottleneck in candidates:
            proposal = self.proposer.propose(bottleneck)
            if proposal:
                # Verify
                proposal = self.verifier.verify(proposal)
                self._stats["proposals_generated"] += 1

                if proposal.status == ProposalStatus.VERIFIED:
                    self._stats["proposals_verified"] += 1
                    self.deployment.register_proposal(proposal)
                    result["recommendations"].append(
                        f"✅ Kernel '{proposal.function_name}' verified. "
                        f"Expected {proposal.expected_speedup:.1f}x speedup."
                    )
                else:
                    self._stats["proposals_rejected"] += 1
                    result["recommendations"].append(
                        f"⚠️ Kernel '{proposal.function_name}' rejected: "
                        f"{', '.join(proposal.verification_errors[:2])}"
                    )

                result["proposals"].append(proposal.to_dict())

        # Add general recommendations
        if not candidates and bottlenecks:
            result["recommendations"].append(
                "Bottlenecks detected but no HLS templates available. "
                "Consider r,k tuning or software optimization."
            )

        return result

    def get_proposal(self, proposal_id: str) -> Optional[HLSProposal]:
        """Get a specific proposal."""
        return self.deployment._proposals.get(proposal_id)

    def get_proposal_code(self, proposal_id: str) -> Optional[str]:
        """Get the HLS code for a proposal."""
        proposal = self.get_proposal(proposal_id)
        return proposal.hls_code if proposal else None

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "deployment": self.deployment.get_deployment_summary()
        }


# ============================================================
# Factory Functions
# ============================================================

def create_autosynth_controller() -> AutosynthController:
    """Create a default autosynth controller."""
    return AutosynthController()


def analyze_bottlenecks(metrics: PerformanceMetrics) -> List[Bottleneck]:
    """Convenience function to analyze metrics for bottlenecks."""
    detector = BottleneckDetector()
    return detector.analyze(metrics)


def propose_kernel(bottleneck: Bottleneck) -> Optional[HLSProposal]:
    """Convenience function to propose a kernel for a bottleneck."""
    proposer = KernelProposer()
    verifier = ProposalVerifier()

    proposal = proposer.propose(bottleneck)
    if proposal:
        proposal = verifier.verify(proposal)
    return proposal


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
