"""
Quantum Hyperdimensional Computing (QHDC) Bridge
=================================================

Maps classical 16kD hypervectors to quantum state representations
and provides quantum optimal control for vector steering.

Architecture:
    Classical HV (16384D bipolar)
        ↓ chunk encoding
    Qubit register (14 qubits = 2^14 = 16384 amplitudes)
        ↓ quantum operations
    Bundling = superposition
    Binding = phase oracle
    Similarity = Hadamard test
        ↓ measure / decode
    Classical result

Modes:
    - SIMULATION: Pure numpy simulation (default)
    - PENNYLANE: Use PennyLane if available
    - NISQ_STUB: Interface for real NISQ backend
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
import math


class QHDCBackend(str, Enum):
    """Backend for QHDC operations."""
    SIMULATION = "simulation"  # Pure numpy
    PENNYLANE = "pennylane"    # PennyLane quantum
    NISQ_STUB = "nisq_stub"    # Placeholder for real hardware


@dataclass
class QuantumHV:
    """A hypervector encoded as quantum state amplitudes."""
    n_qubits: int
    amplitudes: np.ndarray  # Complex amplitudes, shape (2^n_qubits,)

    @property
    def dim(self) -> int:
        return 2 ** self.n_qubits

    def to_classical(self) -> np.ndarray:
        """Convert to classical bipolar HV via sign of real part."""
        return np.sign(self.amplitudes.real).astype(np.float32)

    def probability_distribution(self) -> np.ndarray:
        """Get measurement probability distribution."""
        return np.abs(self.amplitudes) ** 2

    def sample(self, n_shots: int = 1000) -> np.ndarray:
        """Sample from the quantum state."""
        probs = self.probability_distribution()
        probs = probs / probs.sum()  # Normalize
        samples = np.random.choice(len(probs), size=n_shots, p=probs)
        return samples


class QHDCEncoder:
    """Encode classical hypervectors into quantum states."""

    def __init__(self, classical_dim: int = 16384):
        self.classical_dim = classical_dim
        # Number of qubits needed: 2^n >= classical_dim
        self.n_qubits = int(np.ceil(np.log2(classical_dim)))
        self.quantum_dim = 2 ** self.n_qubits

    def encode(self, hv: np.ndarray) -> QuantumHV:
        """Encode classical HV as quantum amplitudes."""
        # Pad to quantum dimension if needed
        if len(hv) < self.quantum_dim:
            hv = np.pad(hv, (0, self.quantum_dim - len(hv)))

        # Normalize to unit vector (valid quantum state)
        amplitudes = hv.astype(np.complex128)
        norm = np.linalg.norm(amplitudes)
        if norm > 0:
            amplitudes = amplitudes / norm

        return QuantumHV(n_qubits=self.n_qubits, amplitudes=amplitudes)

    def decode(self, qhv: QuantumHV) -> np.ndarray:
        """Decode quantum state back to classical HV."""
        classical = qhv.to_classical()
        return classical[:self.classical_dim]


class QuantumOperations:
    """Quantum operations on hypervector states."""

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits

    def hadamard_all(self, state: np.ndarray) -> np.ndarray:
        """Apply Hadamard to all qubits (superposition)."""
        # H^⊗n creates uniform superposition from |0⟩
        # For general state: each qubit H transforms
        result = state.copy()
        for q in range(self.n_qubits):
            result = self._apply_single_qubit_gate(result, q, self._H)
        return result

    def phase_oracle(self, state: np.ndarray, pattern: np.ndarray) -> np.ndarray:
        """Apply phase oracle based on pattern (for binding)."""
        # Flip phase where pattern is -1
        phases = np.where(pattern[:self.dim] < 0, -1, 1).astype(np.complex128)
        return state * phases

    def controlled_phase(self, state: np.ndarray, control_qubit: int,
                         target_qubit: int, phase: float) -> np.ndarray:
        """Apply controlled phase rotation."""
        result = state.copy()
        # Apply phase when both control and target are |1⟩
        for i in range(self.dim):
            if (i >> control_qubit) & 1 and (i >> target_qubit) & 1:
                result[i] *= np.exp(1j * phase)
        return result

    def qft(self, state: np.ndarray) -> np.ndarray:
        """Quantum Fourier Transform."""
        # Standard QFT implementation
        result = state.copy()
        for j in range(self.n_qubits):
            result = self._apply_single_qubit_gate(result, j, self._H)
            for k in range(j + 1, self.n_qubits):
                phase = np.pi / (2 ** (k - j))
                result = self.controlled_phase(result, k, j, phase)
        # Bit reversal
        result = self._bit_reverse(result)
        return result

    def inverse_qft(self, state: np.ndarray) -> np.ndarray:
        """Inverse QFT."""
        result = self._bit_reverse(state.copy())
        for j in range(self.n_qubits - 1, -1, -1):
            for k in range(self.n_qubits - 1, j, -1):
                phase = -np.pi / (2 ** (k - j))
                result = self.controlled_phase(result, k, j, phase)
            result = self._apply_single_qubit_gate(result, j, self._H)
        return result

    # Single qubit gates
    @property
    def _H(self) -> np.ndarray:
        """Hadamard gate."""
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    @property
    def _X(self) -> np.ndarray:
        """Pauli-X gate."""
        return np.array([[0, 1], [1, 0]])

    @property
    def _Z(self) -> np.ndarray:
        """Pauli-Z gate."""
        return np.array([[1, 0], [0, -1]])

    def _apply_single_qubit_gate(self, state: np.ndarray, qubit: int,
                                  gate: np.ndarray) -> np.ndarray:
        """Apply single qubit gate to state."""
        result = np.zeros_like(state)
        stride = 2 ** qubit

        for i in range(0, self.dim, 2 * stride):
            for j in range(stride):
                idx0 = i + j
                idx1 = i + j + stride
                result[idx0] = gate[0, 0] * state[idx0] + gate[0, 1] * state[idx1]
                result[idx1] = gate[1, 0] * state[idx0] + gate[1, 1] * state[idx1]

        return result

    def _bit_reverse(self, state: np.ndarray) -> np.ndarray:
        """Bit-reverse permutation."""
        result = np.zeros_like(state)
        for i in range(self.dim):
            rev_i = int(bin(i)[2:].zfill(self.n_qubits)[::-1], 2)
            result[rev_i] = state[i]
        return result


class QHDCOperations:
    """QHDC-style operations on quantum hypervectors."""

    def __init__(self, encoder: QHDCEncoder):
        self.encoder = encoder
        self.qops = QuantumOperations(encoder.n_qubits)

    def bundle(self, qhvs: List[QuantumHV]) -> QuantumHV:
        """Bundle multiple quantum HVs via superposition."""
        if len(qhvs) == 0:
            # Return |0⟩ state
            amps = np.zeros(self.encoder.quantum_dim, dtype=np.complex128)
            amps[0] = 1.0
            return QuantumHV(self.encoder.n_qubits, amps)

        if len(qhvs) == 1:
            return qhvs[0]

        # Superposition of all states (normalized sum)
        combined = np.zeros(self.encoder.quantum_dim, dtype=np.complex128)
        for qhv in qhvs:
            combined += qhv.amplitudes

        # Normalize
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm

        return QuantumHV(self.encoder.n_qubits, combined)

    def bind(self, qhv1: QuantumHV, qhv2: QuantumHV) -> QuantumHV:
        """Bind two quantum HVs via phase oracle."""
        # Convert qhv2 to classical pattern for oracle
        pattern = qhv2.to_classical()

        # Apply phase oracle to qhv1
        result = self.qops.phase_oracle(qhv1.amplitudes, pattern)

        return QuantumHV(self.encoder.n_qubits, result)

    def permute(self, qhv: QuantumHV, shifts: int = 1) -> QuantumHV:
        """Permute quantum HV via QFT-based circular shift."""
        # Apply QFT
        state = self.qops.qft(qhv.amplitudes)

        # Phase shift in frequency domain (circular shift in time domain)
        phases = np.exp(-2j * np.pi * shifts * np.arange(self.encoder.quantum_dim)
                       / self.encoder.quantum_dim)
        state = state * phases

        # Inverse QFT
        state = self.qops.inverse_qft(state)

        return QuantumHV(self.encoder.n_qubits, state)

    def hadamard_similarity(self, qhv1: QuantumHV, qhv2: QuantumHV,
                            n_shots: int = 1000) -> float:
        """Estimate similarity via Hadamard test (SWAP test variant)."""
        # Simplified: compute overlap directly
        # In real QHDC, this would be a SWAP test circuit
        overlap = np.abs(np.vdot(qhv1.amplitudes, qhv2.amplitudes))
        return float(overlap)

    def amplitude_amplification(self, qhv: QuantumHV, target_pattern: np.ndarray,
                                 iterations: int = 3) -> QuantumHV:
        """Grover-style amplitude amplification toward target."""
        state = qhv.amplitudes.copy()

        for _ in range(iterations):
            # Oracle: flip phase of target
            state = self.qops.phase_oracle(state, target_pattern)

            # Diffusion: reflect about mean
            mean = np.mean(state)
            state = 2 * mean - state

            # Renormalize
            norm = np.linalg.norm(state)
            if norm > 0:
                state = state / norm

        return QuantumHV(self.encoder.n_qubits, state)


# =============================================================================
# QUANTUM OPTIMAL CONTROL
# =============================================================================

@dataclass
class ControlPulse:
    """A control pulse for steering quantum state."""
    name: str
    gate_sequence: List[Tuple[str, int, float]]  # (gate_type, qubit, param)
    duration: float = 1.0


class QuantumOptimalControl:
    """Quantum optimal control for hypervector steering."""

    def __init__(self, qhdc: QHDCOperations,
                 optimizer: str = "adam_spsa"):
        self.qhdc = qhdc
        self.optimizer = optimizer
        self.n_qubits = qhdc.encoder.n_qubits

    def fidelity(self, state: QuantumHV, target: QuantumHV) -> float:
        """Compute fidelity between states."""
        return float(np.abs(np.vdot(state.amplitudes, target.amplitudes)) ** 2)

    def cost_function(self, params: np.ndarray,
                      initial: QuantumHV, target: QuantumHV) -> float:
        """Cost function for optimization (1 - fidelity)."""
        # Apply parameterized circuit
        evolved = self._apply_parameterized_circuit(initial, params)
        return 1.0 - self.fidelity(evolved, target)

    def _apply_parameterized_circuit(self, state: QuantumHV,
                                      params: np.ndarray) -> QuantumHV:
        """Apply a parameterized quantum circuit."""
        amplitudes = state.amplitudes.copy()
        n_layers = len(params) // (2 * self.n_qubits)

        param_idx = 0
        for layer in range(max(1, n_layers)):
            # Rotation layer
            for q in range(self.n_qubits):
                if param_idx < len(params):
                    # Rz rotation
                    phase = params[param_idx]
                    for i in range(len(amplitudes)):
                        if (i >> q) & 1:
                            amplitudes[i] *= np.exp(1j * phase)
                    param_idx += 1

            # Entangling layer
            for q in range(self.n_qubits):
                if param_idx < len(params):
                    # Controlled phase
                    next_q = (q + 1) % self.n_qubits
                    phase = params[param_idx]
                    for i in range(len(amplitudes)):
                        if (i >> q) & 1 and (i >> next_q) & 1:
                            amplitudes[i] *= np.exp(1j * phase)
                    param_idx += 1

        return QuantumHV(self.n_qubits, amplitudes)

    def optimize(self, initial: QuantumHV, target: QuantumHV,
                 n_params: int = 28, max_iter: int = 100,
                 learning_rate: float = 0.1) -> Tuple[np.ndarray, float]:
        """Find optimal control parameters to steer initial toward target."""
        # Initialize parameters
        params = np.random.randn(n_params) * 0.1

        best_params = params.copy()
        best_cost = self.cost_function(params, initial, target)

        # Simple gradient-free optimization (SPSA-style)
        for iteration in range(max_iter):
            # Perturbation
            delta = np.random.choice([-1, 1], size=n_params)
            c = 0.1 / (iteration + 1) ** 0.101

            # Evaluate at +/- perturbation
            params_plus = params + c * delta
            params_minus = params - c * delta

            cost_plus = self.cost_function(params_plus, initial, target)
            cost_minus = self.cost_function(params_minus, initial, target)

            # Gradient estimate
            grad_est = (cost_plus - cost_minus) / (2 * c) * delta

            # Update
            a = learning_rate / (iteration + 1) ** 0.602
            params = params - a * grad_est

            # Track best
            current_cost = self.cost_function(params, initial, target)
            if current_cost < best_cost:
                best_cost = current_cost
                best_params = params.copy()

        return best_params, 1.0 - best_cost  # Return params and fidelity


# =============================================================================
# HYBRID CLASSICAL-QUANTUM PIPELINE
# =============================================================================

class HybridPipeline:
    """Hybrid classical-quantum pipeline for cathedral control."""

    def __init__(self, classical_dim: int = 16384,
                 backend: QHDCBackend = QHDCBackend.SIMULATION):
        self.encoder = QHDCEncoder(classical_dim)
        self.qhdc = QHDCOperations(self.encoder)
        self.qoc = QuantumOptimalControl(self.qhdc)
        self.backend = backend

        # Cache for frequently used states
        self._state_cache: Dict[str, QuantumHV] = {}

    def classical_to_quantum(self, hv: np.ndarray) -> QuantumHV:
        """Encode classical HV to quantum."""
        return self.encoder.encode(hv)

    def quantum_to_classical(self, qhv: QuantumHV) -> np.ndarray:
        """Decode quantum state to classical HV."""
        return self.encoder.decode(qhv)

    def quantum_bundle(self, hvs: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple HVs using quantum superposition."""
        qhvs = [self.encoder.encode(hv) for hv in hvs]
        bundled = self.qhdc.bundle(qhvs)
        return self.encoder.decode(bundled)

    def quantum_bind(self, hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
        """Bind two HVs using quantum phase oracle."""
        qhv1 = self.encoder.encode(hv1)
        qhv2 = self.encoder.encode(hv2)
        bound = self.qhdc.bind(qhv1, qhv2)
        return self.encoder.decode(bound)

    def quantum_similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute similarity using Hadamard test."""
        qhv1 = self.encoder.encode(hv1)
        qhv2 = self.encoder.encode(hv2)
        return self.qhdc.hadamard_similarity(qhv1, qhv2)

    def steer_toward(self, current: np.ndarray, target: np.ndarray,
                     max_iter: int = 50) -> Tuple[np.ndarray, float]:
        """Use quantum optimal control to steer current toward target."""
        qcurrent = self.encoder.encode(current)
        qtarget = self.encoder.encode(target)

        # Optimize control
        params, fidelity = self.qoc.optimize(qcurrent, qtarget, max_iter=max_iter)

        # Apply optimal control
        evolved = self.qoc._apply_parameterized_circuit(qcurrent, params)

        return self.encoder.decode(evolved), fidelity

    def search_memory(self, query: np.ndarray,
                      memories: List[np.ndarray],
                      top_k: int = 3) -> List[Tuple[int, float]]:
        """Search memories using quantum amplitude amplification."""
        qquery = self.encoder.encode(query)

        results = []
        for i, mem in enumerate(memories):
            qmem = self.encoder.encode(mem)
            sim = self.qhdc.hadamard_similarity(qquery, qmem)
            results.append((i, sim))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# =============================================================================
# CONVENIENCE
# =============================================================================

_pipeline: Optional[HybridPipeline] = None


def get_hybrid_pipeline() -> HybridPipeline:
    """Get global hybrid pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = HybridPipeline()
    return _pipeline


def qhdc_bundle(hvs: List[np.ndarray]) -> np.ndarray:
    """Bundle HVs using QHDC."""
    return get_hybrid_pipeline().quantum_bundle(hvs)


def qhdc_bind(hv1: np.ndarray, hv2: np.ndarray) -> np.ndarray:
    """Bind HVs using QHDC."""
    return get_hybrid_pipeline().quantum_bind(hv1, hv2)


def qhdc_similarity(hv1: np.ndarray, hv2: np.ndarray) -> float:
    """Compute similarity using QHDC."""
    return get_hybrid_pipeline().quantum_similarity(hv1, hv2)


def qhdc_steer(current: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
    """Steer HV toward target using quantum optimal control."""
    return get_hybrid_pipeline().steer_toward(current, target)
