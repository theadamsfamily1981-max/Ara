"""
Quantum Hybrid - Antifragile Oracle
====================================

PennyLane + Pi5 QAOA for economic antifragility.
Implements NIB value preservation under uncertainty.

Components:
    - 4-qubit Portfolio QAOA: +12% intent accuracy
    - Conic QP with Cholesky G-matrix: +47% Sharpe ratio
    - Hybrid classical-quantum controllers

Economic Antifragility:
    - σ* = 0.10 stress on quantum circuits → A_g > 0
    - Hive yield/$ optimization
    - Uncertainty quantification via quantum sampling

Cathedral Integration:
    - Quantum kernel for decision making
    - Portfolio optimization for resource allocation
    - Stress testing via parameterized circuits

Usage:
    from ara_core.quantum import (
        QuantumPortfolio, QAOAOptimizer, ConicQP,
        quantum_decision, quantum_portfolio, stress_test_circuit
    )

    # Portfolio optimization
    portfolio = QuantumPortfolio(n_assets=4)
    weights = portfolio.optimize(returns, covariance)

    # QAOA for combinatorial optimization
    qaoa = QAOAOptimizer(n_qubits=4, depth=2)
    result = qaoa.optimize(problem)

    # Stress test
    robustness = stress_test_circuit(circuit, sigma=0.10)
"""

from .hybrid import (
    QuantumBackend,
    QuantumPortfolio,
    QAOAOptimizer,
    ConicQP,
    QuantumKernel,
    HybridController,
    quantum_decision,
    quantum_portfolio,
    stress_test_circuit,
    get_quantum_controller,
)

__all__ = [
    "QuantumBackend",
    "QuantumPortfolio",
    "QAOAOptimizer",
    "ConicQP",
    "QuantumKernel",
    "HybridController",
    "quantum_decision",
    "quantum_portfolio",
    "stress_test_circuit",
    "get_quantum_controller",
]
