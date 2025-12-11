#!/usr/bin/env python3
# ara/oracle/pythia_triad.py
"""
Pythia Triad Orchestrator: The Three Oracles United

Coordinates all three Oracles:
- Alpha (Visionary): Long-horizon planning
- Beta (Analyst): Real-time inference
- Gamma (Arbiter): Safety enforcement

Hardware allocation:
- GPU 1: Oracle Alpha (world model + particles)
- GPU 2: Oracle Beta (fast inference)
- BittWare FPGA: HDC acceleration for Beta
- Forest Kitten FPGA: Safety circuits for Gamma
- Threadripper cores 32-64: MEIS meta-reasoner
"""

import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
import numpy as np

from .alpha_visionary import OracleAlpha, Prophecy
from .beta_analyst import OracleBeta, BittWareFPGAInterface, AnalystPrediction
from .gamma_arbiter import OracleGamma, ForestKittenSafetyCore, ArbitrationResult

logger = logging.getLogger(__name__)

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class OracleConsensus:
    """Result of three-Oracle deliberation."""
    final_action: np.ndarray
    confidence: float
    reasoning: str
    prophecies: List[Dict[str, Any]]
    prediction: Dict[str, Any]
    arbitration: Dict[str, Any]
    latency_ms: float
    entropy_cost: float
    safe: bool
    query: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'final_action': self.final_action.tolist(),
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'prophecies': self.prophecies,
            'prediction': self.prediction,
            'arbitration': self.arbitration,
            'latency_ms': self.latency_ms,
            'entropy_cost': self.entropy_cost,
            'safe': self.safe,
            'query': self.query,
            'timestamp': self.timestamp
        }

    def narrative(self) -> str:
        """Generate human-readable narrative of the consultation."""
        lines = [
            "=" * 70,
            "ORACLE CONSENSUS",
            "=" * 70,
            f"Query: {self.query}" if self.query else "",
            "",
            f"Final Action: {self.final_action}",
            f"Confidence: {self.confidence:.0%}",
            f"Safe: {'Yes' if self.safe else 'NO - SAFETY CONCERNS'}",
            "",
            f"Reasoning: {self.reasoning}",
            "",
            f"Performance:",
            f"  Total Latency: {self.latency_ms:.1f} ms",
            f"  Entropy Cost: {self.entropy_cost:.3e} bits",
            "",
            "Prophecies from Alpha (Visionary):"
        ]

        for i, p in enumerate(self.prophecies[:3], 1):
            lines.append(f"  {i}. {p.get('narrative', 'No narrative')}")

        lines.extend([
            "",
            "Prediction from Beta (Analyst):",
            f"  Latency: {self.prediction.get('latency_ms', 0):.3f} ms",
            f"  Uncertainty: {self.prediction.get('uncertainty', 0):.3f}",
            "",
            "=" * 70
        ])

        return "\n".join(lines)


class PythiaTriad:
    """
    The complete Oracle system: Alpha + Beta + Gamma + MEIS meta-reasoner.

    Provides unified interface for consulting all three Oracles
    and synthesizing their recommendations into coherent actions.
    """

    def __init__(
        self,
        encoder: Optional[Callable] = None,
        world_model_alpha: Optional[Any] = None,
        world_model_beta: Optional[Any] = None,
        latent_dim: int = 10,
        action_dim: int = 4,
        num_particles: int = 10000,  # Reduced default for faster demos
        entropy_budget_bits: float = 1e9
    ):
        logger.info("=" * 70)
        logger.info("INITIALIZING PYTHIA TRIAD")
        logger.info("=" * 70)

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Oracle Alpha: The Visionary (GPU 1)
        logger.info("Spawning Oracle Alpha (The Visionary)...")
        self.alpha = OracleAlpha(
            world_model=world_model_alpha,
            encoder=encoder,
            latent_dim=latent_dim,
            num_particles=num_particles
        )

        # Oracle Beta: The Analyst (GPU 2 + FPGA)
        logger.info("Spawning Oracle Beta (The Analyst)...")
        fpga = BittWareFPGAInterface(simulation_mode=True)
        self.beta = OracleBeta(
            world_model=world_model_beta,
            fpga_interface=fpga,
            latent_dim=latent_dim,
            action_dim=action_dim
        )

        # Oracle Gamma: The Arbiter (Forest Kitten FPGA)
        logger.info("Spawning Oracle Gamma (The Arbiter)...")
        safety_core = ForestKittenSafetyCore(simulation_mode=True)
        self.gamma = OracleGamma(
            safety_core=safety_core,
            entropy_budget_bits=entropy_budget_bits
        )

        # Consultation history
        self.consultation_history: List[OracleConsensus] = []

        # MEIS meta-learning state
        self.oracle_accuracy_history = {
            'alpha': [],
            'beta': []
        }

        logger.info("=" * 70)
        logger.info("PYTHIA TRIAD ACTIVE")
        logger.info("=" * 70)

    def divine(
        self,
        current_state: Any,
        observation: Optional[np.ndarray] = None,
        action_candidates: Optional[List[np.ndarray]] = None,
        query: str = "What should I do?",
        horizon: int = 100,
        return_top_k: int = 3
    ) -> OracleConsensus:
        """
        Consult all three Oracles and synthesize answer.

        Process:
        1. Alpha explores long-horizon futures
        2. Beta provides fast tactical analysis
        3. Gamma arbitrates and enforces covenants
        4. MEIS meta-learns which Oracle to trust

        Args:
            current_state: Current latent state (tensor or numpy)
            observation: Raw sensor data (optional)
            action_candidates: Possible actions to evaluate
            query: Question for the Oracle
            horizon: How many steps ahead for Alpha
            return_top_k: How many prophecies to return

        Returns:
            OracleConsensus with final decision
        """
        start_time = time.time()

        logger.info("=" * 70)
        logger.info("CONSULTING THE PYTHIA TRIAD")
        logger.info("Query: %s", query)
        logger.info("=" * 70)

        # Ensure state is numpy array
        if TORCH_AVAILABLE and isinstance(current_state, torch.Tensor):
            state_np = current_state.cpu().numpy()
        else:
            state_np = np.array(current_state)

        # Default action candidates if not provided
        if action_candidates is None:
            action_candidates = self._generate_default_actions()

        # ================================================================
        # Step 1: Oracle Alpha divines long-term futures
        # ================================================================
        logger.info("Alpha (Visionary) exploring %d-step futures...", horizon)

        alpha_prophecies = self.alpha.divine_futures(
            current_state,
            action_candidates,
            horizon=horizon,
            return_top_k=return_top_k
        )

        # Best prophecy determines Alpha's recommendation
        alpha_best = alpha_prophecies[0] if alpha_prophecies else None
        if alpha_best:
            alpha_action = action_candidates[alpha_best.action_index]
            if TORCH_AVAILABLE and isinstance(alpha_action, torch.Tensor):
                alpha_action = alpha_action.numpy()
        else:
            alpha_action = np.zeros(self.action_dim)

        logger.info("  Alpha recommends action %d (value=%.3f, risk=%.1f%%)",
                    alpha_best.action_index if alpha_best else -1,
                    alpha_best.value if alpha_best else 0,
                    (alpha_best.black_swan_risk if alpha_best else 0) * 100)

        # ================================================================
        # Step 2: Oracle Beta provides real-time analysis
        # ================================================================
        logger.info("Beta (Analyst) running sub-ms inference...")

        beta_action = alpha_action.copy()  # Could be independent
        beta_prediction = self.beta.predict_next_state(state_np, beta_action)

        logger.info("  Beta latency: %.3f ms, uncertainty: %.3f",
                    beta_prediction.latency_ms, beta_prediction.uncertainty)

        # ================================================================
        # Step 3: Oracle Gamma arbitrates
        # ================================================================
        logger.info("Gamma (Arbiter) checking covenants...")

        alpha_rec = {'action': alpha_action}
        beta_rec = {'action': beta_action}

        arbitration = self.gamma.arbitrate_oracles(
            alpha_rec,
            beta_rec,
            state_np
        )

        logger.info("  Agreement: %.0f%%, Safe: %s",
                    arbitration.agreement * 100, arbitration.safe)

        # ================================================================
        # Package consensus
        # ================================================================
        total_latency = (time.time() - start_time) * 1000

        consensus = OracleConsensus(
            final_action=arbitration.action,
            confidence=arbitration.agreement,
            reasoning=arbitration.rationale,
            prophecies=[p.to_dict() for p in alpha_prophecies],
            prediction=beta_prediction.to_dict(),
            arbitration=arbitration.to_dict(),
            latency_ms=total_latency,
            entropy_cost=self.gamma.total_entropy_bits,
            safe=arbitration.safe,
            query=query
        )

        self.consultation_history.append(consensus)

        logger.info("=" * 70)
        logger.info("CONSENSUS REACHED")
        logger.info("  Action: %s", consensus.final_action)
        logger.info("  Confidence: %.0f%%", consensus.confidence * 100)
        logger.info("  Latency: %.1f ms", consensus.latency_ms)
        logger.info("=" * 70)

        return consensus

    def _generate_default_actions(self) -> List[np.ndarray]:
        """Generate default action candidates."""
        actions = []

        # Cardinal directions
        for i in range(self.action_dim):
            action = np.zeros(self.action_dim, dtype=np.float32)
            action[i] = 1.0
            actions.append(action)

        # Mixed actions
        actions.append(np.ones(self.action_dim, dtype=np.float32) / self.action_dim)
        actions.append(np.zeros(self.action_dim, dtype=np.float32))  # No-op

        return actions

    def quick_predict(
        self,
        current_state: np.ndarray,
        action: np.ndarray
    ) -> AnalystPrediction:
        """
        Quick prediction using only Oracle Beta (fastest path).

        Use when latency is critical and full deliberation isn't needed.
        """
        return self.beta.predict_next_state(current_state, action)

    def update_oracle_accuracy(
        self,
        oracle_name: str,
        predicted: np.ndarray,
        actual: np.ndarray
    ):
        """
        Update meta-learning based on prediction accuracy.

        Called when we observe the actual outcome of an action.
        """
        error = np.linalg.norm(predicted - actual)
        accuracy = 1.0 / (1.0 + error)  # Transform to [0, 1]

        self.oracle_accuracy_history[oracle_name].append(accuracy)

        # Update Gamma's trust scores
        self.gamma.update_trust(oracle_name, accuracy)

        logger.info("Updated %s accuracy: %.3f (trust now %.3f)",
                    oracle_name, accuracy, self.gamma.oracle_trust_scores.get(oracle_name, 0))

    def get_triad_status(self) -> Dict[str, Any]:
        """Get status of all three Oracles for dashboard."""
        return {
            'alpha': {
                'name': 'The Visionary',
                'prophecy_count': len(self.alpha.prophecy_history),
                'recent_summary': self.alpha.get_prophecy_summary()
            },
            'beta': {
                'name': 'The Analyst',
                'latency_stats': self.beta.get_latency_stats()
            },
            'gamma': {
                'name': 'The Arbiter',
                'governance': self.gamma.get_governance_summary()
            },
            'consultations': len(self.consultation_history),
            'last_consultation': self.consultation_history[-1].to_dict() if self.consultation_history else None
        }

    def get_narrative_summary(self) -> str:
        """Generate narrative summary of Oracle state for display."""
        status = self.get_triad_status()

        lines = [
            "+" + "=" * 68,
            "| THE PYTHIA TRIAD - STATUS REPORT",
            "+" + "=" * 68,
            "|",
            f"| Oracle Alpha (The Visionary):",
            f"|   Prophecies generated: {status['alpha']['prophecy_count']}",
            "|",
            f"| Oracle Beta (The Analyst):",
            f"|   Mean latency: {status['beta']['latency_stats'].get('mean', 0):.3f} ms",
            f"|   P99 latency: {status['beta']['latency_stats'].get('p99', 0):.3f} ms",
            "|",
            f"| Oracle Gamma (The Arbiter):",
            f"|   Trust (Alpha): {status['gamma']['governance']['trust_scores'].get('alpha', 0):.0%}",
            f"|   Trust (Beta): {status['gamma']['governance']['trust_scores'].get('beta', 0):.0%}",
            f"|   Entropy remaining: {status['gamma']['governance']['entropy_remaining_pct']:.1f}%",
            f"|   System halted: {status['gamma']['governance']['system_halted']}",
            "|",
            f"| Total consultations: {status['consultations']}",
            "+" + "=" * 68
        ]

        return "\n".join(lines)


# ============================================================================
# Integration with Narrative Interface
# ============================================================================

class OracleNarrativeAdapter:
    """
    Adapter connecting PythiaTriad to the narrative interface.

    Translates Oracle consultations into narrative reports
    for the Myth-Maker's Voice system.
    """

    def __init__(self, triad: PythiaTriad):
        self.triad = triad

    def generate_prophecy_narrative(self, consensus: OracleConsensus) -> str:
        """Generate mythic narrative for a prophecy."""
        if not consensus.prophecies:
            return "The Oracle remains silent."

        best = consensus.prophecies[0]
        risk = best.get('black_swan_risk', 0)
        value = best.get('value', 0)

        if risk > 0.1:
            risk_narrative = (
                "Dark clouds gather on the horizon. "
                "The Visionary perceives paths that lead to shadow."
            )
        elif risk > 0.05:
            risk_narrative = (
                "The future holds uncertainty. "
                "Some threads lead to unexpected places."
            )
        else:
            risk_narrative = (
                "The path ahead is clear. "
                "The future unfolds with stability."
            )

        if value > 0:
            value_narrative = "Prosperity awaits those who follow this course."
        else:
            value_narrative = "Caution is advised; the journey may be arduous."

        return f"{risk_narrative}\n{value_narrative}"

    def consultation_to_metrics(self, consensus: OracleConsensus) -> Dict[str, Any]:
        """Convert consultation to metrics for narrative dashboard."""
        return {
            'oracle_confidence': consensus.confidence,
            'prophecy_value': consensus.prophecies[0].get('value', 0) if consensus.prophecies else 0,
            'black_swan_risk': consensus.prophecies[0].get('black_swan_risk', 0) if consensus.prophecies else 0,
            'inference_latency_ms': consensus.prediction.get('latency_ms', 0),
            'total_latency_ms': consensus.latency_ms,
            'safe': consensus.safe,
            'entropy_remaining': consensus.arbitration.get('entropy_remaining_pct', 0)
        }


# ============================================================================
# Example Usage
# ============================================================================

def example_pythia_consultation():
    """Demonstrate complete Oracle consultation."""

    print("PYTHIA TRIAD DEMONSTRATION")
    print("=" * 70)

    # Create Pythia Triad
    pythia = PythiaTriad(
        latent_dim=10,
        action_dim=4,
        num_particles=5000  # Reduced for demo
    )

    # Current state
    current_state = np.random.randn(10).astype(np.float32)
    current_state[0] = 0.5  # Moderate temperature

    observation = np.random.randn(100).astype(np.float32)

    # Consult the Oracle
    print("\nConsulting the Oracle...")
    consensus = pythia.divine(
        current_state,
        observation,
        query="Should I throttle the CPU to prevent thermal damage?",
        horizon=100
    )

    # Display narrative
    print("\n" + consensus.narrative())

    # Show triad status
    print("\n" + pythia.get_narrative_summary())

    # Quick prediction demo
    print("\nQuick prediction (Beta only):")
    action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
    quick = pythia.quick_predict(current_state, action)
    print(f"  Latency: {quick.latency_ms:.3f} ms")
    print(f"  Uncertainty: {quick.uncertainty:.3f}")


if __name__ == "__main__":
    example_pythia_consultation()
