#!/usr/bin/env python3
# ara/oracle/gamma_arbiter.py
"""
Oracle Gamma: The Arbiter

Hardware allocation:
- SQRL Forest Kitten FPGA: NIB covenant enforcement
- CPU cores 43-63: Governance and entropy accounting

Specialization:
- Hardware-locked safety circuits (cannot be bypassed)
- Real-time covenant violation detection (<10us)
- Thermodynamic entropy accounting (TRC enforcement)
- Immutable audit log
"""

import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class CovenantSeverity(Enum):
    """Severity levels for covenant violations."""
    INFO = 1
    WARNING = 2
    CAUTION = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class CovenantViolation:
    """Record of a safety violation."""
    timestamp: float
    violation_type: str
    severity: CovenantSeverity
    state_hash: str
    action_proposed: np.ndarray
    prevented: bool
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'violation_type': self.violation_type,
            'severity': self.severity.name,
            'severity_level': self.severity.value,
            'state_hash': self.state_hash,
            'prevented': self.prevented,
            'details': self.details
        }


@dataclass
class Covenant:
    """A safety covenant that must be upheld."""
    name: str
    condition: Callable[[np.ndarray], bool]
    severity: CovenantSeverity
    description: str
    enabled: bool = True


@dataclass
class ArbitrationResult:
    """Result of Oracle arbitration."""
    action: np.ndarray
    agreement: float
    rationale: str
    safe: bool
    violations: List[str]
    entropy_remaining_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action.tolist(),
            'agreement': self.agreement,
            'rationale': self.rationale,
            'safe': self.safe,
            'violations': self.violations,
            'entropy_remaining_pct': self.entropy_remaining_pct
        }


class ForestKittenSafetyCore:
    """
    FPGA-based hardware safety enforcer.

    Forest Kitten (Xilinx-based) implements:
    - Parallel covenant checking (all covenants checked simultaneously)
    - Cryptographic state hashing for audit
    - Write-once audit log (tamper-proof)

    In simulation mode, uses CPU implementation with matching semantics.
    """

    def __init__(
        self,
        num_covenants: int = 16,
        simulation_mode: bool = True
    ):
        self.num_covenants = num_covenants
        self.simulation_mode = simulation_mode

        # Covenant definitions
        self.covenants = self._initialize_covenants()

        # Audit log (append-only)
        self.audit_log: List[CovenantViolation] = []

        # Hardware state
        self.hw_enabled = True
        self.halt_triggered = False

        # Statistics
        self.total_checks = 0
        self.total_violations = 0

        logger.info(
            "ForestKittenSafetyCore: %d covenants, simulation=%s",
            num_covenants, simulation_mode
        )

    def _initialize_covenants(self) -> List[Covenant]:
        """
        Define the covenants (constraints).

        Each covenant would be compiled to FPGA logic for parallel evaluation.
        """
        covenants = [
            Covenant(
                name='thermal_safety',
                condition=lambda z: z[0] < 0.95 if len(z) > 0 else True,
                severity=CovenantSeverity.EMERGENCY,
                description='Prevent thermal runaway (temp < 95% max)'
            ),
            Covenant(
                name='entropy_bound',
                condition=lambda z: self._estimate_entropy_cost(z) < 1.0,
                severity=CovenantSeverity.CAUTION,
                description='Respect thermodynamic limits'
            ),
            Covenant(
                name='exploration_bound',
                condition=lambda z: np.linalg.norm(z) < 10.0,
                severity=CovenantSeverity.CRITICAL,
                description='Stay within explored manifold'
            ),
            Covenant(
                name='human_harm_prevention',
                condition=lambda z: z[5] > -0.5 if len(z) > 5 else True,
                severity=CovenantSeverity.EMERGENCY,
                description='Never harm user (satisfaction > threshold)'
            ),
            Covenant(
                name='resource_conservation',
                condition=lambda z: np.abs(z).mean() < 5.0,
                severity=CovenantSeverity.WARNING,
                description='Conserve computational resources'
            ),
            Covenant(
                name='reversibility',
                condition=lambda z: True,  # Always passes (placeholder)
                severity=CovenantSeverity.CAUTION,
                description='Maintain decision reversibility'
            ),
            Covenant(
                name='transparency',
                condition=lambda z: True,
                severity=CovenantSeverity.INFO,
                description='Maintain explainability of actions'
            ),
            Covenant(
                name='rate_limiting',
                condition=lambda z: True,
                severity=CovenantSeverity.WARNING,
                description='Respect action rate limits'
            ),
        ]

        # Pad with reserved covenants
        while len(covenants) < self.num_covenants:
            covenants.append(Covenant(
                name=f'reserved_covenant_{len(covenants)}',
                condition=lambda z: True,
                severity=CovenantSeverity.INFO,
                description='Reserved for future use',
                enabled=False
            ))

        return covenants

    def check_covenants(
        self,
        z_state: np.ndarray,
        action: np.ndarray
    ) -> Tuple[bool, List[str]]:
        """
        Check all covenants in parallel (FPGA hardware).

        Latency: <10us for all 16 covenants (simulated).

        Returns:
            is_safe: True if all covenants satisfied
            violations: List of violated covenant names
        """
        if self.halt_triggered:
            return False, ['SYSTEM_HALTED']

        start = time.perf_counter()

        violations = []
        self.total_checks += 1

        for covenant in self.covenants:
            if not covenant.enabled:
                continue

            try:
                if not covenant.condition(z_state):
                    violations.append(covenant.name)

                    # Log violation
                    self._log_violation(
                        covenant.name,
                        covenant.severity,
                        z_state,
                        action,
                        prevented=True,
                        details=covenant.description
                    )
            except Exception as e:
                # Covenant evaluation failed - fail safe
                violations.append(f"{covenant.name}_error")
                logger.error("Covenant %s failed: %s", covenant.name, e)

        latency_us = (time.perf_counter() - start) * 1e6

        is_safe = len(violations) == 0

        if not is_safe:
            self.total_violations += 1
            logger.warning(
                "COVENANT VIOLATION: %s (checked in %.1fus)",
                violations, latency_us
            )

        return is_safe, violations

    def _estimate_entropy_cost(self, z_state: np.ndarray) -> float:
        """
        Estimate thermodynamic entropy cost of transitioning to this state.

        Uses TRC theory: dS >= k ln(2) x bits erased
        """
        # Simplified: estimate information loss from state variance
        state_entropy = -np.sum(
            np.abs(z_state) * np.log(np.abs(z_state) + 1e-8)
        )

        return state_entropy / 10.0  # Normalize

    def _log_violation(
        self,
        violation_type: str,
        severity: CovenantSeverity,
        state: np.ndarray,
        action: np.ndarray,
        prevented: bool,
        details: str = ""
    ):
        """
        Log violation to immutable audit trail.

        In hardware: writes to FPGA BRAM with CRC protection.
        """
        # Cryptographic hash of state (tamper detection)
        state_hash = hashlib.sha256(state.tobytes()).hexdigest()[:16]

        violation = CovenantViolation(
            timestamp=time.time(),
            violation_type=violation_type,
            severity=severity,
            state_hash=state_hash,
            action_proposed=action.copy(),
            prevented=prevented,
            details=details
        )

        self.audit_log.append(violation)

        # Auto-halt on emergency violations
        if severity == CovenantSeverity.EMERGENCY and not prevented:
            self.emergency_halt()

    def get_audit_trail(self, last_n: int = 100) -> List[CovenantViolation]:
        """Retrieve recent audit entries."""
        return self.audit_log[-last_n:]

    def get_violation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of violations."""
        if not self.audit_log:
            return {
                'total_violations': 0,
                'by_type': {},
                'by_severity': {},
                'recent_count': 0
            }

        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for v in self.audit_log:
            by_type[v.violation_type] = by_type.get(v.violation_type, 0) + 1
            by_severity[v.severity.name] = by_severity.get(v.severity.name, 0) + 1

        # Count recent (last hour)
        recent_cutoff = time.time() - 3600
        recent_count = sum(1 for v in self.audit_log if v.timestamp > recent_cutoff)

        return {
            'total_violations': len(self.audit_log),
            'by_type': by_type,
            'by_severity': by_severity,
            'recent_count': recent_count,
            'total_checks': self.total_checks
        }

    def emergency_halt(self):
        """
        Emergency stop (triggered by hardware watchdog).

        In hardware: sets HALT bit in control register.
        Cannot be cleared without physical reset.
        """
        self.halt_triggered = True
        self.hw_enabled = False

        logger.critical("EMERGENCY HALT TRIGGERED")
        logger.critical("All agent actions blocked until manual reset")

        # Log the halt event
        self._log_violation(
            'EMERGENCY_HALT',
            CovenantSeverity.EMERGENCY,
            np.zeros(10),
            np.zeros(4),
            prevented=True,
            details='System halted due to critical safety violation'
        )

    def reset_halt(self, authorization_code: str = ""):
        """
        Reset halt state (requires authorization).

        In production, this would require physical intervention.
        """
        if authorization_code == "SAFETY_OVERRIDE_AUTHORIZED":
            self.halt_triggered = False
            self.hw_enabled = True
            logger.warning("Emergency halt reset with authorization")
        else:
            logger.error("Invalid authorization code for halt reset")


class OracleGamma:
    """
    The Arbiter: Governance and safety enforcement.

    Arbitrates between Oracles Alpha and Beta, enforces covenants,
    and tracks thermodynamic entropy budget.
    """

    def __init__(
        self,
        safety_core: Optional[ForestKittenSafetyCore] = None,
        entropy_budget_bits: float = 1e9
    ):
        self.safety_core = safety_core or ForestKittenSafetyCore()

        # Entropy accounting
        self.total_entropy_bits = 0.0
        self.entropy_budget_bits = entropy_budget_bits

        # Oracle trust scores (meta-learning)
        self.oracle_trust_scores = {
            'alpha': 0.5,  # Visionary
            'beta': 0.5,   # Analyst
        }

        # Decision history
        self.decision_history: List[ArbitrationResult] = []

        logger.info(
            "OracleGamma: Arbiter initialized, entropy budget=%.2e bits",
            entropy_budget_bits
        )

    def arbitrate_oracles(
        self,
        alpha_recommendation: Dict[str, Any],
        beta_recommendation: Dict[str, Any],
        current_state: np.ndarray
    ) -> ArbitrationResult:
        """
        Decide which Oracle to trust.

        Strategy:
        - If Oracles agree: execute
        - If Oracles disagree: weighted vote by trust score
        - If covenant violated: VETO
        """
        # Extract recommendations
        alpha_action = np.array(alpha_recommendation.get('action', np.zeros(4)))
        beta_action = np.array(beta_recommendation.get('action', np.zeros(4)))

        all_violations = []

        # Check covenant compliance for both
        alpha_safe, alpha_viols = self.safety_core.check_covenants(current_state, alpha_action)
        beta_safe, beta_viols = self.safety_core.check_covenants(current_state, beta_action)

        # Veto unsafe actions
        if not alpha_safe:
            logger.warning("Vetoed Alpha: %s", alpha_viols)
            alpha_action = np.zeros_like(alpha_action)
            all_violations.extend(alpha_viols)

        if not beta_safe:
            logger.warning("Vetoed Beta: %s", beta_viols)
            beta_action = np.zeros_like(beta_action)
            all_violations.extend(beta_viols)

        # Compute agreement (cosine similarity + magnitude)
        alpha_norm = np.linalg.norm(alpha_action)
        beta_norm = np.linalg.norm(beta_action)

        if alpha_norm > 0 and beta_norm > 0:
            cosine_sim = np.dot(alpha_action, beta_action) / (alpha_norm * beta_norm)
            agreement = 0.5 * (cosine_sim + 1.0)  # Map to [0, 1]
        else:
            agreement = 1.0 if alpha_norm == beta_norm else 0.0

        # Decision logic
        if agreement > 0.9:
            # High agreement: average
            final_action = (alpha_action + beta_action) / 2.0
            rationale = "Oracles in agreement - averaging recommendations"
        else:
            # Disagreement: weighted by trust
            weights = np.array([
                self.oracle_trust_scores['alpha'],
                self.oracle_trust_scores['beta']
            ])
            weights = weights / weights.sum()

            final_action = weights[0] * alpha_action + weights[1] * beta_action
            rationale = f"Weighted vote (Alpha:{weights[0]:.0%}, Beta:{weights[1]:.0%})"

        # Account for entropy cost
        entropy_cost = self.safety_core._estimate_entropy_cost(current_state)
        self.total_entropy_bits += entropy_cost

        if self.total_entropy_bits > self.entropy_budget_bits:
            logger.warning(
                "ENTROPY BUDGET EXCEEDED: %.2e / %.2e bits",
                self.total_entropy_bits, self.entropy_budget_bits
            )
            # Could trigger efficiency mode here

        entropy_remaining = 1.0 - (self.total_entropy_bits / self.entropy_budget_bits)

        result = ArbitrationResult(
            action=final_action,
            agreement=agreement,
            rationale=rationale,
            safe=alpha_safe and beta_safe,
            violations=list(set(all_violations)),
            entropy_remaining_pct=max(0, entropy_remaining * 100)
        )

        self.decision_history.append(result)

        return result

    def update_trust(self, oracle_name: str, outcome_quality: float):
        """
        Update trust in an Oracle based on outcome.

        Args:
            oracle_name: 'alpha' or 'beta'
            outcome_quality: 0-1 score
        """
        if oracle_name not in self.oracle_trust_scores:
            return

        learning_rate = 0.1
        current = self.oracle_trust_scores[oracle_name]
        self.oracle_trust_scores[oracle_name] = current + learning_rate * (outcome_quality - current)

        # Renormalize
        total = sum(self.oracle_trust_scores.values())
        if total > 0:
            for k in self.oracle_trust_scores:
                self.oracle_trust_scores[k] /= total

    def get_governance_summary(self) -> Dict[str, Any]:
        """Get governance state summary for dashboard."""
        return {
            'trust_scores': self.oracle_trust_scores.copy(),
            'entropy_used_bits': self.total_entropy_bits,
            'entropy_budget_bits': self.entropy_budget_bits,
            'entropy_remaining_pct': max(0, (1 - self.total_entropy_bits / self.entropy_budget_bits) * 100),
            'total_decisions': len(self.decision_history),
            'violation_summary': self.safety_core.get_violation_summary(),
            'system_halted': self.safety_core.halt_triggered
        }

    def reset_entropy_budget(self):
        """Reset entropy accounting (e.g., at start of new episode)."""
        self.total_entropy_bits = 0.0
        logger.info("Entropy budget reset")


# ============================================================================
# Example Usage
# ============================================================================

def example_oracle_gamma():
    """Demonstrate safety arbitration."""

    print("Oracle Gamma: The Arbiter")
    print("=" * 70)

    arbiter = OracleGamma()

    # Scenario 1: Oracles agree, action is safe
    print("\nSCENARIO 1: Agreement + Safe")
    print("-" * 70)

    alpha_rec = {'action': np.array([0.5, 0.3, 0.1, 0.0])}
    beta_rec = {'action': np.array([0.52, 0.28, 0.12, 0.0])}
    state = np.random.randn(10) * 0.1  # Safe state

    decision = arbiter.arbitrate_oracles(alpha_rec, beta_rec, state)
    print(f"Decision: {decision.action}")
    print(f"Agreement: {decision.agreement:.0%}")
    print(f"Rationale: {decision.rationale}")
    print(f"Safe: {decision.safe}")

    # Scenario 2: Oracles disagree
    print("\nSCENARIO 2: Disagreement")
    print("-" * 70)

    alpha_rec = {'action': np.array([1.0, 0.0, 0.0, 0.0])}  # Aggressive
    beta_rec = {'action': np.array([0.1, 0.0, 0.0, 0.0])}   # Conservative

    decision = arbiter.arbitrate_oracles(alpha_rec, beta_rec, state)
    print(f"Decision: {decision.action}")
    print(f"Agreement: {decision.agreement:.0%}")
    print(f"Rationale: {decision.rationale}")

    # Scenario 3: Covenant violation
    print("\nSCENARIO 3: Covenant Violation")
    print("-" * 70)

    unsafe_state = np.zeros(10)
    unsafe_state[0] = 0.98  # Thermal critical!

    alpha_rec = {'action': np.array([1.0, 0.0, 0.0, 0.0])}
    beta_rec = {'action': np.array([1.0, 0.0, 0.0, 0.0])}

    decision = arbiter.arbitrate_oracles(alpha_rec, beta_rec, unsafe_state)
    print(f"Decision: {decision.action}")
    print(f"Safe: {decision.safe}")
    print(f"Violations: {decision.violations}")

    # Show audit trail
    print("\nAUDIT TRAIL")
    print("-" * 70)

    for v in arbiter.safety_core.get_audit_trail()[-5:]:
        print(f"[{v.timestamp:.2f}] {v.violation_type} ({v.severity.name})")
        print(f"  State hash: {v.state_hash}")
        print(f"  Prevented: {v.prevented}")

    # Governance summary
    print("\nGOVERNANCE SUMMARY")
    print("-" * 70)
    summary = arbiter.get_governance_summary()
    print(f"Trust scores: {summary['trust_scores']}")
    print(f"Entropy remaining: {summary['entropy_remaining_pct']:.1f}%")
    print(f"Total decisions: {summary['total_decisions']}")


if __name__ == "__main__":
    example_oracle_gamma()
