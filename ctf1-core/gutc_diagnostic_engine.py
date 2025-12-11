#!/usr/bin/env python3
"""
gutc_diagnostic_engine.py - GUTC Diagnostic and Therapeutic Guidance Engine

Maps patient neurocognitive states onto the (λ, Π) control manifold and
provides quantitative diagnosis with precise intervention targets.

The engine performs:
1. State Classification: Maps (λ̂, Π̂) to pathological quadrants
2. Capacity Assessment: Computes C(λ, Π) for cognitive efficiency
3. Repair Vector: Recommends specific λ and Π modulation strategies

Theoretical Basis:
    - λ (Criticality): Branching ratio from neuronal avalanches
    - Π (Precision): Prior/sensory weighting from neuromodulatory gain
    - C(λ, Π): Computational capacity, peaks at criticality

Quadrant Classification:
    ┌─────────────────┬─────────────────┐
    │   ASD-like      │   Psychosis     │
    │   (λ<1, Π↑)     │   (λ>1, Π↑)     │
    │   Rigid         │   Unstable      │
    ├─────────────────┼─────────────────┤
    │   Depressive    │   Chaotic       │
    │   (λ<1, Π↓)     │   (λ>1, Π↓)     │
    │   Quiescent     │   Volatile      │
    └─────────────────┴─────────────────┘
              λ = 1 (Critical Line)

Usage:
    from gutc_diagnostic_engine import GUTCDiagnosticEngine

    engine = GUTCDiagnosticEngine()
    diagnosis = engine.diagnose(lambda_hat=3.0, pi_hat=6.5)
    print(diagnosis)

Reference:
    GUTC_Theoretical_Connections.md, Sections XIV-XVI
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto


# =============================================================================
# GUTC Control Manifold Parameters
# =============================================================================

# Normalized critical point (Branching ratio ≈ 1 maps to λ = 5)
CRITICAL_LAMBDA = 5.0

# Width of the "Healthy" critical corridor (4.0 < λ < 6.0)
LAMBDA_TOLERANCE = 1.0

# Threshold separating moderate vs high Precision
HIGH_PI_THRESHOLD = 4.5

# Low precision threshold (anhedonic/apathetic states)
LOW_PI_THRESHOLD = 2.0

# Capacity function width parameter
CAPACITY_SIGMA = 1.5


# =============================================================================
# Diagnostic Data Structures
# =============================================================================

class ClinicalState(Enum):
    """Enumeration of GUTC clinical states."""
    HEALTHY = auto()
    ASD_LIKE = auto()
    PSYCHOSIS_LIKE = auto()
    DEPRESSIVE = auto()
    CHAOTIC = auto()
    ANESTHETIC = auto()
    MANIC = auto()
    UNCLASSIFIED = auto()


@dataclass
class RepairVector:
    """Therapeutic intervention vector on the (λ, Π) manifold."""
    delta_lambda: float  # Target change in λ (positive = increase)
    delta_pi: float      # Target change in Π (positive = increase)
    lambda_strategy: str
    pi_strategy: str
    pharmacological: List[str] = field(default_factory=list)
    behavioral: List[str] = field(default_factory=list)
    neuromodulation: List[str] = field(default_factory=list)


@dataclass
class GUTCDiagnosis:
    """
    Complete GUTC diagnostic report.

    Attributes:
        state: Clinical state classification
        state_name: Human-readable state name
        lambda_hat: Estimated criticality coordinate
        pi_hat: Estimated precision coordinate
        capacity: Normalized capacity C(λ, Π) ∈ [0, 1]
        description: Clinical description of the state
        repair_vector: Recommended intervention strategy
        confidence: Diagnostic confidence (0-1)
        risk_level: Clinical risk assessment (low/moderate/high/critical)
    """
    state: ClinicalState
    state_name: str
    lambda_hat: float
    pi_hat: float
    capacity: float
    description: str
    repair_vector: RepairVector
    confidence: float = 0.0
    risk_level: str = "moderate"

    def __str__(self) -> str:
        lines = [
            "=" * 70,
            f"GUTC DIAGNOSTIC REPORT: {self.state_name}",
            "=" * 70,
            f"  Coordinates: (λ̂ = {self.lambda_hat:.2f}, Π̂ = {self.pi_hat:.2f})",
            f"  Capacity: C(λ, Π) = {self.capacity:.3f}",
            f"  Risk Level: {self.risk_level.upper()}",
            f"  Confidence: {self.confidence:.0%}",
            "",
            "  PATHOLOGY:",
            f"    {self.description}",
            "",
            "  REPAIR STRATEGY:",
            f"    λ-control: {self.repair_vector.lambda_strategy}",
            f"    Π-control: {self.repair_vector.pi_strategy}",
            f"    Target: Δλ = {self.repair_vector.delta_lambda:+.1f}, "
            f"Δ Π = {self.repair_vector.delta_pi:+.1f}",
        ]

        if self.repair_vector.pharmacological:
            lines.append("")
            lines.append("  PHARMACOLOGICAL OPTIONS:")
            for rx in self.repair_vector.pharmacological:
                lines.append(f"    • {rx}")

        if self.repair_vector.behavioral:
            lines.append("")
            lines.append("  BEHAVIORAL INTERVENTIONS:")
            for bx in self.repair_vector.behavioral:
                lines.append(f"    • {bx}")

        if self.repair_vector.neuromodulation:
            lines.append("")
            lines.append("  NEUROMODULATION OPTIONS:")
            for nm in self.repair_vector.neuromodulation:
                lines.append(f"    • {nm}")

        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# Capacity Function
# =============================================================================

def compute_capacity(lambda_hat: float, pi_hat: float,
                     sigma: float = CAPACITY_SIGMA) -> float:
    """
    Compute normalized GUTC capacity C(λ, Π).

    C(λ, Π) = Π · exp(-(λ - 5)² / 2σ²)

    Normalized to [0, 1] where 1 = maximum theoretical capacity.

    Args:
        lambda_hat: Criticality estimate
        pi_hat: Precision estimate
        sigma: Ridge width parameter

    Returns:
        Normalized capacity ∈ [0, 1]
    """
    # Raw capacity
    c_raw = pi_hat * np.exp(-((lambda_hat - CRITICAL_LAMBDA) ** 2) / (2 * sigma ** 2))

    # Theoretical maximum (at λ=5, Π=8)
    c_max = 8.0 * np.exp(0)  # = 8.0

    return min(c_raw / c_max, 1.0)


def compute_distance_to_healthy(lambda_hat: float, pi_hat: float) -> float:
    """
    Compute Euclidean distance to healthy corridor center.

    Healthy center: (λ=5.0, Π=3.0)
    """
    healthy_lambda = 5.0
    healthy_pi = 3.0
    return np.sqrt((lambda_hat - healthy_lambda) ** 2 +
                   (pi_hat - healthy_pi) ** 2)


# =============================================================================
# GUTC Diagnostic Engine
# =============================================================================

class GUTCDiagnosticEngine:
    """
    Core engine for mapping neurocognitive states to the (λ, Π) control manifold.

    This engine provides quantitative 'answers' for mental illness by:
    1. Classifying the current state (diagnosis)
    2. Computing cognitive capacity
    3. Recommending precise interventions (repair vector)
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the diagnostic engine.

        Args:
            verbose: Print initialization message
        """
        self.verbose = verbose
        if verbose:
            print("GUTC Diagnostic Engine v1.0 Initialized")
            print(f"  Critical λ: {CRITICAL_LAMBDA} (tolerance: ±{LAMBDA_TOLERANCE})")
            print(f"  High Π threshold: {HIGH_PI_THRESHOLD}")
            print(f"  Capacity σ: {CAPACITY_SIGMA}")

    def diagnose(self, lambda_hat: float, pi_hat: float) -> GUTCDiagnosis:
        """
        Map estimated state coordinates (λ̂, Π̂) to pathological regime
        and determine the necessary repair vector.

        Args:
            lambda_hat: Estimated Criticality (λ̂ ≈ Branching Ratio)
            pi_hat: Estimated Precision (Π̂, DA/ACh gain)

        Returns:
            GUTCDiagnosis with classification and repair strategy
        """
        # Compute capacity
        capacity = compute_capacity(lambda_hat, pi_hat)

        # Classify state
        is_critical = abs(lambda_hat - CRITICAL_LAMBDA) <= LAMBDA_TOLERANCE
        is_subcritical = lambda_hat < (CRITICAL_LAMBDA - LAMBDA_TOLERANCE)
        is_supercritical = lambda_hat > (CRITICAL_LAMBDA + LAMBDA_TOLERANCE)
        is_high_pi = pi_hat >= HIGH_PI_THRESHOLD
        is_low_pi = pi_hat < LOW_PI_THRESHOLD

        # Compute confidence based on distance from quadrant boundaries
        dist_to_crit = abs(lambda_hat - CRITICAL_LAMBDA)
        dist_to_pi_thresh = min(abs(pi_hat - HIGH_PI_THRESHOLD),
                                abs(pi_hat - LOW_PI_THRESHOLD))
        confidence = min(0.95, 0.5 + 0.1 * dist_to_crit + 0.05 * dist_to_pi_thresh)

        # ===== HEALTHY / OPTIMAL COGNITION =====
        if is_critical and LOW_PI_THRESHOLD <= pi_hat < HIGH_PI_THRESHOLD:
            return GUTCDiagnosis(
                state=ClinicalState.HEALTHY,
                state_name="Healthy / Optimal Cognition",
                lambda_hat=lambda_hat,
                pi_hat=pi_hat,
                capacity=capacity,
                description=(
                    "Dynamics near critical point (λ ≈ 1) with maximal computational "
                    "capacity. Moderate precision enables adaptive belief updating."
                ),
                repair_vector=RepairVector(
                    delta_lambda=0.0,
                    delta_pi=0.0,
                    lambda_strategy="MAINTAIN E/I balance (λ ≈ 1)",
                    pi_strategy="MAINTAIN adaptive gain modulation",
                    behavioral=["Continue healthy lifestyle factors",
                                "Maintain cognitive engagement"]
                ),
                confidence=confidence,
                risk_level="low"
            )

        # ===== ASD-LIKE: Subcritical + High Π =====
        if is_subcritical and is_high_pi:
            target_lambda = CRITICAL_LAMBDA
            target_pi = 3.5
            return GUTCDiagnosis(
                state=ClinicalState.ASD_LIKE,
                state_name="ASD-like / Rigid Over-Precision",
                lambda_hat=lambda_hat,
                pi_hat=pi_hat,
                capacity=capacity,
                description=(
                    "Subcritical dynamics (low capacity) with inflexible, over-weighted "
                    "priors. High sensory precision causes detail-focus and rigidity. "
                    "Characterized by repetitive behaviors and sensory hypersensitivity."
                ),
                repair_vector=RepairVector(
                    delta_lambda=target_lambda - lambda_hat,
                    delta_pi=target_pi - pi_hat,
                    lambda_strategy="INCREASE E/I ratio → push λ toward criticality",
                    pi_strategy="DECREASE precision gain → reduce PE amplification",
                    pharmacological=[
                        "Consider NMDA modulators (e.g., memantine) for E/I balance",
                        "5-HT2A partial agonists for cognitive flexibility",
                        "Bumetanide (GABA modulation) in selected cases"
                    ],
                    behavioral=[
                        "Cognitive behavioral therapy for flexibility",
                        "Gradual sensory exposure protocols",
                        "Social skills training"
                    ],
                    neuromodulation=[
                        "TMS to DLPFC for cognitive flexibility",
                        "tDCS protocols targeting E/I balance"
                    ]
                ),
                confidence=confidence,
                risk_level="moderate"
            )

        # ===== PSYCHOSIS-LIKE: Supercritical + High Π =====
        if is_supercritical and is_high_pi:
            target_lambda = CRITICAL_LAMBDA
            target_pi = 3.0
            return GUTCDiagnosis(
                state=ClinicalState.PSYCHOSIS_LIKE,
                state_name="Psychosis-like / Unstable Hyper-Plasticity",
                lambda_hat=lambda_hat,
                pi_hat=pi_hat,
                capacity=capacity,
                description=(
                    "Supercritical dynamics amplify prediction errors uncontrollably. "
                    "High prior precision leads to over-confident, aberrant belief "
                    "updates. Risk of delusions and hallucinations from aberrant salience."
                ),
                repair_vector=RepairVector(
                    delta_lambda=target_lambda - lambda_hat,
                    delta_pi=target_pi - pi_hat,
                    lambda_strategy="DECREASE E/I ratio → stabilize toward criticality",
                    pi_strategy="NORMALIZE DA gain → reduce aberrant salience",
                    pharmacological=[
                        "D2 antagonists (e.g., sulpiride, aripiprazole)",
                        "GABA-A agonists for acute stabilization",
                        "Consider glutamate modulators (lamotrigine)"
                    ],
                    behavioral=[
                        "Cognitive remediation therapy",
                        "Reality testing protocols",
                        "Stress reduction and sleep hygiene"
                    ],
                    neuromodulation=[
                        "TMS to temporal cortex for hallucinations",
                        "Consider ECT for treatment-resistant cases"
                    ]
                ),
                confidence=confidence,
                risk_level="high"
            )

        # ===== DEPRESSIVE: Subcritical + Low Π =====
        if is_subcritical and is_low_pi:
            target_lambda = CRITICAL_LAMBDA
            target_pi = 3.0
            return GUTCDiagnosis(
                state=ClinicalState.DEPRESSIVE,
                state_name="Depressive / Anhedonic Quiescence",
                lambda_hat=lambda_hat,
                pi_hat=pi_hat,
                capacity=capacity,
                description=(
                    "Severely dampened dynamics with low capacity. Low precision means "
                    "prediction errors are ignored—nothing feels salient. Characterized "
                    "by anhedonia, psychomotor retardation, and motivational deficits."
                ),
                repair_vector=RepairVector(
                    delta_lambda=target_lambda - lambda_hat,
                    delta_pi=target_pi - pi_hat,
                    lambda_strategy="BOOST excitability → increase λ toward criticality",
                    pi_strategy="INCREASE DA/NA gain → restore motivational salience",
                    pharmacological=[
                        "SSRIs/SNRIs for baseline restoration",
                        "Bupropion (DA/NA) for anhedonia",
                        "Ketamine/esketamine for rapid effect",
                        "Pramipexole (D3 agonist) for anhedonia"
                    ],
                    behavioral=[
                        "Behavioral activation therapy",
                        "Exercise prescription (↑ BDNF, ↑ DA)",
                        "Mindfulness-based cognitive therapy"
                    ],
                    neuromodulation=[
                        "TMS to left DLPFC (FDA-approved)",
                        "tDCS anodal left DLPFC",
                        "Deep TMS for treatment-resistant cases"
                    ]
                ),
                confidence=confidence,
                risk_level="moderate" if pi_hat > 1.0 else "high"
            )

        # ===== CHAOTIC: Supercritical + Low Π =====
        if is_supercritical and is_low_pi:
            target_lambda = CRITICAL_LAMBDA
            target_pi = 3.0
            return GUTCDiagnosis(
                state=ClinicalState.CHAOTIC,
                state_name="Chaotic / High Volatility Noise",
                lambda_hat=lambda_hat,
                pi_hat=pi_hat,
                capacity=capacity,
                description=(
                    "High instability (λ > 1) with poor filtering (low Π). The system "
                    "amplifies noise without stable attractors. Volatility without "
                    "delusional focus—seen in some ADHD and bipolar mixed states."
                ),
                repair_vector=RepairVector(
                    delta_lambda=target_lambda - lambda_hat,
                    delta_pi=target_pi - pi_hat,
                    lambda_strategy="REDUCE E/I ratio → stabilize dynamics",
                    pi_strategy="INCREASE filtering → establish stable PE weighting",
                    pharmacological=[
                        "Mood stabilizers (lithium, valproate)",
                        "Alpha-2 agonists (guanfacine) for filtering",
                        "Atomoxetine for NA modulation"
                    ],
                    behavioral=[
                        "Structure and routine establishment",
                        "Attention training protocols",
                        "Sleep hygiene optimization"
                    ],
                    neuromodulation=[
                        "TMS to right DLPFC for inhibitory control"
                    ]
                ),
                confidence=confidence,
                risk_level="high"
            )

        # ===== ANESTHETIC: Very low λ and Π =====
        if lambda_hat < 2.0 and pi_hat < 2.0:
            return GUTCDiagnosis(
                state=ClinicalState.ANESTHETIC,
                state_name="Anesthetic / Minimal Responsiveness",
                lambda_hat=lambda_hat,
                pi_hat=pi_hat,
                capacity=capacity,
                description=(
                    "Near-complete suppression of cortical dynamics. Minimal PE "
                    "processing. Seen in deep sedation, coma, or severe catatonia."
                ),
                repair_vector=RepairVector(
                    delta_lambda=CRITICAL_LAMBDA - lambda_hat,
                    delta_pi=3.0 - pi_hat,
                    lambda_strategy="URGENT: Restore baseline excitability",
                    pi_strategy="URGENT: Restore neuromodulatory tone",
                    pharmacological=[
                        "Reversal agents if pharmacological cause",
                        "Stimulants for arousal (methylphenidate)",
                        "Amantadine for consciousness disorders"
                    ],
                    neuromodulation=[
                        "Consider vagus nerve stimulation",
                        "Deep brain stimulation (research)"
                    ]
                ),
                confidence=confidence,
                risk_level="critical"
            )

        # ===== MANIC: Critical but unstable Π =====
        if is_critical and pi_hat > 6.0:
            return GUTCDiagnosis(
                state=ClinicalState.MANIC,
                state_name="Manic / Hyperactive Processing",
                lambda_hat=lambda_hat,
                pi_hat=pi_hat,
                capacity=capacity,
                description=(
                    "Near-critical dynamics but with excessive precision weighting. "
                    "Everything feels maximally salient. Pressured cognition and "
                    "disinhibition without frank psychosis."
                ),
                repair_vector=RepairVector(
                    delta_lambda=0.0,
                    delta_pi=3.0 - pi_hat,
                    lambda_strategy="MAINTAIN λ (already near-critical)",
                    pi_strategy="REDUCE precision gain → normalize salience",
                    pharmacological=[
                        "Mood stabilizers (lithium, valproate)",
                        "Atypical antipsychotics (low dose)",
                        "Benzodiazepines for acute agitation"
                    ],
                    behavioral=[
                        "Sleep prioritization",
                        "Stimulus reduction",
                        "Structure and containment"
                    ]
                ),
                confidence=confidence,
                risk_level="high"
            )

        # ===== UNCLASSIFIED =====
        return GUTCDiagnosis(
            state=ClinicalState.UNCLASSIFIED,
            state_name="Unclassified State",
            lambda_hat=lambda_hat,
            pi_hat=pi_hat,
            capacity=capacity,
            description=(
                f"State at (λ={lambda_hat:.1f}, Π={pi_hat:.1f}) does not clearly "
                "map to established pathological quadrants. Further assessment needed."
            ),
            repair_vector=RepairVector(
                delta_lambda=CRITICAL_LAMBDA - lambda_hat,
                delta_pi=3.0 - pi_hat,
                lambda_strategy="Target λ → critical point",
                pi_strategy="Target Π → moderate range",
            ),
            confidence=0.5,
            risk_level="moderate"
        )

    def batch_diagnose(self, coordinates: List[Tuple[float, float]]) -> List[GUTCDiagnosis]:
        """
        Diagnose multiple subjects.

        Args:
            coordinates: List of (lambda_hat, pi_hat) tuples

        Returns:
            List of GUTCDiagnosis objects
        """
        return [self.diagnose(lam, pi) for lam, pi in coordinates]

    def compute_population_statistics(self,
                                      diagnoses: List[GUTCDiagnosis]) -> Dict:
        """
        Compute population-level statistics from diagnoses.

        Args:
            diagnoses: List of GUTCDiagnosis objects

        Returns:
            Dictionary with population statistics
        """
        n = len(diagnoses)
        if n == 0:
            return {}

        lambdas = [d.lambda_hat for d in diagnoses]
        pis = [d.pi_hat for d in diagnoses]
        capacities = [d.capacity for d in diagnoses]

        state_counts = {}
        for d in diagnoses:
            state_counts[d.state_name] = state_counts.get(d.state_name, 0) + 1

        return {
            "n": n,
            "lambda_mean": np.mean(lambdas),
            "lambda_std": np.std(lambdas),
            "pi_mean": np.mean(pis),
            "pi_std": np.std(pis),
            "capacity_mean": np.mean(capacities),
            "capacity_std": np.std(capacities),
            "state_distribution": state_counts,
            "healthy_fraction": state_counts.get("Healthy / Optimal Cognition", 0) / n
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Demonstration of GUTC Diagnostic Engine."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GUTC Diagnostic Engine - Map (λ, Π) to clinical states"
    )
    parser.add_argument("--lambda", dest="lambda_hat", type=float,
                        help="Criticality estimate (λ̂)")
    parser.add_argument("--pi", dest="pi_hat", type=float,
                        help="Precision estimate (Π̂)")
    parser.add_argument("--demo", action="store_true",
                        help="Run demonstration with example cases")

    args = parser.parse_args()

    engine = GUTCDiagnosticEngine()

    if args.demo or (args.lambda_hat is None and args.pi_hat is None):
        # Demonstration cases
        cases = [
            (3.0, 6.5, "ASD-like"),
            (7.5, 6.0, "Psychosis-like"),
            (4.8, 3.5, "Healthy"),
            (3.0, 1.5, "Depressive"),
            (7.0, 1.8, "Chaotic"),
            (1.5, 0.8, "Anesthetic"),
        ]

        print("\n" + "=" * 70)
        print("GUTC DIAGNOSTIC ENGINE - DEMONSTRATION")
        print("=" * 70)

        for lam, pi, expected in cases:
            diagnosis = engine.diagnose(lam, pi)
            print(f"\n[Input: λ̂={lam:.1f}, Π̂={pi:.1f} | Expected: {expected}]")
            print(diagnosis)

    else:
        # Single diagnosis
        if args.lambda_hat is not None and args.pi_hat is not None:
            diagnosis = engine.diagnose(args.lambda_hat, args.pi_hat)
            print(diagnosis)
        else:
            print("Error: Provide both --lambda and --pi, or use --demo")


if __name__ == "__main__":
    main()
