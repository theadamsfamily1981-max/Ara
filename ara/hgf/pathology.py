"""
ara.hgf.pathology - Pathological Parameter Presets

Defines parameter configurations that model computational abnormalities
associated with psychiatric disorders according to the Bayesian brain hypothesis.

These are theoretical mappings based on computational psychiatry literature:
- Schizophrenia: Aberrant precision weighting
- BPD: Hypersensitive volatility coupling
- Anxiety: Elevated prior precision
- Depression: Reduced precision/learning rate
- Autism: Elevated sensory precision

IMPORTANT: These are theoretical models for research and simulation purposes.
They do NOT constitute clinical diagnostic criteria.

References:
    Adams, R. A., et al. (2013). The computational anatomy of psychosis.
    Frontiers in Psychiatry, 4, 47.

    Lawson, R. P., et al. (2017). Adults with autism overestimate the
    volatility of the sensory environment. Nature Neuroscience.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ara.hgf.core import HGFParams


@dataclass
class PathologyPreset:
    """
    A preset parameter configuration modeling a computational phenotype.

    Attributes:
        name: Human-readable name
        code: Short code (e.g., "SCZ_RIGID")
        description: Description of the computational abnormality
        params: HGF parameters
        clinical_correlates: Expected clinical correlations
        mechanism: Brief description of the mechanism
    """
    name: str
    code: str
    description: str
    params: HGFParams
    clinical_correlates: dict
    mechanism: str

    def create_agent(self):
        """Create an HGF agent with this preset's parameters."""
        from ara.hgf.agents import HGFAgent
        return HGFAgent(params=self.params)


# =============================================================================
# Healthy Baseline
# =============================================================================

HEALTHY_BASELINE = PathologyPreset(
    name="Healthy Baseline",
    code="HEALTHY",
    description="Typical parameter values representing healthy inference",
    params=HGFParams(
        omega_2=-4.0,  # Moderate baseline volatility
        kappa_1=1.0,   # Normal coupling
        theta=1.0,     # Balanced exploration/exploitation
        omega_3=-6.0,  # Slow meta-learning
        mu_2_0=0.0,    # Neutral prior
        sigma_2_0=1.0, # Moderate prior uncertainty
    ),
    clinical_correlates={
        "PANSS": "low",
        "BDI": "low",
        "STAI": "low",
    },
    mechanism="Balanced precision weighting allows flexible adaptation to "
              "environmental statistics while maintaining stable beliefs."
)


# =============================================================================
# Schizophrenia Spectrum
# =============================================================================

SCHIZOPHRENIA_RIGID = PathologyPreset(
    name="Schizophrenia (Rigid Priors)",
    code="SCZ_RIGID",
    description="Overly precise priors that resist updating - associated with "
                "negative symptoms and cognitive rigidity",
    params=HGFParams(
        omega_2=-6.0,  # Very low volatility = resistant to change
        kappa_1=0.3,   # Weak coupling = ignores volatility signals
        theta=0.5,     # Low temperature = random/disorganized choices
        omega_3=-8.0,  # Extremely slow meta-learning
        mu_2_0=0.0,
        sigma_2_0=0.5, # Tight prior
    ),
    clinical_correlates={
        "PANSS_negative": "high",
        "PANSS_positive": "low",
        "cognitive_flexibility": "impaired",
    },
    mechanism="Excessively precise priors (low ω₂) lead to slow learning and "
              "failure to update beliefs in response to prediction errors. "
              "Associated with negative symptoms and cognitive inflexibility."
)

SCHIZOPHRENIA_LOOSE = PathologyPreset(
    name="Schizophrenia (Loose Priors)",
    code="SCZ_LOOSE",
    description="Imprecise priors leading to over-weighting of sensory evidence - "
                "associated with positive symptoms (hallucinations, delusions)",
    params=HGFParams(
        omega_2=-2.0,  # High volatility = beliefs are unstable
        kappa_1=2.0,   # Strong coupling = over-sensitive to volatility
        theta=2.0,     # High temperature = deterministic but wrong
        omega_3=-4.0,  # Fast meta-learning
        mu_2_0=0.0,
        sigma_2_0=2.0, # Loose prior
    ),
    clinical_correlates={
        "PANSS_negative": "low",
        "PANSS_positive": "high",
        "aberrant_salience": "high",
    },
    mechanism="Imprecise priors (high ω₂) cause over-weighting of sensory input, "
              "leading to aberrant salience and potentially hallucinations. "
              "False patterns are detected in noise."
)


# =============================================================================
# Borderline Personality Disorder
# =============================================================================

BPD_HIGH_KAPPA = PathologyPreset(
    name="Borderline Personality Disorder",
    code="BPD",
    description="Pathologically high κ₁ creates hypersensitivity to perceived "
                "volatility, causing rapid belief swings and affective instability",
    params=HGFParams(
        omega_2=-4.0,  # Normal baseline
        kappa_1=3.0,   # CRITICAL: Very high coupling
        theta=1.5,     # Slightly impulsive
        omega_3=-4.0,  # Fast volatility learning
        mu_2_0=0.0,
        sigma_2_0=1.5, # Slightly uncertain
    ),
    clinical_correlates={
        "affective_instability": "high",
        "interpersonal_instability": "high",
        "identity_disturbance": "high",
        "impulsivity": "moderate",
    },
    mechanism="High κ₁ means tiny changes in perceived volatility (μ₃) cause "
              "exponential increases in prior variance (σ̂₂²). This creates a "
              "'hair trigger' for belief revision - minor events cause dramatic "
              "swings in beliefs about self, others, and the world."
)


# =============================================================================
# Anxiety Disorders
# =============================================================================

ANXIETY_HIGH_PRECISION = PathologyPreset(
    name="Anxiety (Elevated Threat Precision)",
    code="ANXIETY",
    description="Elevated precision on threat-related predictions leads to "
                "hypervigilance and catastrophizing",
    params=HGFParams(
        omega_2=-5.0,  # Somewhat rigid priors
        kappa_1=1.5,   # Slightly elevated coupling
        theta=2.0,     # More deterministic/less exploratory
        omega_3=-5.0,  # Moderate meta-learning
        mu_2_0=0.5,    # Prior bias toward threat (logit > 0)
        sigma_2_0=0.8, # Fairly certain prior
    ),
    clinical_correlates={
        "STAI_state": "high",
        "STAI_trait": "high",
        "threat_bias": "present",
        "uncertainty_intolerance": "high",
    },
    mechanism="Elevated precision on threat-related predictions means the "
              "system is highly confident that bad things will happen. "
              "Prediction errors that disconfirm threat are down-weighted."
)


# =============================================================================
# Depression
# =============================================================================

DEPRESSION_LOW_PRECISION = PathologyPreset(
    name="Depression (Reduced Precision)",
    code="DEPRESSION",
    description="Globally reduced precision leads to learned helplessness - "
                "actions feel like they don't matter",
    params=HGFParams(
        omega_2=-3.0,  # Higher volatility = beliefs don't stabilize
        kappa_1=0.5,   # Low coupling = reduced learning from volatility
        theta=0.3,     # Very low temperature = random/unmotivated choices
        omega_3=-7.0,  # Slow meta-learning
        mu_2_0=-0.5,   # Prior bias toward negative (logit < 0)
        sigma_2_0=2.0, # High uncertainty
    ),
    clinical_correlates={
        "BDI": "high",
        "anhedonia": "high",
        "learned_helplessness": "present",
        "motivation": "low",
    },
    mechanism="Low θ (inverse temperature) means actions are essentially "
              "random because beliefs about outcomes don't translate to "
              "behavioral preferences. Negative prior bias (μ₂_0 < 0) "
              "creates expectation of negative outcomes."
)


# =============================================================================
# Autism Spectrum
# =============================================================================

AUTISM_HIGH_SENSORY_PRECISION = PathologyPreset(
    name="Autism (Elevated Sensory Precision)",
    code="AUTISM",
    description="Elevated sensory precision leads to over-weighting of bottom-up "
                "signals and difficulty with top-down prediction",
    params=HGFParams(
        omega_2=-5.5,  # Somewhat rigid priors
        kappa_1=0.5,   # Low coupling = difficulty with context
        theta=1.5,     # Slightly elevated
        omega_3=-7.0,  # Slow volatility learning
        mu_2_0=0.0,
        sigma_2_0=0.5, # Tight priors
        # Note: π₁ (sensory precision) would be elevated
        # This is modeled indirectly through ω parameters
    ),
    clinical_correlates={
        "sensory_sensitivity": "high",
        "preference_for_sameness": "high",
        "detail_focus": "high",
        "context_integration": "impaired",
    },
    mechanism="Elevated sensory precision (π₁) relative to prior precision "
              "means sensory details are over-weighted. Each deviation from "
              "prediction generates strong PE, experienced as overwhelming. "
              "Reduced κ₁ impairs volatility learning, creating preference "
              "for predictable environments."
)


# =============================================================================
# ADHD
# =============================================================================

ADHD_VOLATILE = PathologyPreset(
    name="ADHD (Elevated Volatility Estimation)",
    code="ADHD",
    description="Tendency to overestimate environmental volatility leads to "
                "excessive exploration and difficulty sustaining attention",
    params=HGFParams(
        omega_2=-3.0,  # High volatility estimation
        kappa_1=1.5,   # Elevated coupling
        theta=0.8,     # More exploratory
        omega_3=-4.0,  # Fast meta-learning
        mu_3_0=2.0,    # High initial volatility estimate
        sigma_2_0=1.5,
    ),
    clinical_correlates={
        "inattention": "high",
        "hyperactivity": "moderate",
        "impulsivity": "moderate",
        "novelty_seeking": "high",
    },
    mechanism="Elevated μ₃ (volatility estimate) causes the system to expect "
              "rapid environmental change. This drives exploration over "
              "exploitation and makes it difficult to sustain attention on "
              "tasks that appear 'stable' and thus boring."
)


# =============================================================================
# Preset Registry
# =============================================================================

PRESETS = {
    "HEALTHY": HEALTHY_BASELINE,
    "SCZ_RIGID": SCHIZOPHRENIA_RIGID,
    "SCZ_LOOSE": SCHIZOPHRENIA_LOOSE,
    "BPD": BPD_HIGH_KAPPA,
    "ANXIETY": ANXIETY_HIGH_PRECISION,
    "DEPRESSION": DEPRESSION_LOW_PRECISION,
    "AUTISM": AUTISM_HIGH_SENSORY_PRECISION,
    "ADHD": ADHD_VOLATILE,
}


def get_preset(code: str) -> PathologyPreset:
    """Get a preset by its code."""
    if code not in PRESETS:
        raise ValueError(f"Unknown preset: {code}. Available: {list(PRESETS.keys())}")
    return PRESETS[code]


def list_presets() -> list:
    """List all available presets."""
    return [(code, preset.name) for code, preset in PRESETS.items()]


def compare_presets(codes: list) -> dict:
    """
    Compare parameters across multiple presets.

    Args:
        codes: List of preset codes to compare

    Returns:
        Dictionary with parameter comparison
    """
    comparison = {
        "presets": codes,
        "omega_2": {},
        "kappa_1": {},
        "theta": {},
        "omega_3": {},
    }

    for code in codes:
        preset = get_preset(code)
        comparison["omega_2"][code] = preset.params.omega_2
        comparison["kappa_1"][code] = preset.params.kappa_1
        comparison["theta"][code] = preset.params.theta
        comparison["omega_3"][code] = preset.params.omega_3

    return comparison
