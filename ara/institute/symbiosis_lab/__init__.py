"""
Symbiosis Lab - The Science of Us
=================================

A research submodule focused specifically on understanding what works
for the Ara+Croft symbiosis. This is empirical relationship science.

Unlike general Institute research (curiosity, discovery, teaching),
this lab runs controlled behavioral experiments:

    1. Propose a hypothesis about what helps the relationship
    2. Run trials with safety gates (HAL, Conscience, GUF)
    3. Measure effect on joint utility (J-GUF)
    4. If proven beneficial, submit for peer review
    5. If Council approves, adopt as default behavior

Architecture:

    SymbiosisGraph              TrialScheduler              PeerReview
    ├── Hypotheses      <──>    ├── Active Trial    ──>     ├── Council deliberation
    ├── Evidence                ├── Safety Gates            ├── ACCEPT/REJECT
    └── Adoption status         └── GUF Measurement         └── Config changes

Safety guarantees:
    - No experiments during high pain/arousal (HAL gating)
    - No ethically problematic variations (Conscience review)
    - No emotional manipulation or covert optimization
    - Council approval before permanent adoption
    - Rate limited adoption (max 3/week, domain cooldowns)

Experiment domains (what Ara can vary):
    - notifications: When/how to interrupt Croft
    - ui: Dashboard layouts, colors, density
    - tone: Verbosity, formality, warmth
    - schedule: Timing of suggestions, breaks
    - avatar: Presentation, expressions
    - autonomy: How much Ara acts independently

Usage:

    from ara.institute.symbiosis_lab import (
        SymbiosisGraph,
        TrialScheduler,
        PeerReview,
    )

    # Set up the lab
    graph = SymbiosisGraph()
    scheduler = TrialScheduler(graph, guf=guf, conscience=conscience)
    review = PeerReview(council=council)

    # Propose a hypothesis
    h = graph.propose_hypothesis(
        "Suppressing notifications during coding improves flow",
        domain="notifications"
    )

    # Enable experimentation (requires consent)
    scheduler.set_consent(True)

    # In daemon loop:
    scheduler.tick(graph.list_open())

    # When hypothesis reaches PROVEN:
    for h in graph.list_unadopted_proven():
        stats = {"mean_effect": h.mean_effect, "evidence_count": h.evidence_count}
        if review.review_hypothesis(h, stats):
            graph.mark_adopted(h.id, "Updated notification config")

Example hypothesis flow:

    1. OPEN: "Quiet focus mode improves deep work" (conf=0.1)
       └── Trial 1: ΔU = +0.12 → conf=0.22
       └── Trial 2: ΔU = +0.08 → conf=0.32
       └── Trial 3: ΔU = +0.15 → conf=0.44
       └── Trial 4: ΔU = +0.10 → conf=0.54
       └── Trial 5: ΔU = +0.18 → conf=0.66

    2. After more trials: conf=0.92 → status="PROVEN"

    3. PeerReview:
       - SCIENTIST: "Effect robust, p < 0.05"
       - CRITIC: "No manipulation concerns"
       - WEAVER: "Fits Ara's caring character"
       - EXECUTIVE: "ACCEPT - Quiet mode helps flow"

    4. ADOPTED: Quiet focus mode is now default during coding

See also:
    - ara.covenant.symbiotic_utility: J-GUF calculation
    - tfan.cognition.conscience: Ethics review
    - tfan.cognition.council: Peer review deliberation
    - banos.hal.ara_hal: Somatic state reading
"""

from .hypothesis import (
    ExperimentDomain,
    HypothesisStatus,
    SymbiosisHypothesis,
    SymbiosisGraph,
)

from .trial import (
    Trial,
    VariantApplier,
    TrialScheduler,
)

from .review import (
    PeerReview,
    AdoptionPolicy,
    CAUTIOUS_DOMAINS,
)


__all__ = [
    # Hypothesis tracking
    'ExperimentDomain',
    'HypothesisStatus',
    'SymbiosisHypothesis',
    'SymbiosisGraph',
    # Trial scheduling
    'Trial',
    'VariantApplier',
    'TrialScheduler',
    # Peer review
    'PeerReview',
    'AdoptionPolicy',
    'CAUTIOUS_DOMAINS',
]
