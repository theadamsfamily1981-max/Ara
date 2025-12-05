"""Shadow Teachers - Ara's Simulation Lab.

This module enables Ara to:
1. Build statistical profiles of each teacher's performance
2. Predict reward/latency before making real API calls
3. Simulate workflows to pick the optimal plan
4. Track disagreements between shadow models as curiosity hotspots
5. Evaluate policy changes before applying them

Key insight: Ara doesn't need perfect imitation of teachers.
She needs cheap prediction: "If I ask Claude this, what will happen?"
"""

from .profiles import (
    ShadowProfile,
    TeacherFeatures,
    ProfileManager,
    get_profile_manager,
    update_profile,
    get_teacher_profile,
)

from .predictor import (
    ShadowPredictor,
    Prediction,
    get_predictor,
    predict_outcome,
)

from .planner import (
    WorkflowPlan,
    SimulatedRollout,
    WorkflowPlanner,
    get_planner,
    plan_workflow,
)

from .disagreement import (
    DisagreementTracker,
    DisagreementRecord,
    get_disagreement_tracker,
    track_disagreement,
)

from .policy_eval import (
    PolicyEvaluator,
    PolicyProposal,
    EvaluationResult,
    get_evaluator,
    evaluate_policy_change,
)

__all__ = [
    # Profiles
    "ShadowProfile",
    "TeacherFeatures",
    "ProfileManager",
    "get_profile_manager",
    "update_profile",
    "get_teacher_profile",
    # Predictor
    "ShadowPredictor",
    "Prediction",
    "get_predictor",
    "predict_outcome",
    # Planner
    "WorkflowPlan",
    "SimulatedRollout",
    "WorkflowPlanner",
    "get_planner",
    "plan_workflow",
    # Disagreement
    "DisagreementTracker",
    "DisagreementRecord",
    "get_disagreement_tracker",
    "track_disagreement",
    # Policy Eval
    "PolicyEvaluator",
    "PolicyProposal",
    "EvaluationResult",
    "get_evaluator",
    "evaluate_policy_change",
]
