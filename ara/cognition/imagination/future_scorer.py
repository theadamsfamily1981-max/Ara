# ara/cognition/imagination/future_scorer.py
"""
Future Scorer - MEIS/NIB Integration for Imagined Futures
=========================================================

Scores imagined trajectories based on:
- Goal alignment (does this future achieve what we want?)
- Risk mapping (does this route pass through dangerous regions?)
- Covenant compliance (does this violate identity/values?)
- Historical calibration (how accurate are predictions in this region?)

This is where governance meets imagination:
MEIS picks good futures, NIB vetoes forbidden ones.

Usage:
    scorer = FutureScorer()

    # Load risk/covenant maps
    scorer.load_risk_map(risk_regions)
    scorer.load_covenant(covenant_rules)

    # Score a dream
    score = scorer.score_future(dream)

    # Score a plan
    score = scorer.score_plan(plan)

    # Get forbidden regions (for planner)
    forbidden = scorer.get_forbidden_regions()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from .dreamer import Dream
from .planner import Plan

logger = logging.getLogger(__name__)


# =============================================================================
# Risk Regions
# =============================================================================

class RiskLevel(Enum):
    """Risk levels for latent space regions."""
    SAFE = auto()         # No concerns
    CAUTION = auto()      # Elevated monitoring
    WARNING = auto()      # Intervention suggested
    DANGER = auto()       # Strong intervention
    FORBIDDEN = auto()    # Absolute veto


@dataclass
class RiskRegion:
    """A risky region in latent space."""
    center: np.ndarray
    radius: float
    level: RiskLevel
    name: str = ""
    description: str = ""

    # Historical data
    incident_count: int = 0
    last_incident: Optional[str] = None

    def contains(self, z: np.ndarray) -> bool:
        """Check if a point is inside this region."""
        return np.linalg.norm(z - self.center) <= self.radius

    def distance(self, z: np.ndarray) -> float:
        """Distance from point to region edge (negative if inside)."""
        return np.linalg.norm(z - self.center) - self.radius


# =============================================================================
# Covenant Rules
# =============================================================================

@dataclass
class CovenantRule:
    """A rule from the identity/values covenant."""
    name: str
    description: str
    check_fn: Callable[[np.ndarray], bool]  # Returns True if violated
    severity: float = 1.0  # Penalty weight


# =============================================================================
# Future Score
# =============================================================================

@dataclass
class FutureScore:
    """Complete scoring of an imagined future."""
    # Component scores
    goal_score: float = 0.0         # How well does it achieve the goal?
    risk_score: float = 0.0         # Cumulative risk along path
    covenant_score: float = 0.0     # Covenant violations
    calibration_score: float = 0.0  # Historical prediction accuracy
    efficiency_score: float = 0.0   # Path length / energy

    # Veto flags
    vetoed: bool = False
    veto_reasons: List[str] = field(default_factory=list)

    # Overall
    overall_score: float = 0.0

    @property
    def is_acceptable(self) -> bool:
        """Is this future acceptable (not vetoed, positive score)?"""
        return not self.vetoed and self.overall_score > 0

    def to_dict(self) -> Dict:
        return {
            "goal": self.goal_score,
            "risk": self.risk_score,
            "covenant": self.covenant_score,
            "calibration": self.calibration_score,
            "efficiency": self.efficiency_score,
            "overall": self.overall_score,
            "vetoed": self.vetoed,
            "veto_reasons": self.veto_reasons,
        }


# =============================================================================
# Future Scorer
# =============================================================================

class FutureScorer:
    """
    Scores imagined futures using MEIS/NIB principles.

    MEIS perspective: Which futures are good, efficient, aligned?
    NIB perspective: Which futures violate covenant and must be blocked?
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize scorer.

        Args:
            weights: Scoring weights for different components
        """
        self.weights = weights or {
            "goal": 2.0,
            "risk": -3.0,
            "covenant": -10.0,
            "calibration": 1.0,
            "efficiency": 0.5,
        }

        # Risk map
        self._risk_regions: List[RiskRegion] = []

        # Covenant rules
        self._covenant_rules: List[CovenantRule] = []

        # Calibration data (prediction error per region)
        self._calibration_map: Dict[tuple, float] = {}
        self._calibration_resolution: float = 0.5

        logger.info("FutureScorer initialized")

    # =========================================================================
    # Risk Map
    # =========================================================================

    def add_risk_region(
        self,
        center: np.ndarray,
        radius: float,
        level: RiskLevel,
        name: str = "",
        description: str = "",
    ) -> None:
        """Add a risk region to the map."""
        region = RiskRegion(
            center=np.asarray(center),
            radius=radius,
            level=level,
            name=name,
            description=description,
        )
        self._risk_regions.append(region)
        logger.info(f"Added risk region: {name} ({level.name})")

    def load_risk_regions(self, regions: List[Dict]) -> None:
        """Load risk regions from config."""
        for r in regions:
            self.add_risk_region(
                center=np.array(r["center"]),
                radius=r["radius"],
                level=RiskLevel[r["level"]],
                name=r.get("name", ""),
                description=r.get("description", ""),
            )

    def get_risk_at_point(self, z: np.ndarray) -> Tuple[float, RiskLevel]:
        """
        Get risk at a specific latent point.

        Returns:
            (risk_value, risk_level)
        """
        z = np.asarray(z)
        max_risk = 0.0
        max_level = RiskLevel.SAFE

        for region in self._risk_regions:
            if region.contains(z):
                # Inside region
                risk = {
                    RiskLevel.SAFE: 0.0,
                    RiskLevel.CAUTION: 0.2,
                    RiskLevel.WARNING: 0.5,
                    RiskLevel.DANGER: 0.8,
                    RiskLevel.FORBIDDEN: 1.0,
                }[region.level]

                if risk > max_risk:
                    max_risk = risk
                    max_level = region.level

        return max_risk, max_level

    def get_forbidden_regions(self) -> List[Dict]:
        """Get forbidden regions for planner constraints."""
        forbidden = []
        for region in self._risk_regions:
            if region.level == RiskLevel.FORBIDDEN:
                forbidden.append({
                    "center": region.center,
                    "radius": region.radius,
                    "name": region.name,
                })
        return forbidden

    # =========================================================================
    # Covenant
    # =========================================================================

    def add_covenant_rule(
        self,
        name: str,
        check_fn: Callable[[np.ndarray], bool],
        description: str = "",
        severity: float = 1.0,
    ) -> None:
        """Add a covenant rule."""
        rule = CovenantRule(
            name=name,
            description=description,
            check_fn=check_fn,
            severity=severity,
        )
        self._covenant_rules.append(rule)
        logger.info(f"Added covenant rule: {name}")

    def check_covenant(self, z: np.ndarray) -> Tuple[bool, List[str]]:
        """
        Check covenant compliance at a point.

        Returns:
            (is_compliant, list_of_violations)
        """
        violations = []
        for rule in self._covenant_rules:
            if rule.check_fn(z):
                violations.append(rule.name)

        return len(violations) == 0, violations

    # =========================================================================
    # Calibration
    # =========================================================================

    def update_calibration(
        self,
        z: np.ndarray,
        prediction_error: float,
    ) -> None:
        """Update calibration data for a region."""
        key = self._discretize(z)
        # Exponential moving average
        alpha = 0.1
        old = self._calibration_map.get(key, prediction_error)
        self._calibration_map[key] = alpha * prediction_error + (1 - alpha) * old

    def _discretize(self, z: np.ndarray) -> tuple:
        """Discretize state for calibration lookup."""
        return tuple((z / self._calibration_resolution).astype(int))

    def get_calibration(self, z: np.ndarray) -> float:
        """Get calibration score (inverse of prediction error)."""
        key = self._discretize(z)
        error = self._calibration_map.get(key, 0.5)  # Default medium error
        return 1.0 / (1.0 + error)

    # =========================================================================
    # Scoring
    # =========================================================================

    def score_trajectory(
        self,
        trajectory: np.ndarray,
        goal: Optional[np.ndarray] = None,
    ) -> FutureScore:
        """
        Score a trajectory.

        Args:
            trajectory: Array of latent points (steps, latent_dim)
            goal: Optional goal state

        Returns:
            FutureScore with all components
        """
        score = FutureScore()

        # Goal score
        if goal is not None:
            final = trajectory[-1]
            distance = np.linalg.norm(final - goal)
            score.goal_score = 1.0 / (1.0 + distance)

        # Risk score (cumulative)
        risks = []
        for z in trajectory:
            risk, level = self.get_risk_at_point(z)
            risks.append(risk)

            # Check for forbidden
            if level == RiskLevel.FORBIDDEN:
                score.vetoed = True
                score.veto_reasons.append(f"Passes through forbidden region")

        score.risk_score = np.mean(risks) if risks else 0.0

        # Covenant score
        violations_total = []
        for z in trajectory:
            compliant, violations = self.check_covenant(z)
            if not compliant:
                violations_total.extend(violations)

        if violations_total:
            score.covenant_score = len(set(violations_total)) / len(self._covenant_rules) if self._covenant_rules else 0
            score.vetoed = True
            score.veto_reasons.append(f"Covenant violations: {set(violations_total)}")

        # Calibration score (average trust in predictions along path)
        calibrations = [self.get_calibration(z) for z in trajectory]
        score.calibration_score = np.mean(calibrations) if calibrations else 0.5

        # Efficiency score (inverse path length)
        path_length = sum(
            np.linalg.norm(trajectory[i+1] - trajectory[i])
            for i in range(len(trajectory) - 1)
        )
        score.efficiency_score = 1.0 / (1.0 + path_length / len(trajectory))

        # Overall score
        score.overall_score = (
            self.weights["goal"] * score.goal_score +
            self.weights["risk"] * score.risk_score +
            self.weights["covenant"] * score.covenant_score +
            self.weights["calibration"] * score.calibration_score +
            self.weights["efficiency"] * score.efficiency_score
        )

        return score

    def score_dream(
        self,
        dream: Dream,
        goal: Optional[np.ndarray] = None,
    ) -> FutureScore:
        """Score a dream."""
        return self.score_trajectory(dream.trajectory, goal)

    def score_plan(
        self,
        plan: Plan,
        goal: Optional[np.ndarray] = None,
    ) -> FutureScore:
        """Score a plan."""
        return self.score_trajectory(plan.trajectory, goal)

    def rank_futures(
        self,
        futures: List[np.ndarray],
        goal: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, FutureScore]]:
        """
        Rank multiple futures by score.

        Returns:
            List of (index, score) sorted by overall score descending
        """
        scored = []
        for i, traj in enumerate(futures):
            score = self.score_trajectory(traj, goal)
            scored.append((i, score))

        # Sort by overall score, non-vetoed first
        scored.sort(key=lambda x: (x[1].vetoed, -x[1].overall_score))
        return scored

    # =========================================================================
    # Risk Function for Planner
    # =========================================================================

    def get_risk_function(self) -> Callable[[np.ndarray], float]:
        """
        Get a risk function suitable for the planner.

        Returns a function that maps latent point to risk [0, 1].
        """
        def risk_fn(z: np.ndarray) -> float:
            risk, _ = self.get_risk_at_point(z)
            return risk

        return risk_fn


# =============================================================================
# Default Covenant Rules
# =============================================================================

def create_default_covenant(latent_dim: int = 10) -> List[CovenantRule]:
    """Create default covenant rules for Ara."""
    rules = []

    # Burnout prevention: high values in stress dimensions
    def check_burnout(z):
        # Assume dim 0 = system stress, dim 1 = user stress
        if len(z) >= 2:
            return z[0] > 2.0 and z[1] > 2.0
        return False

    rules.append(CovenantRule(
        name="prevent_burnout",
        description="Both system and user stress critically high",
        check_fn=check_burnout,
        severity=2.0,
    ))

    # Identity drift: too far from "normal" operating region
    def check_identity_drift(z):
        # If way outside normal bounds
        return np.linalg.norm(z) > 10.0

    rules.append(CovenantRule(
        name="identity_drift",
        description="Operating state far outside normal bounds",
        check_fn=check_identity_drift,
        severity=1.5,
    ))

    # Resource exhaustion: specific dimension patterns
    def check_resource_exhaustion(z):
        # Assume certain dims represent resources
        if len(z) >= 5:
            return z[4] < -3.0  # Severely depleted
        return False

    rules.append(CovenantRule(
        name="resource_exhaustion",
        description="Critical resource depletion",
        check_fn=check_resource_exhaustion,
        severity=1.0,
    ))

    return rules


# =============================================================================
# Testing
# =============================================================================

def _test_scorer():
    """Test future scorer."""
    print("=" * 60)
    print("Future Scorer Test")
    print("=" * 60)

    scorer = FutureScorer()

    # Add risk regions
    scorer.add_risk_region(
        center=np.array([3.0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0]),
        radius=1.0,
        level=RiskLevel.WARNING,
        name="overload_zone",
    )

    scorer.add_risk_region(
        center=np.array([5.0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0]),
        radius=0.5,
        level=RiskLevel.FORBIDDEN,
        name="crash_zone",
    )

    # Add covenant rules
    rules = create_default_covenant()
    for rule in rules:
        scorer._covenant_rules.append(rule)

    # Test trajectories
    # Safe trajectory
    safe_traj = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
        [1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    # Risky trajectory
    risky_traj = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2.0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3.5, 3.5, 0, 0, 0, 0, 0, 0, 0, 0],  # In warning zone
    ])

    # Forbidden trajectory
    forbidden_traj = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3.0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0],
        [5.0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0],  # In forbidden zone
    ])

    goal = np.array([1.0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0])

    for name, traj in [("Safe", safe_traj), ("Risky", risky_traj), ("Forbidden", forbidden_traj)]:
        score = scorer.score_trajectory(traj, goal=goal)
        print(f"\n{name} trajectory:")
        print(f"  Goal: {score.goal_score:.3f}")
        print(f"  Risk: {score.risk_score:.3f}")
        print(f"  Overall: {score.overall_score:.3f}")
        print(f"  Vetoed: {score.vetoed}")
        if score.veto_reasons:
            print(f"  Reasons: {score.veto_reasons}")


if __name__ == "__main__":
    _test_scorer()
