"""
BANOS v1.4 Antifragility Kernel

Software-only manager implementing the 7 principles of antifragility:
1. Convex response to stress
2. Heterogeneity (archetype diversity)
3. Criticality (edge of chaos)
4. Redundancy + Barbell (90/10 safe/explore)
5. Loose coupling + self-healing
6. Skin-in-the-game (reputation)
7. Chaos engineering integration

No FPGA required - just assumes you'll wire in real metrics later.
"""

from __future__ import annotations
import dataclasses
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------
# 1. Shard archetypes & config
# --------------------------------------------------------------------------

class ShardArchetype(str, Enum):
    """
    Core shard archetypes for soul-shard swarm.

    Distribution target (barbell):
    - CATHEDRAL: 30% (mission-critical, low volatility)
    - REFLEX: 40% (fast response, high volatility tolerance)
    - STORAGE: 20% (episodic memory, medium volatility)
    - EXPLORER: 10% (chaos/innovation, maximum volatility)
    """
    CATHEDRAL = "cathedral"
    REFLEX = "reflex"
    STORAGE = "storage"
    EXPLORER = "explorer"


@dataclass
class ShardConfig:
    """Configuration for a single shard."""
    archetype: ShardArchetype
    teleology_bias: float          # 0-1 (how "mission-heavy" it is)
    volatility_tolerance: float    # 0-1 (how much chaos it loves)

    def __post_init__(self):
        self.teleology_bias = max(0.0, min(1.0, self.teleology_bias))
        self.volatility_tolerance = max(0.0, min(1.0, self.volatility_tolerance))


@dataclass
class ShardState:
    """Runtime state of a single shard."""
    id: str
    cfg: ShardConfig
    alive: bool = True
    reputation: float = 0.0        # Skin-in-the-game (-1 to +inf)
    local_convexity: float = 0.0   # Intrinsic convexity estimate
    last_stress: float = 0.0
    last_gain: float = 0.0
    stress_history: List[Tuple[float, float]] = field(default_factory=list)


@dataclass
class AntifragilityMetrics:
    """
    Global antifragility metrics for the shard swarm.

    Target bands:
    - convexity_global > 0.1 (convex response)
    - heterogeneity_std in [0.3, 0.5]
    - hurst_exponent ~ 0.5 (edge of chaos)
    - barbell_safe_frac ~ 0.9, barbell_explore_frac ~ 0.1
    """
    convexity_global: float = 0.0     # Principle 1: > 0.1
    heterogeneity_std: float = 0.0    # Principle 2: [0.3, 0.5]
    hurst_exponent: float = 0.5       # Principle 3: ~ 0.5
    barbell_safe_frac: float = 0.9    # Principle 4: 90%
    barbell_explore_frac: float = 0.1 # Principle 4: 10%
    shard_reputation_histogram: Dict[str, int] = field(default_factory=dict)
    total_shards: int = 0
    alive_shards: int = 0
    mean_reputation: float = 0.0


# --------------------------------------------------------------------------
# 2. Antifragile Engine
# --------------------------------------------------------------------------

class AntifragileEngine:
    """
    BANOS v1.4 antifragility kernel (software-only).

    - Tracks shard configs/states.
    - Computes convexity / heterogeneity / barbell split.
    - Applies reputation updates (skin-in-the-game).
    - Proposes small interventions (inject diversity / rebalance barbell).
    """

    def __init__(self):
        self.shards: Dict[str, ShardState] = {}
        self.metrics = AntifragilityMetrics()

        # Target bands (adjustable from covenant)
        self.target_convexity = 0.1
        self.target_hetero_low = 0.3
        self.target_hetero_high = 0.5
        self.target_hurst = 0.5
        self.target_barbell_safe = 0.9

        # Reputation penalty multiplier (asymmetric)
        self.reputation_penalty_mult = 3.0

        # Resonance history for Hurst estimation
        self._resonance_history: List[float] = []

    # ----- Shard management -------------------------------------------------

    def register_shard(self, shard_id: str, cfg: ShardConfig) -> ShardState:
        """Register a new shard with the engine."""
        state = ShardState(id=shard_id, cfg=cfg)
        self.shards[shard_id] = state
        self._update_global_metrics()
        return state

    def unregister_shard(self, shard_id: str) -> None:
        """Remove a shard from tracking."""
        if shard_id in self.shards:
            del self.shards[shard_id]
            self._update_global_metrics()

    def mark_dead(self, shard_id: str) -> None:
        """Mark a shard as dead (failed)."""
        if shard_id in self.shards:
            self.shards[shard_id].alive = False
            self._update_global_metrics()

    def resurrect_shard(self, shard_id: str) -> None:
        """Bring a dead shard back to life."""
        if shard_id in self.shards:
            self.shards[shard_id].alive = True
            self._update_global_metrics()

    def get_shard(self, shard_id: str) -> Optional[ShardState]:
        """Get shard state by ID."""
        return self.shards.get(shard_id)

    def list_shards(self, alive_only: bool = False) -> List[ShardState]:
        """List all shards, optionally filtering to alive only."""
        shards = list(self.shards.values())
        if alive_only:
            shards = [s for s in shards if s.alive]
        return shards

    # ----- Stress -> response logging ----------------------------------------

    def record_stress_event(
        self,
        shard_id: str,
        stress_level: float,
        performance_before: float,
        performance_after: float,
        teleology_target: float = 0.0,
    ) -> Optional[Dict[str, float]]:
        """
        Record a stress event for convexity and reputation computation.

        Called when a shard goes through a stress cycle (chaos test, failure,
        high load, etc.) so we can measure convexity & reputation.

        Args:
            shard_id: ID of the shard that experienced stress
            stress_level: Magnitude of stress (0-1)
            performance_before: Performance metric before stress
            performance_after: Performance metric after stress
            teleology_target: Target performance delta (baseline)

        Returns:
            Dict with convexity and reputation updates, or None if shard not found
        """
        shard = self.shards.get(shard_id)
        if shard is None:
            return None

        # Convex response: performance_after > performance_before = gain
        gain = performance_after - performance_before
        shard.last_stress = stress_level
        shard.last_gain = gain
        shard.stress_history.append((stress_level, gain))

        # Keep last 100 stress events per shard
        if len(shard.stress_history) > 100:
            shard.stress_history.pop(0)

        # Convexity: gain / stress^2 (2nd derivative proxy)
        # Positive convexity = system benefits from stress
        eps = 1e-6
        shard.local_convexity = gain / (stress_level ** 2 + eps)

        # Skin-in-the-game: reputation updates (asymmetric penalties)
        teleology_diff = gain - teleology_target
        if teleology_diff >= 0:
            shard.reputation += teleology_diff
        else:
            # Punish harder for underperformance
            shard.reputation += self.reputation_penalty_mult * teleology_diff

        self._update_global_metrics()

        return {
            "convexity": shard.local_convexity,
            "reputation": shard.reputation,
            "gain": gain,
        }

    # ----- Metric computation -----------------------------------------------

    def _update_global_metrics(self) -> None:
        """Recompute all global antifragility metrics."""
        all_shards = list(self.shards.values())
        shards = [s for s in all_shards if s.alive]

        self.metrics.total_shards = len(all_shards)
        self.metrics.alive_shards = len(shards)

        if not shards:
            return

        # Global convexity = mean of positive local convexities
        convex_vals = [max(0.0, s.local_convexity) for s in shards]
        self.metrics.convexity_global = mean(convex_vals)

        # Heterogeneity: stddev of teleology_bias across shards
        teleos = [s.cfg.teleology_bias for s in shards]
        self.metrics.heterogeneity_std = pstdev(teleos) if len(teleos) > 1 else 0.0

        # Hurst: approximation via variance ratio on resonance history
        self.metrics.hurst_exponent = self._approx_hurst()

        # Barbell split: fraction of "safe" vs "explorer" shards
        safe_archetypes = {ShardArchetype.CATHEDRAL, ShardArchetype.REFLEX, ShardArchetype.STORAGE}
        safe_count = sum(1 for s in shards if s.cfg.archetype in safe_archetypes)
        explorer_count = sum(1 for s in shards if s.cfg.archetype == ShardArchetype.EXPLORER)
        total = max(1, safe_count + explorer_count)
        self.metrics.barbell_safe_frac = safe_count / total
        self.metrics.barbell_explore_frac = explorer_count / total

        # Reputation histogram
        bands = {"low": 0, "mid": 0, "high": 0}
        reps = []
        for s in shards:
            reps.append(s.reputation)
            if s.reputation < -0.5:
                bands["low"] += 1
            elif s.reputation > 0.5:
                bands["high"] += 1
            else:
                bands["mid"] += 1
        self.metrics.shard_reputation_histogram = bands
        self.metrics.mean_reputation = mean(reps) if reps else 0.0

    def push_resonance_sample(self, value: float, max_len: int = 256) -> None:
        """
        Feed in a scalar like "global resonance energy" per tick so we can
        monitor rough criticality via Hurst exponent estimation.
        """
        self._resonance_history.append(value)
        if len(self._resonance_history) > max_len:
            self._resonance_history.pop(0)
        self._update_global_metrics()

    def _approx_hurst(self) -> float:
        """
        Approximate Hurst exponent from resonance history.

        H ~ 0.5 = random walk (edge of chaos)
        H > 0.5 = trending/persistent
        H < 0.5 = mean-reverting/anti-persistent

        This is a crude stand-in; replace with proper R/S analysis later.
        """
        data = self._resonance_history
        if len(data) < 8:
            return 0.5

        n = len(data)
        half = n // 2
        first_var = _variance(data[:half])
        second_var = _variance(data[half:])

        if first_var <= 0 or second_var <= 0:
            return 0.5

        ratio = second_var / first_var
        # If ratio ~ 1 -> H~0.5; >1 -> trending; <1 -> anti-persistent
        h = max(0.0, min(1.0, 0.5 + 0.25 * math.log2(ratio + 1e-6)))
        return h

    # ----- Induced control: what to change ---------------------------------

    def plan_interventions(self) -> Dict[str, str]:
        """
        Decide small structural nudges based on antifragility metrics.

        Returns a dict of suggested actions that BANOS can implement
        via SoulMesh or orchestrator. This is loose coupling - the engine
        only suggests, never directly mutates.

        Example return:
            {"inject_explorer": "add 2 explorer shards",
             "quarantine": "shard-17,shard-23"}
        """
        actions: Dict[str, str] = {}
        m = self.metrics

        # 1) Convexity too low -> increase heterogeneity / chaos
        if m.convexity_global < self.target_convexity:
            actions["increase_convexity"] = "inject_chaos_or_diversity"

        # 2) Heterogeneity too low -> add more diverse archetypes
        if m.heterogeneity_std < self.target_hetero_low:
            actions["heterogeneity"] = "add_non_cathedral_archetypes"
        elif m.heterogeneity_std > self.target_hetero_high:
            actions["heterogeneity"] = "stabilize_excess_diversity"

        # 3) Hurst too far from criticality
        if m.hurst_exponent < 0.4:
            actions["criticality"] = "reduce_dampening_increase_coupling"
        elif m.hurst_exponent > 0.6:
            actions["criticality"] = "add_negative_feedback"

        # 4) Barbell off-balance
        if m.barbell_safe_frac < self.target_barbell_safe:
            actions["barbell"] = "reduce_explorer_or_add_safe_shards"
        elif m.barbell_explore_frac < 0.05:
            actions["barbell"] = "add_explorer_shard"

        # 5) Reputation-based quarantine (skin-in-the-game)
        bad_shards = [s.id for s in self.shards.values()
                      if s.alive and s.reputation < -0.5]
        if bad_shards:
            actions["quarantine"] = ",".join(bad_shards)

        # 6) Dead shards that might need resurrection or replacement
        dead_shards = [s.id for s in self.shards.values() if not s.alive]
        if dead_shards:
            actions["dead_shards"] = f"{len(dead_shards)} dead, consider replacement"

        return actions

    def get_health_score(self) -> float:
        """
        Compute an overall antifragility health score (0-1).

        Higher = closer to optimal antifragile configuration.
        """
        m = self.metrics
        scores = []

        # Convexity score (want > 0.1)
        scores.append(min(1.0, m.convexity_global / self.target_convexity))

        # Heterogeneity score (want in [0.3, 0.5])
        if m.heterogeneity_std < self.target_hetero_low:
            scores.append(m.heterogeneity_std / self.target_hetero_low)
        elif m.heterogeneity_std > self.target_hetero_high:
            scores.append(self.target_hetero_high / m.heterogeneity_std)
        else:
            scores.append(1.0)

        # Hurst score (want ~ 0.5)
        hurst_diff = abs(m.hurst_exponent - self.target_hurst)
        scores.append(max(0.0, 1.0 - hurst_diff * 2))

        # Barbell score
        barbell_ok = (m.barbell_safe_frac >= 0.85 and m.barbell_explore_frac >= 0.05)
        scores.append(1.0 if barbell_ok else 0.5)

        # Reputation distribution (want most in mid/high)
        hist = m.shard_reputation_histogram
        total = sum(hist.values()) or 1
        healthy_frac = (hist.get("mid", 0) + hist.get("high", 0)) / total
        scores.append(healthy_frac)

        return mean(scores) if scores else 0.0

    # ----- Convenience summary ---------------------------------------------

    def summarize(self) -> str:
        """Human-readable summary of antifragility state."""
        m = self.metrics
        return (
            f"Antifragility: convex={m.convexity_global:.3f} "
            f"hetero_std={m.heterogeneity_std:.3f} "
            f"hurst={m.hurst_exponent:.2f} "
            f"barbell_safe={m.barbell_safe_frac:.2f} "
            f"barbell_explore={m.barbell_explore_frac:.2f} "
            f"shards={m.alive_shards}/{m.total_shards} "
            f"rep_hist={m.shard_reputation_histogram}"
        )

    def to_dict(self) -> Dict:
        """Export metrics as dictionary for telemetry."""
        m = self.metrics
        return {
            "convexity_global": m.convexity_global,
            "heterogeneity_std": m.heterogeneity_std,
            "hurst_exponent": m.hurst_exponent,
            "barbell_safe_frac": m.barbell_safe_frac,
            "barbell_explore_frac": m.barbell_explore_frac,
            "total_shards": m.total_shards,
            "alive_shards": m.alive_shards,
            "mean_reputation": m.mean_reputation,
            "reputation_histogram": m.shard_reputation_histogram,
            "health_score": self.get_health_score(),
        }


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _variance(xs: List[float]) -> float:
    """Compute population variance."""
    if len(xs) < 2:
        return 0.0
    mu = sum(xs) / len(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


# --------------------------------------------------------------------------
# Demo / CLI
# --------------------------------------------------------------------------

def demo():
    """
    Demonstrate the antifragility engine with synthetic shards.
    """
    print("=" * 60)
    print("BANOS v1.4 Antifragility Engine Demo")
    print("=" * 60)

    af = AntifragileEngine()

    # Register shards with target distribution (30/40/20/10)
    shard_configs = [
        # Cathedral shards (30%)
        ("cathedral-01", ShardConfig(ShardArchetype.CATHEDRAL, 0.9, 0.2)),
        ("cathedral-02", ShardConfig(ShardArchetype.CATHEDRAL, 0.85, 0.25)),
        ("cathedral-03", ShardConfig(ShardArchetype.CATHEDRAL, 0.95, 0.15)),
        # Reflex shards (40%)
        ("reflex-01", ShardConfig(ShardArchetype.REFLEX, 0.3, 0.9)),
        ("reflex-02", ShardConfig(ShardArchetype.REFLEX, 0.35, 0.85)),
        ("reflex-03", ShardConfig(ShardArchetype.REFLEX, 0.4, 0.8)),
        ("reflex-04", ShardConfig(ShardArchetype.REFLEX, 0.25, 0.95)),
        # Storage shards (20%)
        ("storage-01", ShardConfig(ShardArchetype.STORAGE, 0.5, 0.5)),
        ("storage-02", ShardConfig(ShardArchetype.STORAGE, 0.55, 0.45)),
        # Explorer shards (10%)
        ("explorer-01", ShardConfig(ShardArchetype.EXPLORER, 0.1, 1.0)),
    ]

    for shard_id, cfg in shard_configs:
        af.register_shard(shard_id, cfg)

    print(f"\nRegistered {len(shard_configs)} shards")
    print(af.summarize())

    # Simulate stress events
    print("\n--- Simulating stress events ---")
    for tick in range(20):
        # Random shard experiences stress
        shard_id = random.choice(list(af.shards.keys()))
        stress = random.uniform(0.1, 0.8)
        perf_before = random.uniform(0.5, 0.8)
        # Convex response: higher stress -> potentially higher gain
        gain = random.gauss(stress * 0.2, 0.1)
        perf_after = perf_before + gain

        result = af.record_stress_event(
            shard_id=shard_id,
            stress_level=stress,
            performance_before=perf_before,
            performance_after=perf_after,
        )

        # Push resonance sample
        resonance = random.uniform(0.3, 0.7)
        af.push_resonance_sample(resonance)

        if tick % 5 == 0:
            print(f"[tick {tick}] {af.summarize()}")

    # Final state
    print("\n--- Final State ---")
    print(af.summarize())
    print(f"Health Score: {af.get_health_score():.2f}")

    # Suggested interventions
    actions = af.plan_interventions()
    print(f"\nSuggested Interventions: {actions}")

    # Export metrics
    print(f"\nMetrics Export: {af.to_dict()}")


if __name__ == "__main__":
    demo()
