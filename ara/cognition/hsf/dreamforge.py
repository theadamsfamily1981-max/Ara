"""
HSF Dreamforge - Counterfactual Field Simulation
==================================================

The Dreamforge transforms Ara from reactive to generative:
instead of just responding to the world, she imagines alternate
universes and helps you choose which one to walk into.

Architecture:
    TopologySketcher: Generates plausible config scenarios
            ↓
    FieldSimulator: Replays real workloads under each scenario
            ↓
    DreamOutcome: Scored results (stability, cost, "you-ness")
            ↓
    ScenarioMarket: Presents futures to Sovereign/Treasury

This is the "city planner" layer: designing the next version of
your empire before you spend a dollar or lift a screwdriver.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum, auto

from .lanes import ItemMemory
from .zones import Zone
from .counterfactual import (
    ChangeType, ConfigDelta, ConfigScenario,
    ConfigEncoder, LoadTrace, FieldDynamics, GhostReplay, ReplayResult
)


class ScenarioArchetype(Enum):
    """Common scenario archetypes for generation."""
    MINIMAL_HARDWARE = auto()     # Maximize software, minimize hardware
    BALANCED = auto()             # Balance hardware and software changes
    HARDWARE_HEAVY = auto()        # Add hardware to solve problems
    POLICY_TUNE = auto()          # Only tune existing policies
    TOPOLOGY_REFACTOR = auto()     # Reorganize existing resources
    GROWTH = auto()               # Scale up capacity
    CONSOLIDATION = auto()        # Reduce complexity


@dataclass
class DreamOutcome:
    """
    The scored result of dreaming a scenario.

    Combines multiple dimensions of "goodness" into a
    coherent picture of this possible future.
    """
    scenario: ConfigScenario
    replay_results: List[ReplayResult]

    # Stability metrics
    stability_gain: float = 0.0       # vs baseline, positive = better
    antifragility_score: float = 0.0  # How well does it handle stress?
    immune_load_delta: float = 0.0    # Change in antibody firing rate

    # Performance metrics
    throughput_delta: float = 0.0     # Estimated throughput change
    latency_delta: float = 0.0        # Estimated latency change

    # Cost metrics
    total_cost: float = 0.0
    roi_score: float = 0.0            # Return on investment

    # Alignment metrics
    complexity_delta: float = 0.0     # Change in system complexity
    you_ness_score: float = 0.0       # Does this feel like your style?

    # Summary
    confidence: float = 0.0           # How confident are we in this dream?
    narrative_pitch: str = ""         # Human-readable summary

    @property
    def composite_score(self) -> float:
        """
        Composite score combining all dimensions.

        Weights can be tuned based on current priorities.
        """
        return (
            self.stability_gain * 0.30 +
            self.antifragility_score * 0.15 +
            self.roi_score * 0.25 +
            self.you_ness_score * 0.10 +
            (1.0 - self.complexity_delta) * 0.10 +
            self.throughput_delta * 0.10
        )


@dataclass
class TopologySketcher:
    """
    Generates plausible configuration scenarios.

    Takes human intuitions like "we need more GPU capacity" and
    turns them into formal ConfigScenarios that respect:
    - Fleet: what machines exist / could exist cheaply
    - Treasury: what capital is available
    - Teleology: does this push toward the Horizon?
    - Style: does this feel like a "Croft move"?
    """
    available_nodes: List[str] = field(default_factory=list)
    node_capabilities: Dict[str, List[str]] = field(default_factory=dict)
    budget_limit: float = 500.0  # Max hardware spend
    human_hours_limit: float = 20.0  # Max human time

    # Junkyard inventory (cheap options)
    junkyard: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        # Default junkyard items for demos
        if not self.junkyard:
            self.junkyard = [
                {"name": "salvage-gpu-box", "cost": 80, "caps": ["gpu", "compute"],
                 "description": "Salvaged workstation with decent GPU"},
                {"name": "spare-pi-4", "cost": 0, "caps": ["light-compute", "monitoring"],
                 "description": "Raspberry Pi 4 already in drawer"},
                {"name": "old-nas", "cost": 50, "caps": ["storage", "backup"],
                 "description": "Old NAS with 4TB usable"},
                {"name": "mining-fpga", "cost": 120, "caps": ["fpga", "neural"],
                 "description": "Repurposed crypto miner FPGA board"},
            ]

    def sketch_from_goal(self, goal: str, archetype: ScenarioArchetype,
                         n_scenarios: int = 3) -> List[ConfigScenario]:
        """
        Generate scenarios from a high-level goal.

        Examples:
            "We need more GPU capacity"
            "Print farm is too fragile"
            "Network monitoring is blind"
        """
        scenarios = []

        if archetype == ScenarioArchetype.MINIMAL_HARDWARE:
            scenarios.extend(self._sketch_minimal_hardware(goal))
        elif archetype == ScenarioArchetype.HARDWARE_HEAVY:
            scenarios.extend(self._sketch_hardware_heavy(goal))
        elif archetype == ScenarioArchetype.BALANCED:
            scenarios.extend(self._sketch_balanced(goal))
        elif archetype == ScenarioArchetype.POLICY_TUNE:
            scenarios.extend(self._sketch_policy_only(goal))
        elif archetype == ScenarioArchetype.TOPOLOGY_REFACTOR:
            scenarios.extend(self._sketch_refactor(goal))

        return scenarios[:n_scenarios]

    def _sketch_minimal_hardware(self, goal: str) -> List[ConfigScenario]:
        """Generate scenarios that minimize hardware changes."""
        scenarios = []

        # Scenario: Just tune policies
        scenarios.append(ConfigScenario(
            id=f"min_hw_{int(time.time())}",
            name="Policy Tuning Only",
            description=f"Address '{goal}' through policy adjustments only",
            deltas=[
                ConfigDelta(
                    change_type=ChangeType.MODIFY_POLICY,
                    target="reflex_thresholds",
                    params={"sensitivity": "higher"},
                    description="Tune reflex thresholds for earlier intervention",
                ),
                ConfigDelta(
                    change_type=ChangeType.ADD_POLICY,
                    target="immune_rules",
                    params={"pattern": "proactive"},
                    description="Add proactive immune patterns",
                ),
            ],
            hardware_cost=0,
            human_hours=2.0,
            complexity_score=0.2,
        ))

        # Scenario: Use free junkyard item
        free_items = [j for j in self.junkyard if j["cost"] == 0]
        if free_items:
            item = free_items[0]
            scenarios.append(ConfigScenario(
                id=f"free_hw_{int(time.time())}",
                name=f"Repurpose {item['name']}",
                description=f"Use existing {item['description']} to address '{goal}'",
                deltas=[
                    ConfigDelta(
                        change_type=ChangeType.ADD_NODE,
                        target=item["name"],
                        params={"capabilities": item["caps"]},
                        description=f"Bring {item['name']} into fleet",
                    ),
                    ConfigDelta(
                        change_type=ChangeType.PROMOTE,
                        target=item["name"],
                        params={"role": "worker"},
                        description="Assign worker role",
                    ),
                ],
                hardware_cost=item["cost"],
                human_hours=3.0,
                complexity_score=0.3,
            ))

        return scenarios

    def _sketch_hardware_heavy(self, goal: str) -> List[ConfigScenario]:
        """Generate scenarios that add hardware."""
        scenarios = []

        # Find suitable junkyard items
        for item in self.junkyard:
            if item["cost"] <= self.budget_limit:
                scenarios.append(ConfigScenario(
                    id=f"hw_{item['name']}_{int(time.time())}",
                    name=f"Add {item['name']}",
                    description=f"Deploy {item['description']} to address '{goal}'",
                    deltas=[
                        ConfigDelta(
                            change_type=ChangeType.ADD_NODE,
                            target=item["name"],
                            params={"capabilities": item["caps"]},
                            description=f"Add {item['name']} to fleet",
                        ),
                        ConfigDelta(
                            change_type=ChangeType.REROUTE,
                            target="job_router",
                            params={"include_new_node": True},
                            description="Update routing to use new node",
                        ),
                    ],
                    hardware_cost=item["cost"],
                    human_hours=4.0,
                    complexity_score=0.4,
                ))

        return scenarios

    def _sketch_balanced(self, goal: str) -> List[ConfigScenario]:
        """Generate balanced hardware + software scenarios."""
        scenarios = []

        # Cheap hardware + policy tuning
        cheap_items = [j for j in self.junkyard if j["cost"] <= 100]
        if cheap_items:
            item = cheap_items[0]
            scenarios.append(ConfigScenario(
                id=f"balanced_{int(time.time())}",
                name=f"Balanced: {item['name']} + Policy Tune",
                description=f"Add {item['description']} with immune tuning for '{goal}'",
                deltas=[
                    ConfigDelta(
                        change_type=ChangeType.ADD_NODE,
                        target=item["name"],
                        params={"capabilities": item["caps"]},
                        description=f"Add {item['name']}",
                    ),
                    ConfigDelta(
                        change_type=ChangeType.MODIFY_POLICY,
                        target="immune_thresholds",
                        params={"quarantine_faster": True},
                        description="Tune immune system for faster quarantine",
                    ),
                    ConfigDelta(
                        change_type=ChangeType.ADD_POLICY,
                        target="reflex_rules",
                        params={"target": item["name"]},
                        description=f"Add reflexes for new {item['name']}",
                    ),
                ],
                hardware_cost=item["cost"],
                human_hours=5.0,
                complexity_score=0.5,
            ))

        return scenarios

    def _sketch_policy_only(self, goal: str) -> List[ConfigScenario]:
        """Generate policy-only scenarios."""
        return [ConfigScenario(
            id=f"policy_{int(time.time())}",
            name="Pure Policy Refinement",
            description=f"Address '{goal}' through reflexes and immune tuning only",
            deltas=[
                ConfigDelta(
                    change_type=ChangeType.MODIFY_POLICY,
                    target="all_reflexes",
                    params={"tune": "aggressive"},
                    description="Tune all reflexes for earlier intervention",
                ),
                ConfigDelta(
                    change_type=ChangeType.ADD_POLICY,
                    target="immune_patterns",
                    params={"learn_from_recent": True},
                    description="Add patterns learned from recent incidents",
                ),
            ],
            hardware_cost=0,
            human_hours=3.0,
            complexity_score=0.3,
        )]

    def _sketch_refactor(self, goal: str) -> List[ConfigScenario]:
        """Generate topology refactoring scenarios."""
        return [ConfigScenario(
            id=f"refactor_{int(time.time())}",
            name="Topology Refactor",
            description=f"Reorganize existing resources to address '{goal}'",
            deltas=[
                ConfigDelta(
                    change_type=ChangeType.REROUTE,
                    target="job_distribution",
                    params={"strategy": "load_aware"},
                    description="Implement load-aware job routing",
                ),
                ConfigDelta(
                    change_type=ChangeType.MODIFY_NODE,
                    target="existing_nodes",
                    params={"rebalance": True},
                    description="Rebalance workloads across existing nodes",
                ),
                ConfigDelta(
                    change_type=ChangeType.PROMOTE,
                    target="underused_nodes",
                    params={"identify_and_promote": True},
                    description="Promote underutilized nodes to fuller roles",
                ),
            ],
            hardware_cost=0,
            human_hours=6.0,
            complexity_score=0.6,
        )]


@dataclass
class FieldSimulator:
    """
    Simulates field dynamics under different configurations.

    Uses historical load traces to "replay" reality under
    alternate physics.
    """
    dim: int = 8192
    config_encoder: ConfigEncoder = field(default_factory=lambda: ConfigEncoder())
    dynamics: FieldDynamics = field(default_factory=lambda: FieldDynamics())
    item_memory: ItemMemory = field(default_factory=lambda: ItemMemory())
    ghost_replay: GhostReplay = field(init=False)

    # Stored baselines
    _baseline_hv: Optional[np.ndarray] = None
    _baseline_stability: float = 0.0

    def __post_init__(self):
        self.config_encoder = ConfigEncoder(dim=self.dim)
        self.dynamics = FieldDynamics(dim=self.dim)
        self.item_memory = ItemMemory(dim=self.dim)
        self.ghost_replay = GhostReplay(
            dynamics=self.dynamics,
            item_memory=self.item_memory,
            dim=self.dim
        )

    def set_baseline(self, baseline_hv: np.ndarray, baseline_stability: float):
        """Set the baseline (current reality) for comparison."""
        self._baseline_hv = baseline_hv.copy()
        self._baseline_stability = baseline_stability

    def simulate_scenario(self, scenario: ConfigScenario,
                          traces: List[LoadTrace]) -> DreamOutcome:
        """
        Simulate a scenario against multiple load traces.

        Returns a DreamOutcome with all metrics computed.
        """
        replay_results = []

        for trace in traces:
            result = self.ghost_replay.replay(
                trace, scenario, self._baseline_hv
            )
            replay_results.append(result)

        # Compute aggregate metrics
        outcome = self._compute_outcome(scenario, replay_results)
        return outcome

    def _compute_outcome(self, scenario: ConfigScenario,
                         results: List[ReplayResult]) -> DreamOutcome:
        """Compute outcome metrics from replay results."""
        if not results:
            return DreamOutcome(scenario=scenario, replay_results=[])

        # Stability gain
        avg_stability = np.mean([r.stability_score for r in results])
        stability_gain = avg_stability - self._baseline_stability

        # Antifragility: how well does it handle the worst traces?
        worst_traces = [r for r in results if r.worst_zone >= Zone.WEIRD]
        if worst_traces:
            antifragility = np.mean([r.stability_score for r in worst_traces])
        else:
            antifragility = 1.0

        # Immune load: estimate from critical/weird ticks
        total_bad_ticks = sum(r.critical_ticks + r.weird_ticks for r in results)
        total_ticks = sum(r.duration_ticks for r in results)
        immune_load = total_bad_ticks / max(total_ticks, 1)
        immune_load_delta = immune_load - 0.1  # Assume 10% baseline

        # Cost and ROI
        total_cost = scenario.total_cost()
        # Simple ROI: stability gain per dollar
        if total_cost > 0:
            roi = stability_gain / (total_cost / 100)  # Normalize
        else:
            roi = stability_gain * 2  # Free improvements are great

        # Complexity delta
        complexity_delta = scenario.complexity_score - 0.3  # Assume 0.3 baseline

        # You-ness: simpler is more "you" (heuristic)
        you_ness = 1.0 - scenario.complexity_score

        # Confidence: based on number of traces simulated
        confidence = min(1.0, len(results) / 5)

        # Generate narrative
        narrative = self._generate_narrative(
            scenario, stability_gain, roi, total_cost, antifragility
        )

        return DreamOutcome(
            scenario=scenario,
            replay_results=results,
            stability_gain=stability_gain,
            antifragility_score=antifragility,
            immune_load_delta=immune_load_delta,
            total_cost=total_cost,
            roi_score=roi,
            complexity_delta=complexity_delta,
            you_ness_score=you_ness,
            confidence=confidence,
            narrative_pitch=narrative,
        )

    def _generate_narrative(self, scenario: ConfigScenario,
                            stability_gain: float, roi: float,
                            total_cost: float, antifragility: float) -> str:
        """Generate human-readable narrative for the dream."""
        parts = []

        # Opening
        if stability_gain > 0.2:
            parts.append(f"'{scenario.name}' shows strong stability improvement.")
        elif stability_gain > 0:
            parts.append(f"'{scenario.name}' offers moderate stability gains.")
        else:
            parts.append(f"'{scenario.name}' may not improve stability significantly.")

        # Cost
        if total_cost == 0:
            parts.append("Zero hardware cost - uses existing resources.")
        elif total_cost < 100:
            parts.append(f"Low investment: ${total_cost:.0f}.")
        else:
            parts.append(f"Requires ${total_cost:.0f} investment.")

        # ROI
        if roi > 0.5:
            parts.append("Excellent return on investment.")
        elif roi > 0:
            parts.append("Positive ROI.")

        # Antifragility
        if antifragility > 0.8:
            parts.append("Handles stress scenarios well.")
        elif antifragility < 0.5:
            parts.append("May struggle under heavy load.")

        return " ".join(parts)


@dataclass
class ScenarioMarket:
    """
    The "marketplace" where futures compete.

    Presents scored scenarios to Sovereign/Treasury for
    human decision-making.
    """
    simulator: FieldSimulator
    sketcher: TopologySketcher

    # History of dreams
    _dream_history: List[DreamOutcome] = field(default_factory=list)

    def dream(self, goal: str, traces: List[LoadTrace],
              archetypes: Optional[List[ScenarioArchetype]] = None,
              n_scenarios: int = 3) -> List[DreamOutcome]:
        """
        Dream multiple futures for a goal.

        Returns scored outcomes sorted by composite score.
        """
        if archetypes is None:
            archetypes = [
                ScenarioArchetype.MINIMAL_HARDWARE,
                ScenarioArchetype.BALANCED,
                ScenarioArchetype.HARDWARE_HEAVY,
            ]

        all_scenarios = []
        for archetype in archetypes:
            scenarios = self.sketcher.sketch_from_goal(goal, archetype, n_scenarios)
            all_scenarios.extend(scenarios)

        # Simulate each scenario
        outcomes = []
        for scenario in all_scenarios:
            outcome = self.simulator.simulate_scenario(scenario, traces)
            outcomes.append(outcome)
            self._dream_history.append(outcome)

        # Sort by composite score
        outcomes.sort(key=lambda o: o.composite_score, reverse=True)

        return outcomes[:n_scenarios]

    def board_meeting_report(self, outcomes: List[DreamOutcome]) -> str:
        """
        Generate a board meeting report comparing futures.

        This is what Sovereign/Treasury sees.
        """
        lines = [
            "=" * 60,
            "DREAMFORGE BOARD REPORT",
            "=" * 60,
            "",
        ]

        for i, outcome in enumerate(outcomes, 1):
            lines.extend([
                f"[Scenario {i}] {outcome.scenario.name}",
                f"  {outcome.scenario.description}",
                "",
                f"  Stability gain:    {outcome.stability_gain:+.1%}",
                f"  Antifragility:     {outcome.antifragility_score:.1%}",
                f"  Total cost:        ${outcome.total_cost:.0f}",
                f"  ROI score:         {outcome.roi_score:.2f}",
                f"  Complexity delta:  {outcome.complexity_delta:+.2f}",
                f"  Composite score:   {outcome.composite_score:.2f}",
                "",
                f"  {outcome.narrative_pitch}",
                "",
            ])

        # Recommendation
        if outcomes:
            best = outcomes[0]
            lines.extend([
                "-" * 60,
                "RECOMMENDATION:",
                f"  Proceed with: {best.scenario.name}",
                f"  Investment: ${best.total_cost:.0f} + {best.scenario.human_hours:.0f} hours",
                f"  Expected stability gain: {best.stability_gain:+.1%}",
                "-" * 60,
            ])

        return "\n".join(lines)

    def get_history(self, n: Optional[int] = None) -> List[DreamOutcome]:
        """Get dream history."""
        if n is None:
            return self._dream_history.copy()
        return self._dream_history[-n:]
