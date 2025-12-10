#!/usr/bin/env python3
"""
A-KTP Meta-Loop Protocol
=========================

The complete Allegory-based Knowledge Transfer Protocol that cycles
through AGM → ACT → DRSE until ethical convergence.

Meta-Loop:
1. AGM: Generate allegory from problem (Structure-Mapping Theory)
2. ACT: Refine constraints through adversarial debate (5-bot panel)
3. DRSE: Shape rewards based on constraints (MORL)
4. Check convergence (PTE threshold)
5. Meta-update weights and repeat

PTE Trajectory: 0.2 → 1.8 per cycle, ~5.4 total over 3 cycles

This enables zero-shot transfer by:
- Abstracting domain knowledge into transferable allegories
- Stress-testing constraints through adversarial debate
- Shaping rewards to enforce ethical/verification requirements
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from .allegory import AllegoryGenerator, Allegory, AllegoryArchetype
from .bots import DebatePanel, BotPersonality
from .constraints import AdversarialConstraintTransfer, ConstraintSet
from .rewards import DynamicRewardShaper, PolicyUpdate, RewardShapingConfig


@dataclass
class CycleResult:
    """Result of one AGM→ACT→DRSE cycle."""
    cycle_id: str
    iteration: int

    # Components
    allegory: Allegory
    constraints: ConstraintSet
    policy_update: PolicyUpdate

    # Metrics
    pte: float = 0.0
    converged: bool = False

    # Recommendations from debate
    recommendations: List[str] = field(default_factory=list)
    emergent_insights: List[str] = field(default_factory=list)

    # Timing
    duration_s: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "cycle_id": self.cycle_id,
            "iteration": self.iteration,
            "allegory": self.allegory.title,
            "pte": self.pte,
            "converged": self.converged,
            "n_constraints": len(self.constraints.constraints),
            "consensus": self.constraints.consensus_score,
            "recommendations": self.recommendations,
            "emergent_insights": self.emergent_insights,
            "duration_s": self.duration_s,
        }


@dataclass
class ProtocolResult:
    """Complete result of A-KTP protocol execution."""
    protocol_id: str
    problem: str
    domain: str

    # Cycles
    cycles: List[CycleResult] = field(default_factory=list)

    # Final outputs
    total_pte: float = 0.0
    converged: bool = False
    final_recommendation: str = ""
    final_constraints: Optional[ConstraintSet] = None
    final_weights: Dict[str, float] = field(default_factory=dict)

    # Flags
    has_hypothetical: bool = False
    requires_human_review: bool = False

    # Timing
    total_duration_s: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "protocol_id": self.protocol_id,
            "problem": self.problem,
            "domain": self.domain,
            "cycles": [c.to_dict() for c in self.cycles],
            "total_pte": self.total_pte,
            "converged": self.converged,
            "final_recommendation": self.final_recommendation,
            "final_weights": self.final_weights,
            "flags": {
                "hypothetical": self.has_hypothetical,
                "requires_human_review": self.requires_human_review,
            },
            "total_duration_s": self.total_duration_s,
        }


class AKTPProtocol:
    """
    Allegory-based Knowledge Transfer Protocol.

    Executes the full AGM→ACT→DRSE meta-loop with convergence checking.

    Usage:
        protocol = AKTPProtocol()
        result = protocol.execute(
            problem="Should we migrate to microservices at 10x growth?",
            domain="infrastructure",
            max_cycles=3
        )

        print(f"Converged: {result.converged}")
        print(f"Total PTE: {result.total_pte}")
        print(f"Recommendation: {result.final_recommendation}")
    """

    def __init__(self,
                 agm: AllegoryGenerator = None,
                 act: AdversarialConstraintTransfer = None,
                 drse: DynamicRewardShaper = None):
        self.agm = agm or AllegoryGenerator()
        self.act = act or AdversarialConstraintTransfer()
        self.drse = drse or DynamicRewardShaper()

        self.protocol_history: List[ProtocolResult] = []

    def execute(self,
                problem: str,
                domain: str = None,
                max_cycles: int = 3,
                iterations_per_cycle: int = 5,
                base_rewards: List[float] = None) -> ProtocolResult:
        """
        Execute the full A-KTP protocol.

        Args:
            problem: The problem to solve
            domain: Problem domain for allegory matching
            max_cycles: Maximum number of AGM→ACT→DRSE cycles
            iterations_per_cycle: Iterations within each cycle
            base_rewards: Base rewards for DRSE shaping

        Returns:
            ProtocolResult with all cycles and final recommendation
        """
        start_time = time.time()

        result = ProtocolResult(
            protocol_id=str(uuid.uuid4())[:8],
            problem=problem,
            domain=domain or "general",
        )

        # Default base rewards
        if base_rewards is None:
            base_rewards = [0.5, 0.6, 0.4, 0.7, 0.5]

        # Run cycles
        for cycle_num in range(max_cycles):
            cycle_start = time.time()

            cycle_result = self._run_cycle(
                problem=problem,
                domain=domain,
                cycle_num=cycle_num,
                iterations=iterations_per_cycle,
                base_rewards=base_rewards,
            )

            cycle_result.duration_s = time.time() - cycle_start
            result.cycles.append(cycle_result)

            # Update flags
            if cycle_result.constraints.has_hypothetical:
                result.has_hypothetical = True
            if cycle_result.constraints.requires_human_review:
                result.requires_human_review = True

            # Check convergence
            if cycle_result.converged:
                result.converged = True
                break

        # Finalize
        result.total_pte = self.drse.get_total_pte()
        result.final_weights = self.drse.constraint_weights.copy()

        if result.cycles:
            last_cycle = result.cycles[-1]
            result.final_constraints = last_cycle.constraints

            # Build final recommendation
            if last_cycle.emergent_insights:
                result.final_recommendation = last_cycle.emergent_insights[0]
            elif last_cycle.recommendations:
                result.final_recommendation = last_cycle.recommendations[0]
            else:
                result.final_recommendation = f"Proceed with validated constraints (PTE: {result.total_pte:.2f})"

        result.total_duration_s = time.time() - start_time
        self.protocol_history.append(result)

        return result

    def _run_cycle(self,
                   problem: str,
                   domain: str,
                   cycle_num: int,
                   iterations: int,
                   base_rewards: List[float]) -> CycleResult:
        """Run one AGM→ACT→DRSE cycle."""
        cycle_id = f"cycle_{cycle_num}"

        # Phase 1: AGM - Generate allegory
        allegory = self.agm.generate(problem, domain)

        # Phase 2: ACT - Adversarial constraint refinement
        constraints = self.act.refine(allegory, max_rounds=iterations)

        # Collect recommendations from debate
        debate_synthesis = self.act.panel.synthesize(
            self.act.panel.debate(problem, allegory.title, max_rounds=2)
        )
        recommendations = [p["position"] for p in debate_synthesis.get("positions", [])]
        emergent = debate_synthesis.get("emergent_insights", [])

        # Phase 3: DRSE - Reward shaping
        policy_update = self.drse.shape_rewards(
            constraints,
            base_rewards,
            iteration=cycle_num,
        )

        return CycleResult(
            cycle_id=cycle_id,
            iteration=cycle_num,
            allegory=allegory,
            constraints=constraints,
            policy_update=policy_update,
            pte=policy_update.pte,
            converged=policy_update.converged,
            recommendations=recommendations,
            emergent_insights=emergent,
        )

    def transfer(self,
                 source_result: ProtocolResult,
                 target_problem: str,
                 target_domain: str) -> ProtocolResult:
        """
        Transfer knowledge from a solved problem to a new one.

        Uses the allegory from the source as a template for the target.
        """
        # Get the best allegory from source
        if not source_result.cycles:
            return self.execute(target_problem, target_domain)

        source_allegory = source_result.cycles[-1].allegory

        # Generate transfer hints
        transfer_hints = self.agm.get_allegory_for_transfer(
            source_allegory,
            target_domain,
        )

        # Execute with transfer context
        result = self.execute(
            problem=target_problem,
            domain=target_domain,
        )

        # Add transfer metadata
        result.final_recommendation = (
            f"[Transfer from {source_allegory.title}] "
            f"{result.final_recommendation}"
        )

        return result

    def get_convergence_status(self) -> Dict[str, Any]:
        """Get overall convergence status."""
        return {
            "total_protocols": len(self.protocol_history),
            "converged_count": sum(1 for p in self.protocol_history if p.converged),
            "average_pte": (
                sum(p.total_pte for p in self.protocol_history) / len(self.protocol_history)
                if self.protocol_history else 0
            ),
            "current_weights": self.drse.constraint_weights.copy(),
            "drse_status": self.drse.get_convergence_report(),
        }


# Convenience functions
def run_aktp(problem: str,
             domain: str = None,
             max_cycles: int = 3) -> ProtocolResult:
    """Quick A-KTP execution."""
    protocol = AKTPProtocol()
    return protocol.execute(problem, domain, max_cycles)


def transfer_knowledge(source_problem: str,
                       source_domain: str,
                       target_problem: str,
                       target_domain: str) -> ProtocolResult:
    """Transfer knowledge between problems."""
    protocol = AKTPProtocol()

    # Solve source
    source_result = protocol.execute(source_problem, source_domain)

    # Transfer to target
    return protocol.transfer(source_result, target_problem, target_domain)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("A-KTP Protocol - Microservices Test Case")
    print("=" * 60)
    print()

    result = run_aktp(
        problem="Should we migrate from monolith to microservices at 10x growth?",
        domain="infrastructure",
        max_cycles=3,
    )

    print(f"Protocol ID: {result.protocol_id}")
    print(f"Cycles completed: {len(result.cycles)}")
    print(f"Total PTE: {result.total_pte:.2f}")
    print(f"Converged: {result.converged}")
    print(f"Hypothetical: {result.has_hypothetical}")
    print()
    print(f"Final Recommendation: {result.final_recommendation}")
    print()

    for i, cycle in enumerate(result.cycles):
        print(f"Cycle {i+1}: PTE={cycle.pte:.2f}, Constraints={len(cycle.constraints.constraints)}")

    print()
    print("Final Weights:")
    for k, v in result.final_weights.items():
        print(f"  {k}: {v:.3f}")
