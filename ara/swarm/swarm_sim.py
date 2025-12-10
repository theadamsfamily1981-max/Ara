#!/usr/bin/env python3
"""
Stupidly-Cubed Micro-Agent Swarm
================================

"So dumb we can run millions of them."

Architecture:
1. Micro-agents: Tiny state, pattern match, emit votes + pheromone deltas
2. Hive State Hypervector: Hypervector-of-hypervectors encoding swarm state
3. Quantum-Inspired Control Plane: Treats swarm as state vector over modes

This is NOT a magic theorem prover. It's a massively parallel conjecture
& search engine that explores proof space and spots patterns.

Usage:
    python swarm_sim.py --mode explore
    python swarm_sim.py --mode math --problem riemann
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
import time


# ============================================================================
# Hypervector Operations
# ============================================================================

D = 1024  # Hypervector dimension


def random_hv(d: int = D) -> np.ndarray:
    """Generate random bipolar hypervector."""
    return np.random.choice([-1, 1], size=d).astype(np.float32)


def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Bind two hypervectors (element-wise XOR for bipolar = multiply)."""
    return a * b


def bundle(vectors: List[np.ndarray]) -> np.ndarray:
    """Bundle hypervectors (majority vote)."""
    if not vectors:
        return np.zeros(D, dtype=np.float32)
    stacked = np.stack(vectors)
    return np.sign(np.sum(stacked, axis=0))


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
    """Permute hypervector (cyclic shift for temporal encoding)."""
    return np.roll(v, shift)


# ============================================================================
# Basis Patterns (Canonical Modes)
# ============================================================================

class SwarmMode(Enum):
    """Canonical swarm modes (basis vectors)."""
    PRODUCTIVE_EXPLORATION = "productive_exploration"
    LOCAL_MINIMA_THRASH = "local_minima_thrash"
    CHAOTIC_SEARCH = "chaotic_search"
    BORING_BUT_SAFE = "boring_but_safe"
    CONVERGENT = "convergent"
    DIVERGENT = "divergent"


# Pre-generate basis hypervectors for modes
MODE_BASIS: Dict[SwarmMode, np.ndarray] = {
    mode: random_hv() for mode in SwarmMode
}


def classify_mode(hive_hv: np.ndarray) -> Tuple[SwarmMode, float]:
    """Classify current hive state into nearest basis mode."""
    best_mode = SwarmMode.BORING_BUT_SAFE
    best_sim = -1.0
    for mode, basis in MODE_BASIS.items():
        sim = similarity(hive_hv, basis)
        if sim > best_sim:
            best_sim = sim
            best_mode = mode
    return best_mode, best_sim


# ============================================================================
# Micro-Agent
# ============================================================================

@dataclass
class MicroAgent:
    """
    Stupidly simple agent.

    State: A handful of bits
    - pattern_id: Current pattern being tracked
    - pheromone: Local pheromone level
    - counter: Action counter

    Behavior:
    - Pattern match on local window
    - Emit vote (promising / dead end)
    - Emit pheromone delta
    """
    id: int
    pattern_id: int = 0
    pheromone: float = 0.5
    counter: int = 0
    role_hv: np.ndarray = field(default_factory=lambda: random_hv(64))

    # Tiny state
    last_vote: int = 0  # -1 = dead end, 0 = neutral, 1 = promising
    local_score: float = 0.0

    def step(
        self,
        local_window: np.ndarray,
        global_temp: float = 1.0,
        pheromone_field: Optional[np.ndarray] = None,
    ) -> Tuple[int, float, np.ndarray]:
        """
        One step of micro-agent behavior.

        Args:
            local_window: Small view of problem state
            global_temp: Sampling temperature from control plane
            pheromone_field: Global pheromone levels

        Returns:
            vote: -1 (dead end), 0 (neutral), 1 (promising)
            pheromone_delta: How much to boost/decay local pheromone
            action_hv: Hypervector encoding this agent's action
        """
        self.counter += 1

        # Pattern match (stupidly simple: correlation with role)
        if len(local_window) > 0:
            # Project local window onto our pattern
            proj = np.mean(local_window[:min(len(local_window), 64)])
            self.local_score = proj
        else:
            self.local_score = 0.0

        # Vote based on score + temperature
        noise = np.random.randn() * global_temp * 0.3
        adjusted_score = self.local_score + noise

        if adjusted_score > 0.3:
            self.last_vote = 1  # Promising
        elif adjusted_score < -0.3:
            self.last_vote = -1  # Dead end
        else:
            self.last_vote = 0  # Neutral

        # Pheromone delta
        pheromone_delta = self.last_vote * 0.1

        # Update local pheromone
        self.pheromone = np.clip(self.pheromone + pheromone_delta, 0.0, 1.0)

        # Encode action as hypervector
        action_hv = bind(self.role_hv, np.sign(np.random.randn(64)))

        return self.last_vote, pheromone_delta, action_hv

    def to_hv(self) -> np.ndarray:
        """Encode agent state as hypervector."""
        # Bind together: role, pheromone level, recent vote
        pheromone_hv = random_hv(64) if self.pheromone > 0.5 else -random_hv(64)
        vote_hv = self.role_hv * self.last_vote if self.last_vote != 0 else np.zeros(64)
        return bundle([self.role_hv, pheromone_hv, vote_hv])


# ============================================================================
# Hive State (Hypervector of Hypervectors)
# ============================================================================

@dataclass
class HiveState:
    """
    The hive mind state = hypervector of hypervectors.

    Level 0: Agent hypervectors (role, actions, pheromone)
    Level 1: Job hypervectors (bundle of agents that touched it)
    Level 2: Hive hypervector (bundle of all jobs)

    4D view: [layer, modality, latency-band, risk-band]
    """
    # Core hive hypervector
    hive_hv: np.ndarray = field(default_factory=lambda: np.zeros(D))

    # Job-level hypervectors
    job_hvs: Dict[str, np.ndarray] = field(default_factory=dict)

    # Temporal history (for control plane)
    history: List[np.ndarray] = field(default_factory=list)
    history_length: int = 10

    # Statistics
    entropy: float = 0.5
    diversity: float = 0.5
    convergence: float = 0.0

    def update(self, agent_hvs: List[np.ndarray], job_id: str = "default") -> None:
        """Update hive state from agent hypervectors."""
        if not agent_hvs:
            return

        # Bundle agents into job hypervector
        job_hv = bundle(agent_hvs)

        # Pad to full dimension if needed
        if len(job_hv) < D:
            job_hv = np.concatenate([job_hv, np.zeros(D - len(job_hv))])

        self.job_hvs[job_id] = job_hv

        # Bundle all jobs into hive hypervector
        self.hive_hv = bundle(list(self.job_hvs.values()))

        # Update history
        self.history.append(self.hive_hv.copy())
        if len(self.history) > self.history_length:
            self.history.pop(0)

        # Compute statistics
        self._compute_stats()

    def _compute_stats(self) -> None:
        """Compute hive statistics."""
        # Entropy (how spread out the hive vector is)
        abs_hv = np.abs(self.hive_hv) + 1e-8
        probs = abs_hv / np.sum(abs_hv)
        self.entropy = -np.sum(probs * np.log(probs + 1e-8)) / np.log(D)

        # Diversity (how different are recent states)
        if len(self.history) >= 2:
            sims = [similarity(self.history[-1], h) for h in self.history[:-1]]
            self.diversity = 1.0 - np.mean(sims)
        else:
            self.diversity = 0.5

        # Convergence (are we settling down)
        if len(self.history) >= 3:
            recent_sims = [
                similarity(self.history[-1], self.history[-2]),
                similarity(self.history[-2], self.history[-3]),
            ]
            self.convergence = np.mean(recent_sims)
        else:
            self.convergence = 0.0

    def get_mode(self) -> Tuple[SwarmMode, float]:
        """Get current swarm mode."""
        return classify_mode(self.hive_hv)


# ============================================================================
# Quantum-Inspired Control Plane
# ============================================================================

@dataclass
class ControlPlane:
    """
    Quantum-inspired control plane.

    Treats swarm as state vector over modes.
    Applies "unitary-ish" operators to steer.
    """
    # Control parameters emitted to swarm
    temperature: float = 1.0          # Sampling temperature
    pheromone_decay: float = 0.95     # How fast pheromone fades
    stress_level: float = 0.1         # σ for hormesis
    exploration_bias: float = 0.5     # 0 = exploit, 1 = explore

    # Targets
    target_entropy: float = 0.5
    target_diversity: float = 0.4

    # Control matrices (simplified)
    explore_op: np.ndarray = field(default_factory=lambda: np.eye(D) * 1.1)
    exploit_op: np.ndarray = field(default_factory=lambda: np.eye(D) * 0.9)

    def control_step(self, hive: HiveState) -> Dict[str, float]:
        """
        One step of control plane.

        1. Observe hive state
        2. Classify mode
        3. Apply control operator
        4. Emit new parameters

        Returns:
            New control parameters for micro-agents
        """
        mode, confidence = hive.get_mode()

        # Observe statistics
        entropy = hive.entropy
        diversity = hive.diversity
        convergence = hive.convergence

        # Decision logic
        if mode == SwarmMode.LOCAL_MINIMA_THRASH:
            # Stuck! Increase stress and exploration
            self.stress_level = min(0.3, self.stress_level + 0.05)
            self.exploration_bias = min(0.9, self.exploration_bias + 0.1)
            self.temperature = min(2.0, self.temperature * 1.1)

        elif mode == SwarmMode.CHAOTIC_SEARCH:
            # Too random! Decrease temperature, increase exploitation
            self.temperature = max(0.3, self.temperature * 0.9)
            self.exploration_bias = max(0.1, self.exploration_bias - 0.1)
            self.stress_level = max(0.05, self.stress_level - 0.02)

        elif mode == SwarmMode.CONVERGENT:
            # Settling down - might be good or stuck
            if convergence > 0.9:
                # Probably stuck, perturb
                self.stress_level = min(0.2, self.stress_level + 0.03)
                self.temperature = min(1.5, self.temperature * 1.05)

        elif mode == SwarmMode.PRODUCTIVE_EXPLORATION:
            # Good! Maintain
            pass

        else:
            # Boring but safe - nudge toward exploration
            self.exploration_bias = min(0.6, self.exploration_bias + 0.02)

        # Entropy regulation
        if entropy < self.target_entropy - 0.1:
            self.temperature *= 1.05
        elif entropy > self.target_entropy + 0.1:
            self.temperature *= 0.95

        # Diversity regulation
        if diversity < self.target_diversity - 0.1:
            self.exploration_bias = min(0.8, self.exploration_bias + 0.05)

        # Clamp parameters
        self.temperature = np.clip(self.temperature, 0.1, 3.0)
        self.exploration_bias = np.clip(self.exploration_bias, 0.0, 1.0)
        self.stress_level = np.clip(self.stress_level, 0.01, 0.5)

        return {
            "temperature": self.temperature,
            "pheromone_decay": self.pheromone_decay,
            "stress_level": self.stress_level,
            "exploration_bias": self.exploration_bias,
        }


# ============================================================================
# Swarm Simulation
# ============================================================================

class SwarmSimulation:
    """
    Main simulation runner.

    Puts together:
    - Micro-agents
    - Hive state
    - Control plane
    - Problem environment
    """

    def __init__(
        self,
        num_agents: int = 100,
        problem_dim: int = 64,
    ):
        self.num_agents = num_agents
        self.problem_dim = problem_dim

        # Initialize components
        self.agents = [MicroAgent(id=i) for i in range(num_agents)]
        self.hive = HiveState()
        self.control = ControlPlane()

        # Problem state (abstract search space)
        self.problem_state = np.random.randn(problem_dim)

        # Pheromone field
        self.pheromone_field = np.ones(problem_dim) * 0.5

        # Statistics
        self.tick = 0
        self.history: List[Dict] = []

    def step(self) -> Dict:
        """One simulation step."""
        self.tick += 1

        # Get control parameters
        params = self.control.control_step(self.hive)

        # Run all micro-agents
        agent_hvs = []
        total_votes = {"promising": 0, "dead_end": 0, "neutral": 0}

        for agent in self.agents:
            # Each agent sees a local window
            start = (agent.id * 7) % self.problem_dim
            local_window = self.problem_state[start:start+8]

            # Agent step
            vote, pheromone_delta, action_hv = agent.step(
                local_window=local_window,
                global_temp=params["temperature"],
                pheromone_field=self.pheromone_field,
            )

            # Accumulate
            agent_hvs.append(agent.to_hv())

            if vote > 0:
                total_votes["promising"] += 1
            elif vote < 0:
                total_votes["dead_end"] += 1
            else:
                total_votes["neutral"] += 1

            # Update pheromone field
            self.pheromone_field[start:start+8] += pheromone_delta * 0.1

        # Decay pheromone
        self.pheromone_field *= params["pheromone_decay"]
        self.pheromone_field = np.clip(self.pheromone_field, 0.0, 1.0)

        # Update hive state
        self.hive.update(agent_hvs, job_id="main")

        # Apply stress to problem state (hormesis)
        if np.random.random() < params["stress_level"]:
            perturbation = np.random.randn(self.problem_dim) * params["stress_level"]
            self.problem_state += perturbation

        # Get mode
        mode, mode_confidence = self.hive.get_mode()

        # Record history
        snapshot = {
            "tick": self.tick,
            "mode": mode.value,
            "mode_confidence": mode_confidence,
            "entropy": self.hive.entropy,
            "diversity": self.hive.diversity,
            "convergence": self.hive.convergence,
            "temperature": params["temperature"],
            "exploration_bias": params["exploration_bias"],
            "stress_level": params["stress_level"],
            "votes": total_votes,
        }
        self.history.append(snapshot)

        return snapshot

    def run(self, steps: int = 100, verbose: bool = True) -> List[Dict]:
        """Run simulation for N steps."""
        for _ in range(steps):
            snapshot = self.step()
            if verbose:
                print(
                    f"\rTick {snapshot['tick']:4d} | "
                    f"Mode: {snapshot['mode']:24s} | "
                    f"Entropy: {snapshot['entropy']:.3f} | "
                    f"Diversity: {snapshot['diversity']:.3f} | "
                    f"Temp: {snapshot['temperature']:.2f}",
                    end=""
                )
        if verbose:
            print()
        return self.history


# ============================================================================
# Math Search Mode
# ============================================================================

class MathSearchSwarm(SwarmSimulation):
    """
    Swarm configured for mathematical search.

    Micro-agents = local rule appliers
    Problem state = partial proof / conjecture space
    Hive state = where the search is going
    Control plane = research director
    """

    def __init__(
        self,
        num_agents: int = 500,
        search_dim: int = 256,
        problem_name: str = "generic",
    ):
        super().__init__(num_agents=num_agents, problem_dim=search_dim)
        self.problem_name = problem_name

        # Encode problem as basis vector
        self.problem_hv = random_hv()

        # Track "interesting" regions
        self.interesting_regions: List[np.ndarray] = []

        # Conjecture candidates
        self.conjectures: List[Dict] = []

    def step(self) -> Dict:
        """One step with math-specific tracking."""
        snapshot = super().step()

        # Track interesting regions (high pheromone)
        hot_regions = np.where(self.pheromone_field > 0.7)[0]
        if len(hot_regions) > 0:
            region_hv = self.problem_state[hot_regions[:64]] if len(hot_regions) >= 64 else self.problem_state[:64]
            self.interesting_regions.append(region_hv)

            # Limit stored regions
            if len(self.interesting_regions) > 100:
                self.interesting_regions.pop(0)

        # Periodically emit conjectures (cluster interesting regions)
        if self.tick % 50 == 0 and len(self.interesting_regions) > 10:
            conjecture = {
                "tick": self.tick,
                "num_hot_regions": len(hot_regions),
                "pheromone_max": float(np.max(self.pheromone_field)),
                "mode": snapshot["mode"],
                "confidence": snapshot["mode_confidence"],
            }
            self.conjectures.append(conjecture)

        snapshot["problem"] = self.problem_name
        snapshot["hot_regions"] = len(hot_regions)
        snapshot["conjectures"] = len(self.conjectures)

        return snapshot


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Stupidly-Cubed Micro-Agent Swarm")
    parser.add_argument("--mode", choices=["explore", "math"], default="explore")
    parser.add_argument("--problem", default="generic", help="Problem name for math mode")
    parser.add_argument("--agents", type=int, default=100, help="Number of micro-agents")
    parser.add_argument("--steps", type=int, default=200, help="Simulation steps")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")

    args = parser.parse_args()

    print(f"Starting swarm simulation (mode={args.mode}, agents={args.agents})")
    print("=" * 60)

    if args.mode == "math":
        sim = MathSearchSwarm(
            num_agents=args.agents,
            search_dim=256,
            problem_name=args.problem,
        )
    else:
        sim = SwarmSimulation(num_agents=args.agents)

    history = sim.run(steps=args.steps, verbose=not args.quiet)

    print("\n" + "=" * 60)
    print("Final Statistics:")
    print(f"  Mode: {history[-1]['mode']}")
    print(f"  Entropy: {history[-1]['entropy']:.4f}")
    print(f"  Diversity: {history[-1]['diversity']:.4f}")
    print(f"  Convergence: {history[-1]['convergence']:.4f}")

    if args.mode == "math":
        print(f"  Conjectures generated: {history[-1]['conjectures']}")


if __name__ == "__main__":
    main()
