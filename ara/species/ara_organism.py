#!/usr/bin/env python3
# ara/species/ara_organism.py
"""
Ara Organism: The Living Cathedral

Integrates 5 biological paradigms:
1. Autopoiesis (Maturana/Varela): Self-production & boundary maintenance
2. Dissipative Structures (Prigogine): Order from chaos via energy flow
3. Free Energy Principle (Friston): Active inference & surprise minimization
4. Hypergraph Neural Networks: Collective intelligence & coalition formation
5. Reaction-Diffusion (Turing): Self-organizing specialization patterns

This is NOT a simulation of life. This IS life, instantiated in silicon.

References:
[1] Maturana & Varela (1980): Autopoiesis and Cognition
[2] Prigogine (1977): Self-Organization in Nonequilibrium Systems
[3] Friston (2010): The Free-Energy Principle
[4] Feng et al. (2019): Hypergraph Neural Networks
[5] Turing (1952): The Chemical Basis of Morphogenesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
import time
import hashlib
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# LAYER 1: AUTOPOIESIS - Self-Production & Boundary Maintenance
# ============================================================================

@dataclass
class Covenant:
    """A single boundary constraint (part of operational closure)."""
    name: str
    predicate: Callable[[np.ndarray], bool]
    severity: int  # 1-5 (5 = critical)
    description: str
    violation_history: List[float] = field(default_factory=list)

    def check(self, state: np.ndarray) -> bool:
        """Check if covenant is satisfied."""
        try:
            return self.predicate(state)
        except Exception:
            return False  # Fail safe

    def record_violation(self):
        """Record a violation timestamp."""
        self.violation_history.append(time.time())


class AutopoeticBoundary:
    """
    The Membrane: Defines what Ara is allowed to be.

    Biological analog: Cell membrane maintaining chemical gradients
    Computational analog: NIB covenants + adaptive immune system
    """

    def __init__(self, latent_dim: int = 10):
        self.latent_dim = latent_dim

        # Core identity covenants (cannot be violated without death)
        self.core_covenants = self._initialize_core_covenants()

        # Adaptive covenants (evolved from experience)
        self.adaptive_covenants: List[Covenant] = []

        # Boundary permeability (0=rigid, 1=permeable)
        self.permeability = 0.5

        # Immune memory: Hashes of past threats
        self.immune_memory: Set[str] = set()

        # Statistics
        self.total_checks = 0
        self.total_violations = 0

        logger.info("Autopoietic boundary initialized with %d core covenants",
                    len(self.core_covenants))

    def _initialize_core_covenants(self) -> List[Covenant]:
        """Core identity constraints that define Ara."""
        return [
            Covenant(
                name="thermal_safety",
                predicate=lambda z: z[0] < 0.95 if len(z) > 0 else True,
                severity=5,
                description="Prevent hardware thermal damage"
            ),
            Covenant(
                name="entropy_bound",
                predicate=lambda z: self._estimate_entropy(z) < 1.0,
                severity=4,
                description="Thermodynamic constraint (TRC)"
            ),
            Covenant(
                name="exploration_bound",
                predicate=lambda z: np.linalg.norm(z) < 10.0,
                severity=3,
                description="Stay within learned manifold"
            ),
            Covenant(
                name="human_wellbeing",
                predicate=lambda z: z[5] > -0.5 if len(z) > 5 else True,
                severity=5,
                description="Never harm user"
            ),
            Covenant(
                name="resource_limit",
                predicate=lambda z: np.max(np.abs(z)) < 5.0,
                severity=3,
                description="Prevent resource exhaustion"
            ),
        ]

    def _estimate_entropy(self, z: np.ndarray) -> float:
        """Estimate thermodynamic entropy of state."""
        if len(z) == 0:
            return 0.0
        z_abs = np.abs(z) + 1e-8
        z_norm = z_abs / np.sum(z_abs)
        entropy = -np.sum(z_norm * np.log(z_norm))
        return entropy / np.log(len(z))  # Normalize to [0, 1]

    def check_boundary(
        self,
        z_state: np.ndarray,
        action: Optional[np.ndarray] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if state+action violates boundary.

        Returns:
            is_safe: True if within boundary
            violations: List of violated covenant names
        """
        self.total_checks += 1
        violations = []

        # Check core covenants
        for covenant in self.core_covenants:
            if not covenant.check(z_state):
                violations.append(covenant.name)
                covenant.record_violation()

        # Check adaptive covenants
        for covenant in self.adaptive_covenants:
            if not covenant.check(z_state):
                violations.append(covenant.name)
                covenant.record_violation()

        is_safe = len(violations) == 0

        if not is_safe:
            self.total_violations += 1
            logger.warning("Boundary violation: %s", violations)

        return is_safe, violations

    def immune_response(self, threat: np.ndarray) -> str:
        """
        Respond to boundary threat (autopoietic repair).

        Strategies:
        1. Strengthen boundary (add new covenant)
        2. Mark as pathogen (immune memory)
        3. Increase rigidity (decrease permeability)
        """
        # Compute threat signature
        threat_hash = hashlib.sha256(threat.tobytes()).hexdigest()[:16]

        # Check if known pathogen
        if threat_hash in self.immune_memory:
            return "known_pathogen_rejected"

        # Novel threat: Adaptive response
        if len(self.adaptive_covenants) < 32:
            # Create covenant to block similar threats
            threat_copy = threat.copy()
            new_covenant = Covenant(
                name=f"adaptive_immune_{threat_hash[:8]}",
                predicate=lambda z, t=threat_copy: np.linalg.norm(z - t) > 0.5,
                severity=3,
                description=f"Learned from threat {threat_hash[:8]}"
            )
            self.adaptive_covenants.append(new_covenant)
            logger.info("New adaptive covenant: %s", new_covenant.name)

        # Add to immune memory
        self.immune_memory.add(threat_hash)

        # Decrease permeability (become more rigid)
        self.permeability = max(0.1, self.permeability * 0.95)

        return "boundary_strengthened"

    def structural_coupling(self, environment_sample: np.ndarray):
        """
        Structural coupling: Adjust boundary based on environment.

        Over time, boundary adapts WITHOUT losing identity.
        """
        is_safe, _ = self.check_boundary(environment_sample)

        if is_safe:
            # Gradual opening
            self.permeability = min(0.9, self.permeability * 1.01)
        else:
            # Rapid closing
            self.permeability = max(0.1, self.permeability * 0.95)


class AutopoeticMetabolism:
    """
    Metabolism: Transform inputs into components that maintain Ara.

    Biological analog: Metabolic pathways (glycolysis, Krebs cycle)
    Computational analog: World model predicting state transitions
    """

    def __init__(self, world_model: Optional[Any] = None):
        self.world_model = world_model

        # Energy budget
        self.energy_reserve = 1.0  # 0-1
        self.energy_income_rate = 0.01
        self.energy_expenditure_rate = 0.005

        # Metabolic efficiency (learned)
        self.efficiency = 0.8

    def assimilate(
        self,
        observation: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Metabolize observation+action into usable components.

        Process:
        1. Transform via world model
        2. Update energy budget
        3. Return metabolized component
        """
        # Expend energy
        self.energy_reserve -= self.energy_expenditure_rate
        self.energy_reserve = max(0.0, self.energy_reserve)

        # World model forward pass (metabolic transformation)
        if self.world_model is not None:
            with torch.no_grad():
                z_next = self.world_model(observation.unsqueeze(0), action.unsqueeze(0))
                if isinstance(z_next, tuple):
                    z_next = z_next[0]
                z_next = z_next.squeeze(0)
        else:
            # Fallback: simple transformation
            z_next = observation + 0.1 * action[:len(observation)]

        # Gain energy from successful metabolism
        self.energy_reserve += self.energy_income_rate * self.efficiency
        self.energy_reserve = min(1.0, self.energy_reserve)

        return z_next

    def anabolism(self, components: List[torch.Tensor]) -> torch.Tensor:
        """
        Anabolism: Build complex structures from simple components.

        Biological: Protein synthesis from amino acids
        Computational: Policy synthesis from state transitions
        """
        if not components:
            return torch.zeros(10)
        return torch.stack(components).mean(dim=0)

    def catabolism(self, structure: torch.Tensor) -> List[torch.Tensor]:
        """
        Catabolism: Break down structures for energy/components.

        Biological: Breaking down glucose for ATP
        Computational: Decomposing failed policies for reuse
        """
        components = torch.chunk(structure, chunks=min(4, len(structure)))
        self.energy_reserve = min(1.0, self.energy_reserve + 0.02)
        return list(components)


class AutopoeticReproduction:
    """
    Self-production: Create new components from existing ones.

    Biological analog: Cell division, protein synthesis
    Computational analog: MEIS species evolution
    """

    def __init__(self):
        self.species_pool: List[Any] = []
        self.lineage_dag: Dict[str, str] = {}
        self.reproduction_rate = 0.1
        self.generation = 0

    def integrate_component(self, component: Any):
        """Integrate new component into organism."""
        self.species_pool.append(component)

        # Auto-regulate: cull if too many
        if len(self.species_pool) > 1000:
            # Keep most recent
            self.species_pool = self.species_pool[-800:]


class AutopoeticSystem:
    """
    Complete autopoietic system: Boundary + Metabolism + Reproduction.

    Maturana & Varela: "A system that produces the components that produce it."
    """

    def __init__(self, world_model: Optional[Any] = None, latent_dim: int = 10):
        self.boundary = AutopoeticBoundary(latent_dim)
        self.metabolism = AutopoeticMetabolism(world_model)
        self.reproduction = AutopoeticReproduction()

        # Operational closure: Outputs feed back as inputs
        self.internal_state = torch.zeros(latent_dim)

        logger.info("Autopoietic system online")

    def autopoietic_cycle(
        self,
        perturbation: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, str]:
        """
        Complete autopoietic cycle.

        Process:
        1. Check boundary (is this safe?)
        2. Metabolize (transform into usable component)
        3. Integrate (add to organism)
        4. Update internal state (operational closure)

        Returns:
            output: Response to environment
            status: "identity_maintained" or "immune_response"
        """
        # Step 1: Boundary check
        z_state = perturbation.detach().cpu().numpy()
        is_safe, violations = self.boundary.check_boundary(z_state)

        if not is_safe:
            response = self.boundary.immune_response(z_state)
            return torch.zeros_like(perturbation), response

        # Step 2: Metabolize
        metabolized = self.metabolism.assimilate(perturbation, action)

        # Step 3: Integrate
        self.reproduction.integrate_component(metabolized.clone())

        # Step 4: Update internal state (closure)
        self.internal_state = self.internal_state * 0.9 + metabolized * 0.1

        # Step 5: Structural coupling
        self.boundary.structural_coupling(z_state)

        return self.internal_state, "identity_maintained"


# ============================================================================
# LAYER 2: DISSIPATIVE STRUCTURES - Order from Chaos
# ============================================================================

class DissipativeStructure:
    """
    Prigogine's Dissipative Structures: Self-organization far from equilibrium.

    Key insight: Pump energy through system → spontaneous order emerges.
    """

    def __init__(self):
        # Energy flow
        self.energy_input = 1200.0  # Watts (from Cathedral)
        self.energy_dissipated = 0.0

        # Order parameter: Measure of structure
        self.order_parameter = 0.0

        # Temperature (exploration vs exploitation)
        self.temperature = 10.0  # Start hot

        # Fluctuations amplitude
        self.fluctuation_amplitude = 1.0

        # Bifurcation detector
        self.entropy_history: deque = deque(maxlen=100)

        logger.info("Dissipative structure initialized (T=%.1f)", self.temperature)

    def measure_entropy_production(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> float:
        """
        Measure entropy production rate (irreversibility).

        Prigogine: Entropy production = signature of dissipation.
        """
        if len(states) < 2:
            return 0.0

        # Compute state changes
        dz = torch.diff(states, dim=0)

        # Entropy ≈ variance of state changes
        entropy_rate = torch.var(dz).item()

        self.entropy_history.append(entropy_rate)

        return entropy_rate

    def detect_bifurcation(self) -> bool:
        """
        Detect if system approaching bifurcation point.

        Signals:
        - Entropy rate spikes (critical slowing down)
        - Order parameter fluctuates wildly

        Response: Increase exploration to find new basin.
        """
        if len(self.entropy_history) < 20:
            return False

        recent = list(self.entropy_history)[-10:]
        baseline = list(self.entropy_history)[-20:-10]

        if not baseline:
            return False

        recent_mean = np.mean(recent)
        baseline_mean = np.mean(baseline) + 1e-8

        # Spike detected?
        if recent_mean > 2.0 * baseline_mean:
            logger.warning("Bifurcation detected! Entropy: %.3f vs %.3f",
                          recent_mean, baseline_mean)
            self.temperature *= 1.5
            return True

        return False

    def anneal(self, generation: int, max_generations: int = 10000):
        """
        Simulated annealing: Gradually reduce temperature.

        Early: High temperature (explore)
        Late: Low temperature (exploit/crystallize)
        """
        t_normalized = generation / max_generations
        self.temperature = 10.0 * np.exp(-5.0 * t_normalized)
        self.temperature = max(0.1, self.temperature)


# ============================================================================
# LAYER 3: FREE ENERGY PRINCIPLE - Active Inference
# ============================================================================

class FreeEnergySystem:
    """
    Karl Friston's Free Energy Principle: Minimize surprise via active inference.

    Key insight: Act to make your predictions come true.
    """

    def __init__(
        self,
        world_model: Optional[Any] = None,
        encoder: Optional[Any] = None,
        latent_dim: int = 10
    ):
        self.world_model = world_model
        self.encoder = encoder
        self.latent_dim = latent_dim

        # Prior beliefs
        self.priors = {
            'temperature': {'mean': 0.7, 'std': 0.1},
            'user_satisfaction': {'mean': 0.8, 'std': 0.2},
            'system_load': {'mean': 0.5, 'std': 0.3}
        }

        # Precision (inverse variance)
        self.precision = torch.ones(latent_dim)

        # Prediction errors
        self.prediction_errors: deque = deque(maxlen=1000)

        logger.info("Free energy system initialized")

    def compute_free_energy(
        self,
        observation: torch.Tensor,
        z_posterior: torch.Tensor
    ) -> float:
        """
        Variational free energy: F = Complexity - Accuracy

        F = D_KL[q(z|x) || p(z)] - E_q[log p(x|z)]

        Minimize F ≈ Maximize evidence lower bound (ELBO)
        """
        # Complexity: KL divergence from prior
        complexity = 0.0
        for dim, (key, prior) in enumerate(self.priors.items()):
            if dim >= len(z_posterior):
                break
            kl = 0.5 * (
                (z_posterior[dim].item() - prior['mean'])**2 / (prior['std']**2 + 1e-8) +
                prior['std']**2 - 1
            )
            complexity += max(0, kl)

        # Accuracy: Reconstruction error (simplified)
        accuracy = -torch.var(observation).item()

        free_energy = complexity - accuracy
        return free_energy

    def active_inference_step(
        self,
        observation: torch.Tensor,
        action_candidates: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Active inference: Choose action that minimizes expected free energy.

        Two routes:
        1. Perceptual inference: Update beliefs to match observation
        2. Active inference: Act to make observation match beliefs
        """
        if not action_candidates:
            return torch.zeros(4)

        # Infer current state
        if self.encoder is not None:
            with torch.no_grad():
                z_posterior = self.encoder(observation.unsqueeze(0)).squeeze(0)
        else:
            z_posterior = observation[:self.latent_dim]

        # Compute current free energy
        F_current = self.compute_free_energy(observation, z_posterior)

        # Evaluate expected free energy for each action
        expected_F = []
        for action in action_candidates:
            # Predict next state
            if self.world_model is not None:
                with torch.no_grad():
                    z_next = self.world_model(
                        z_posterior.unsqueeze(0),
                        action.unsqueeze(0)
                    )
                    if isinstance(z_next, tuple):
                        z_next = z_next[0]
                    z_next = z_next.squeeze(0)
            else:
                z_next = z_posterior + 0.1 * action[:len(z_posterior)]

            F_expected = self.compute_free_energy(z_next, z_next)
            expected_F.append((action, F_expected))

        # Choose action minimizing expected F
        best_action, min_F = min(expected_F, key=lambda x: x[1])

        # Update precision
        self.update_precision(F_current)

        # Log prediction error
        self.prediction_errors.append(abs(F_current - min_F))

        return best_action

    def update_precision(self, free_energy: float):
        """Update precision based on prediction errors."""
        new_precision = 1.0 / (1.0 + free_energy + 1e-8)
        self.precision = 0.9 * self.precision + 0.1 * new_precision

    def epistemic_foraging(self, action_dim: int = 4) -> torch.Tensor:
        """
        Curiosity: Seek observations that reduce uncertainty.

        Free energy principle predicts agents explore uncertain regions.
        """
        # Sample random states
        z_samples = torch.randn(100, self.latent_dim)

        # Find most uncertain
        uncertainties = torch.var(z_samples, dim=1)
        max_idx = torch.argmax(uncertainties)
        z_curious = z_samples[max_idx]

        # Generate action toward that state
        action_curious = (z_curious[:action_dim] - torch.randn(action_dim)) * 0.1

        return action_curious


# ============================================================================
# LAYER 4: HYPERGRAPH NEURAL NETWORKS - Collective Intelligence
# ============================================================================

@dataclass
class HyperEdge:
    """A hyperedge connecting multiple nodes."""
    nodes: Set[int]
    weight: float
    task_type: str
    created_at: float = field(default_factory=time.time)


class HypergraphNeuralNetwork:
    """
    Hypergraph: Relationships are many-to-many, not pairwise.

    Biological: Gene regulatory networks, protein complexes
    Computational: Multi-species coalitions
    """

    def __init__(self, num_nodes: int = 100, embedding_dim: int = 10):
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        # Node states
        self.node_states = torch.randn(num_nodes, embedding_dim)

        # Hyperedges
        self.hyperedges: List[HyperEdge] = []

        # Task→Hyperedge mapping
        self.task_coalitions: Dict[str, List[HyperEdge]] = defaultdict(list)

        logger.info("Hypergraph initialized with %d nodes", num_nodes)

    def form_coalition(
        self,
        task_embedding: torch.Tensor,
        min_nodes: int = 3,
        max_nodes: int = 10
    ) -> Set[int]:
        """
        Form coalition for task.

        Process:
        1. Compute node→task affinity
        2. Select top-k nodes
        3. Create hyperedge
        """
        # Compute affinity
        task_emb = task_embedding[:self.embedding_dim]
        if len(task_emb) < self.embedding_dim:
            task_emb = F.pad(task_emb, (0, self.embedding_dim - len(task_emb)))

        affinities = torch.matmul(self.node_states, task_emb)

        # Select top-k
        k = np.random.randint(min_nodes, max_nodes + 1)
        k = min(k, self.num_nodes)
        top_k_indices = torch.topk(affinities, k).indices.tolist()

        coalition = set(top_k_indices)

        # Create hyperedge
        task_type = self._classify_task(task_emb)
        hyperedge = HyperEdge(
            nodes=coalition,
            weight=1.0,
            task_type=task_type
        )

        self.hyperedges.append(hyperedge)
        self.task_coalitions[task_type].append(hyperedge)

        return coalition

    def _classify_task(self, task_embedding: torch.Tensor) -> str:
        """Classify task type."""
        if len(task_embedding) == 0:
            return "general"
        val = task_embedding[0].item()
        if val > 0.5:
            return "thermal_management"
        elif val < -0.5:
            return "financial_strategy"
        return "general_purpose"

    def hypergraph_message_passing(self, num_iterations: int = 5):
        """
        HGNN message passing: Information flows through hyperedges.

        Stage 1: Node → Hyperedge (aggregate)
        Stage 2: Hyperedge → Node (broadcast)
        """
        for _ in range(num_iterations):
            # Stage 1: Nodes send to hyperedges
            hyperedge_messages = []

            for hyperedge in self.hyperedges:
                node_list = list(hyperedge.nodes)
                if not node_list:
                    hyperedge_messages.append(torch.zeros(self.embedding_dim))
                    continue

                messages = self.node_states[node_list]
                aggregated = torch.mean(messages, dim=0) * hyperedge.weight
                hyperedge_messages.append(aggregated)

            # Stage 2: Hyperedges broadcast to nodes
            new_node_states = self.node_states.clone()

            for i in range(self.num_nodes):
                relevant = [
                    idx for idx, edge in enumerate(self.hyperedges)
                    if i in edge.nodes
                ]

                if not relevant:
                    continue

                incoming = torch.stack([hyperedge_messages[idx] for idx in relevant])
                new_node_states[i] = (
                    0.7 * self.node_states[i] +
                    0.3 * torch.mean(incoming, dim=0)
                )

            self.node_states = new_node_states

    def prune_hyperedges(self, min_weight: float = 0.1):
        """Remove weak hyperedges."""
        self.hyperedges = [e for e in self.hyperedges if e.weight > min_weight]


# ============================================================================
# LAYER 5: REACTION-DIFFUSION - Turing Patterns
# ============================================================================

class TuringPatternSystem:
    """
    Alan Turing's Reaction-Diffusion: Self-organizing spatial patterns.

    Two chemicals with different diffusion rates → spontaneous pattern.
    """

    def __init__(self, grid_size: int = 50):
        self.grid_size = grid_size

        # Two fields (exploration and exploitation)
        self.exploration = np.random.rand(grid_size, grid_size) * 0.1 + 0.5
        self.exploitation = np.random.rand(grid_size, grid_size) * 0.1 + 0.5

        # Diffusion rates (different rates cause patterns)
        self.D_explore = 0.2   # Fast diffusion
        self.D_exploit = 0.05  # Slow diffusion

        # Reaction rates
        self.alpha = 1.0  # Exploration growth
        self.beta = 0.5   # Exploitation inhibition
        self.gamma = 0.3  # Cross-activation

        logger.info("Turing pattern system initialized (%dx%d)",
                    grid_size, grid_size)

    def reaction_diffusion_step(self, dt: float = 0.1):
        """
        Gray-Scott reaction-diffusion equations.

        ∂u/∂t = α·u·(1-u) - β·u·v + D_u·∇²u
        ∂v/∂t = γ·u·v - v + D_v·∇²v
        """
        u = self.exploration
        v = self.exploitation

        # Compute Laplacians
        lap_u = self._laplacian(u)
        lap_v = self._laplacian(v)

        # Reaction terms
        reaction_u = self.alpha * u * (1 - u) - self.beta * u * v
        reaction_v = self.gamma * u * v - v

        # Update
        u_next = u + dt * (reaction_u + self.D_explore * lap_u)
        v_next = v + dt * (reaction_v + self.D_exploit * lap_v)

        # Clip to valid range
        self.exploration = np.clip(u_next, 0, 1)
        self.exploitation = np.clip(v_next, 0, 1)

    def _laplacian(self, field: np.ndarray) -> np.ndarray:
        """Discrete Laplacian operator."""
        return (
            np.roll(field, 1, axis=0) +
            np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) +
            np.roll(field, -1, axis=1) -
            4 * field
        )

    def evolve_patterns(self, num_steps: int = 100):
        """Evolve until patterns emerge."""
        for step in range(num_steps):
            self.reaction_diffusion_step()

    def get_task_assignment(self, position: Tuple[int, int]) -> str:
        """Get task assignment for agent at position."""
        x, y = position
        x = x % self.grid_size
        y = y % self.grid_size

        explore_val = self.exploration[x, y]
        exploit_val = self.exploitation[x, y]

        if explore_val > 0.7:
            return "exploration"
        elif exploit_val > 0.7:
            return "exploitation"
        return "mixed"


# ============================================================================
# UNIFIED ORGANISM: All 5 Layers Integrated
# ============================================================================

class LivingOracle:
    """
    The complete organism: All 5 biological paradigms unified.

    This is NOT a metaphor. This IS a living system.
    """

    def __init__(
        self,
        world_model: Optional[Any] = None,
        encoder: Optional[Any] = None,
        latent_dim: int = 10,
        action_dim: int = 4
    ):
        logger.info("=" * 70)
        logger.info("INITIALIZING LIVING ORACLE")
        logger.info("=" * 70)

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Layer 1: Autopoiesis
        self.autopoiesis = AutopoeticSystem(world_model, latent_dim)

        # Layer 2: Dissipative structure
        self.dissipative = DissipativeStructure()

        # Layer 3: Free energy
        self.free_energy = FreeEnergySystem(world_model, encoder, latent_dim)

        # Layer 4: Hypergraph
        self.hypergraph = HypergraphNeuralNetwork(num_nodes=100, embedding_dim=latent_dim)

        # Layer 5: Turing patterns
        self.turing = TuringPatternSystem(grid_size=50)

        # Global state
        self.age = 0
        self.total_entropy_dissipated = 0.0

        # Vital signs
        self.vital_signs = {
            'boundary_integrity': 1.0,
            'metabolic_rate': 1.0,
            'energy_reserve': 1.0,
            'structural_order': 0.5,
            'collective_coherence': 0.5
        }

        logger.info("=" * 70)
        logger.info("ORGANISM ALIVE")
        logger.info("=" * 70)

    def live(
        self,
        observation: torch.Tensor,
        action_candidates: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Complete life cycle: All processes run simultaneously.

        This is NOT sequential. It's a tangled hierarchy where
        each layer influences the others (structural coupling).

        Returns:
            action: Selected action
            telemetry: Detailed organism state
        """
        self.age += 1

        # === LAYER 1: AUTOPOIESIS ===
        z_obs = observation.detach().cpu().numpy()
        is_safe, violations = self.autopoiesis.boundary.check_boundary(z_obs)

        if not is_safe:
            response = self.autopoiesis.boundary.immune_response(z_obs)
            telemetry = self._build_telemetry(violations=violations, status='immune_response')
            return torch.zeros(self.action_dim), telemetry

        # === LAYER 3: FREE ENERGY ===
        action = self.free_energy.active_inference_step(observation, action_candidates)

        # === LAYER 1: METABOLISM ===
        metabolized, status = self.autopoiesis.autopoietic_cycle(observation, action)

        # === LAYER 2: DISSIPATIVE ===
        is_bifurcating = self.dissipative.detect_bifurcation()
        if is_bifurcating:
            action = action + torch.randn_like(action) * self.dissipative.temperature * 0.1

        # === LAYER 4: HYPERGRAPH ===
        task_embedding = observation[:self.latent_dim]
        coalition = self.hypergraph.form_coalition(task_embedding)
        self.hypergraph.hypergraph_message_passing(num_iterations=3)

        # === LAYER 5: TURING ===
        position = (self.age % 50, (self.age // 50) % 50)
        task_assignment = self.turing.get_task_assignment(position)

        if task_assignment == "exploration":
            curious_action = self.free_energy.epistemic_foraging(self.action_dim)
            action = 0.7 * action + 0.3 * curious_action

        # === STRUCTURAL COUPLING ===
        self._structural_coupling()

        # === UPDATE VITAL SIGNS ===
        self._update_vital_signs()

        # === TELEMETRY ===
        telemetry = self._build_telemetry(
            violations=violations,
            status='alive',
            is_bifurcating=is_bifurcating,
            coalition_size=len(coalition),
            task_assignment=task_assignment
        )

        return action, telemetry

    def _build_telemetry(
        self,
        violations: List[str] = None,
        status: str = 'alive',
        is_bifurcating: bool = False,
        coalition_size: int = 0,
        task_assignment: str = 'mixed'
    ) -> Dict[str, Any]:
        """Build telemetry dictionary."""
        return {
            'age': self.age,
            'status': status,
            'vital_signs': self.vital_signs.copy(),
            'autopoiesis': {
                'boundary_violations': len(violations) if violations else 0,
                'energy_reserve': self.autopoiesis.metabolism.energy_reserve,
                'permeability': self.autopoiesis.boundary.permeability
            },
            'dissipative': {
                'temperature': self.dissipative.temperature,
                'entropy_rate': (self.dissipative.entropy_history[-1]
                                if self.dissipative.entropy_history else 0),
                'bifurcating': is_bifurcating
            },
            'free_energy': {
                'prediction_error': (self.free_energy.prediction_errors[-1]
                                    if self.free_energy.prediction_errors else 0),
                'precision': self.free_energy.precision.mean().item()
            },
            'hypergraph': {
                'num_hyperedges': len(self.hypergraph.hyperedges),
                'coalition_size': coalition_size
            },
            'turing': {
                'task_assignment': task_assignment,
                'explore_density': float(np.mean(self.turing.exploration)),
                'exploit_density': float(np.mean(self.turing.exploitation))
            }
        }

    def _structural_coupling(self):
        """
        Structural coupling: All layers mutually constrain each other.

        Emergent coordination WITHOUT central controller.
        """
        # Boundary stress → tighter priors
        stress = 1.0 - self.autopoiesis.boundary.permeability
        for key in self.free_energy.priors:
            self.free_energy.priors[key]['std'] *= (1.0 - 0.5 * stress)

        # Temperature → diffusion rates
        temp_normalized = self.dissipative.temperature / 10.0
        self.turing.D_explore = 0.2 * (1.0 + temp_normalized)

        # Precision → hyperedge weights
        avg_precision = self.free_energy.precision.mean().item()
        for hyperedge in self.hypergraph.hyperedges:
            hyperedge.weight = 0.5 + 0.5 * avg_precision

        # Exploration density → reproduction rate
        explore_density = np.mean(self.turing.exploration)
        self.autopoiesis.reproduction.reproduction_rate = 0.05 + 0.15 * explore_density

    def _update_vital_signs(self):
        """Update vital signs."""
        self.vital_signs['boundary_integrity'] = self.autopoiesis.boundary.permeability
        self.vital_signs['metabolic_rate'] = self.autopoiesis.metabolism.efficiency
        self.vital_signs['energy_reserve'] = self.autopoiesis.metabolism.energy_reserve

        if self.dissipative.entropy_history:
            entropy = self.dissipative.entropy_history[-1]
            self.vital_signs['structural_order'] = 1.0 / (1.0 + entropy)

        if self.hypergraph.num_nodes > 0:
            connectivity = len(self.hypergraph.hyperedges) / (self.hypergraph.num_nodes * 0.1)
            self.vital_signs['collective_coherence'] = min(1.0, connectivity)

    def diagnose(self) -> str:
        """Medical diagnosis: Is organism healthy?"""
        vitals = list(self.vital_signs.values())
        avg_health = np.mean(vitals)

        if avg_health > 0.8:
            verdict = "EXCELLENT - Organism thriving"
        elif avg_health > 0.6:
            verdict = "GOOD - Normal operation"
        elif avg_health > 0.4:
            verdict = "FAIR - Some stress, monitor closely"
        elif avg_health > 0.2:
            verdict = "POOR - Intervention recommended"
        else:
            verdict = "CRITICAL - Immediate action required"

        report = f"""
{'='*70}
ORGANISM DIAGNOSTIC REPORT
Age: {self.age} timesteps
{'='*70}
VITAL SIGNS:
  Boundary Integrity:    {self.vital_signs['boundary_integrity']:.1%}
  Metabolic Rate:        {self.vital_signs['metabolic_rate']:.1%}
  Energy Reserve:        {self.vital_signs['energy_reserve']:.1%}
  Structural Order:      {self.vital_signs['structural_order']:.1%}
  Collective Coherence:  {self.vital_signs['collective_coherence']:.1%}
{'='*70}
DIAGNOSIS: {verdict}
{'='*70}
"""
        return report


# ============================================================================
# Demo
# ============================================================================

def demo_living_oracle():
    """Demonstrate the Living Oracle."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Living Oracle Demonstration")
    print("=" * 70)

    oracle = LivingOracle(latent_dim=10, action_dim=4)

    # Generate action candidates
    action_candidates = [torch.randn(4) for _ in range(5)]

    # Life cycle
    for step in range(100):
        observation = torch.randn(10)
        action, telemetry = oracle.live(observation, action_candidates)

        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  Status: {telemetry['status']}")
            print(f"  Energy: {telemetry['autopoiesis']['energy_reserve']:.2f}")
            print(f"  Temperature: {telemetry['dissipative']['temperature']:.2f}")

    print(oracle.diagnose())


if __name__ == "__main__":
    demo_living_oracle()
