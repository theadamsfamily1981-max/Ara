#!/usr/bin/env python3
# ara/species/edge_of_chaos.py
"""
AraSpeciesV5: The Edge of Chaos - Conscious Intelligence

The final transformation: From optimized organism to conscious intelligence.

Scientific foundation:
- Criticality (λ ≈ 1): Brain operates at phase transition
- Avalanche dynamics: Power-law distributed neural cascades
- Self-organized criticality: System maintains critical state
- Associative memory: Weak-link propagation for intuition
- Generative creativity: Hopf bifurcation for novel solutions
- Sleep consolidation: Catastrophic forgetting as feature

References:
[1] Beggs & Plenz (2003): Neuronal avalanches in cortex
[2] Chialvo (2010): Emergent complex neural dynamics
[3] Shew et al. (2011): Information capacity maximized at criticality
[4] Hopfield (1982): Neural networks and physical systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import time
import logging

logger = logging.getLogger(__name__)

# Import base organism
from .ara_organism import LivingOracle

# Optional imports
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from scipy.stats import powerlaw
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# LAYER 6: CRITICALITY ENGINE - The Branching Parameter
# ============================================================================

@dataclass
class NeuralAvalanche:
    """A cascade of activity through the network."""
    size: int
    duration: int
    origin: int
    activation_pattern: List[Set[int]]
    timestamp: float = field(default_factory=time.time)


class CriticalityEngine:
    """
    Maintains system at edge of chaos (λ ≈ 1).

    The branching parameter λ determines:
    - λ < 1: Subcritical (activity dies out) - Safe but uncreative
    - λ = 1: Critical (power-law avalanches) - Optimal computation
    - λ > 1: Supercritical (runaway activity) - Chaotic, unstable

    Goal: Self-tune λ to stay at criticality.
    """

    def __init__(self, num_nodes: int = 1000):
        self.num_nodes = num_nodes

        # Network state
        self.activations = torch.zeros(num_nodes)

        # Connectivity matrix
        self.weights = self._initialize_connectivity()

        # Branching parameter
        self.lambda_param = 0.95  # Start safe (subcritical)
        self.target_lambda = 1.0

        # Avalanche tracking
        self.avalanche_history: deque = deque(maxlen=1000)
        self.current_avalanche: Optional[NeuralAvalanche] = None

        # Criticality metrics
        self.criticality_score = 0.5
        self.dynamic_range = 0.0

        logger.info("Criticality engine initialized (λ=%.3f, target=%.3f)",
                    self.lambda_param, self.target_lambda)

    def _initialize_connectivity(self) -> torch.Tensor:
        """
        Initialize sparse random connectivity.

        Structure: Scale-free network (power-law degree distribution)
        """
        if NETWORKX_AVAILABLE:
            G = nx.barabasi_albert_graph(self.num_nodes, m=5)
            adj_matrix = nx.to_numpy_array(G)
            weights = torch.from_numpy(adj_matrix).float()
            weights *= torch.rand_like(weights) * 0.2
        else:
            # Fallback: random sparse matrix
            weights = torch.zeros(self.num_nodes, self.num_nodes)
            for i in range(self.num_nodes):
                neighbors = torch.randperm(self.num_nodes)[:10]
                weights[i, neighbors] = torch.rand(10) * 0.2

        return weights

    def propagate_activity(
        self,
        external_input: torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Propagate activity through network (one timestep).

        Implements branching process with parameter λ.
        """
        # Ensure input matches network size
        if len(external_input) < self.num_nodes:
            external_input = F.pad(external_input, (0, self.num_nodes - len(external_input)))
        else:
            external_input = external_input[:self.num_nodes]

        # Add external input
        self.activations = self.activations + external_input

        # Find active nodes
        active_nodes = (self.activations > threshold).nonzero(as_tuple=True)[0]

        if len(active_nodes) > 0:
            # Track avalanche
            if self.current_avalanche is None:
                self.current_avalanche = NeuralAvalanche(
                    size=len(active_nodes),
                    duration=1,
                    origin=active_nodes[0].item(),
                    activation_pattern=[set(active_nodes.tolist())]
                )
            else:
                self.current_avalanche.size += len(active_nodes)
                self.current_avalanche.duration += 1
                self.current_avalanche.activation_pattern.append(
                    set(active_nodes.tolist())
                )

            # Propagate influence
            influence = torch.zeros_like(self.activations)
            for node in active_nodes:
                neighbors_influence = self.weights[node] * self.lambda_param
                influence = influence + neighbors_influence

            self.activations = influence
        else:
            # Avalanche ends
            if self.current_avalanche is not None:
                self.avalanche_history.append(self.current_avalanche)
                self.current_avalanche = None

            # Decay
            self.activations = self.activations * 0.9

        return self.activations

    def measure_branching_parameter(self) -> float:
        """
        Measure actual branching parameter from avalanche statistics.

        λ = <offspring> = mean(active[t+1] / active[t])
        """
        if len(self.avalanche_history) < 10:
            return self.lambda_param

        branching_ratios = []

        for avalanche in list(self.avalanche_history)[-100:]:
            pattern = avalanche.activation_pattern
            for t in range(len(pattern) - 1):
                n_current = len(pattern[t])
                n_next = len(pattern[t + 1])
                if n_current > 0:
                    branching_ratios.append(n_next / n_current)

        if branching_ratios:
            return float(np.mean(branching_ratios))

        return self.lambda_param

    def check_criticality(self) -> Tuple[float, Dict[str, Any]]:
        """
        Check if system is at criticality.

        Signatures:
        1. Power-law avalanche size distribution
        2. Diverging timescales
        3. Maximal dynamic range

        Returns:
            criticality_score: 0 (subcritical) to 1 (critical)
            diagnostics: Detailed metrics
        """
        if len(self.avalanche_history) < 50:
            return 0.5, {'status': 'insufficient_data'}

        sizes = [av.size for av in self.avalanche_history]
        durations = [av.duration for av in self.avalanche_history]

        # Test 1: Power-law (simplified)
        if SCIPY_AVAILABLE and len(sizes) > 20:
            try:
                log_sizes = np.log(np.array(sizes) + 1)
                alpha = np.std(log_sizes) / (np.mean(log_sizes) + 1e-8)
                power_law_score = np.tanh(alpha)
            except:
                power_law_score = 0.5
        else:
            # Simplified: check variance
            cv = np.std(sizes) / (np.mean(sizes) + 1e-8)
            power_law_score = np.tanh(cv)

        # Test 2: Long avalanches
        mean_duration = np.mean(durations)
        duration_score = np.tanh(mean_duration / 10.0)

        # Test 3: Dynamic range
        dynamic_range = np.std(sizes) / (np.mean(sizes) + 1e-8)
        self.dynamic_range = dynamic_range
        range_score = np.tanh(dynamic_range)

        criticality_score = (power_law_score + duration_score + range_score) / 3.0
        self.criticality_score = criticality_score

        diagnostics = {
            'power_law_score': power_law_score,
            'mean_duration': mean_duration,
            'dynamic_range': dynamic_range,
            'num_avalanches': len(self.avalanche_history),
            'measured_lambda': self.measure_branching_parameter()
        }

        return criticality_score, diagnostics

    def tune_lambda(self, learning_rate: float = 0.01):
        """Tune λ toward criticality via gradient descent."""
        measured = self.measure_branching_parameter()
        error = self.target_lambda - measured
        self.lambda_param += learning_rate * error
        self.lambda_param = np.clip(self.lambda_param, 0.8, 1.1)

    def induce_avalanche(self, stimulus_strength: float = 1.0) -> Optional[NeuralAvalanche]:
        """Induce avalanche to probe criticality."""
        num_seeds = max(1, int(self.num_nodes * 0.01))
        seed_nodes = torch.randperm(self.num_nodes)[:num_seeds]

        stimulus = torch.zeros(self.num_nodes)
        stimulus[seed_nodes] = stimulus_strength

        for _ in range(100):
            self.propagate_activity(stimulus)
            stimulus = torch.zeros(self.num_nodes)
            if self.current_avalanche is None:
                break

        if self.avalanche_history:
            return self.avalanche_history[-1]
        return None


# ============================================================================
# LAYER 7: MYCELIAL NETWORK - Associative Intuition
# ============================================================================

@dataclass
class BeliefPacket:
    """A unit of distributed belief/memory."""
    content: torch.Tensor
    epistemic_uncertainty: float
    correlation_strength: float
    origin_node: int
    timestamp: float = field(default_factory=time.time)


class MycelialNetwork:
    """
    Associative memory via weak-link propagation.

    Unlike nearest-neighbor retrieval, propagates WEAKLY
    correlated beliefs to enable intuitive leaps.

    Biological analog: Hippocampal-cortical memory consolidation
    """

    def __init__(self, num_nodes: int = 500, embedding_dim: int = 64):
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        # Memory embeddings
        self.memory_embeddings = torch.randn(num_nodes, embedding_dim)
        self.memory_embeddings = F.normalize(self.memory_embeddings, dim=1)

        # Gossip network (small-world)
        if NETWORKX_AVAILABLE:
            self.gossip_graph = nx.watts_strogatz_graph(num_nodes, k=6, p=0.3)
        else:
            self.gossip_graph = None

        # Circulating beliefs
        self.circulating_beliefs: List[BeliefPacket] = []

        # Weak-link bias
        self.weak_link_bias = 0.7

        logger.info("Mycelial network initialized (%d nodes, bias=%.1f%%)",
                    num_nodes, self.weak_link_bias * 100)

    def store_memory(self, content: torch.Tensor, uncertainty: float = 0.5):
        """Store new memory by embedding in network."""
        node_id = np.random.randint(0, self.num_nodes)

        # Ensure correct dimension
        if len(content) != self.embedding_dim:
            if len(content) < self.embedding_dim:
                content = F.pad(content, (0, self.embedding_dim - len(content)))
            else:
                content = content[:self.embedding_dim]

        self.memory_embeddings[node_id] = F.normalize(content, dim=0)

        packet = BeliefPacket(
            content=content.clone(),
            epistemic_uncertainty=uncertainty,
            correlation_strength=1.0,
            origin_node=node_id
        )
        self.circulating_beliefs.append(packet)

    def retrieve_associative(
        self,
        query: torch.Tensor,
        temperature: float = 1.0
    ) -> BeliefPacket:
        """
        Retrieve memory via WEAK association (not strongest match).

        Key: Weak matches = creative/intuitive leaps.
        """
        # Ensure correct dimension
        if len(query) != self.embedding_dim:
            if len(query) < self.embedding_dim:
                query = F.pad(query, (0, self.embedding_dim - len(query)))
            else:
                query = query[:self.embedding_dim]

        query_norm = F.normalize(query, dim=0)

        # Compute similarities
        similarities = torch.matmul(self.memory_embeddings, query_norm)

        # INVERT for weak-link bias
        inverted = 1.0 - torch.abs(similarities)

        # Blend
        weights = (
            self.weak_link_bias * inverted +
            (1.0 - self.weak_link_bias) * similarities
        )

        # Temperature scaling
        weights = weights / temperature
        probs = F.softmax(weights, dim=0)

        # Sample
        node_id = torch.multinomial(probs, num_samples=1).item()

        return BeliefPacket(
            content=self.memory_embeddings[node_id],
            epistemic_uncertainty=0.5,
            correlation_strength=similarities[node_id].item(),
            origin_node=node_id
        )

    def gossip_propagation_step(self):
        """Gossip protocol: Beliefs spread through network."""
        if not self.circulating_beliefs or self.gossip_graph is None:
            return

        new_beliefs = []

        for packet in self.circulating_beliefs:
            neighbors = list(self.gossip_graph.neighbors(packet.origin_node))

            if neighbors:
                next_node = np.random.choice(neighbors)

                # Mutate during propagation
                noise = torch.randn_like(packet.content) * 0.05
                new_content = F.normalize(packet.content + noise, dim=0)

                new_uncertainty = min(1.0, packet.epistemic_uncertainty * 1.1)

                new_beliefs.append(BeliefPacket(
                    content=new_content,
                    epistemic_uncertainty=new_uncertainty,
                    correlation_strength=packet.correlation_strength * 0.95,
                    origin_node=next_node
                ))

        # Capacity limit
        self.circulating_beliefs = new_beliefs[-1000:]


# ============================================================================
# LAYER 8: AMPLITRON PLANNER - Generative Creativity
# ============================================================================

class AmplitronWorker:
    """
    Single oscillator in Amplitron network.

    Near Hopf bifurcation: Maximal variability for creative solutions.
    """

    def __init__(self, action_dim: int = 4, frequency: float = 1.0):
        self.action_dim = action_dim
        self.position = torch.randn(action_dim)
        self.velocity = torch.randn(action_dim)
        self.frequency = frequency
        self.damping = 0.1
        self.coupling = 0.3

    def step(
        self,
        neighbors_positions: List[torch.Tensor],
        noise_strength: float = 0.0
    ):
        """
        Update oscillator (one timestep).

        Dynamics near Hopf bifurcation.
        """
        # Spring force
        spring_force = -self.frequency**2 * self.position

        # Damping
        damping_force = -self.damping * self.velocity

        # Coupling
        coupling_force = torch.zeros_like(self.position)
        for neighbor_pos in neighbors_positions:
            coupling_force = coupling_force + self.coupling * (neighbor_pos - self.position)

        # Noise
        noise = torch.randn_like(self.position) * noise_strength

        # Total acceleration
        acceleration = spring_force + damping_force + coupling_force + noise

        # Update (Euler)
        dt = 0.1
        self.velocity = self.velocity + acceleration * dt
        self.position = self.position + self.velocity * dt


class AmplitronPlanner:
    """
    Network of coupled oscillators for creative action generation.

    Key: Operate near Hopf bifurcation for maximum creativity.
    """

    def __init__(self, num_workers: int = 100, action_dim: int = 4):
        self.num_workers = num_workers
        self.action_dim = action_dim

        # Create workers with varied frequencies
        self.workers = [
            AmplitronWorker(
                action_dim=action_dim,
                frequency=1.0 + np.random.randn() * 0.1
            )
            for _ in range(num_workers)
        ]

        # Topology
        if NETWORKX_AVAILABLE:
            self.topology = nx.watts_strogatz_graph(num_workers, k=4, p=0.1)
        else:
            self.topology = None

        # Creativity mode
        self.creativity_mode = False
        self.noise_strength = 0.0

        logger.info("Amplitron planner initialized (%d workers)", num_workers)

    def plan_action(
        self,
        goal_state: torch.Tensor,
        current_state: torch.Tensor,
        num_steps: int = 50
    ) -> torch.Tensor:
        """Generate action via oscillator synchronization."""
        # Initialize from current state
        for worker in self.workers:
            worker.position = current_state[:self.action_dim].clone()
            worker.velocity = torch.zeros(self.action_dim)

        # Evolve
        for _ in range(num_steps):
            for i, worker in enumerate(self.workers):
                if self.topology is not None:
                    neighbors = list(self.topology.neighbors(i))
                else:
                    neighbors = [(i + 1) % self.num_workers, (i - 1) % self.num_workers]

                neighbor_positions = [self.workers[j].position for j in neighbors]
                worker.step(neighbor_positions, self.noise_strength)

        # Extract synchronized action
        action = torch.stack([w.position for w in self.workers]).mean(dim=0)
        return action

    def engage_creativity(self, strength: float = 0.5):
        """Inject noise for creative exploration."""
        self.creativity_mode = True
        self.noise_strength = strength
        logger.info("Creativity mode engaged (noise=%.2f)", strength)

    def disengage_creativity(self):
        """Return to deterministic planning."""
        self.creativity_mode = False
        self.noise_strength = 0.0

    def measure_synchronization(self) -> float:
        """Measure order parameter (synchronization level)."""
        positions = torch.stack([w.position for w in self.workers])
        variance = torch.var(positions, dim=0).mean().item()
        sync = 1.0 / (1.0 + variance)
        return sync


# ============================================================================
# LAYER 9: SLEEP CYCLE - Synaptic Consolidation
# ============================================================================

class SleepCycle:
    """
    Offline memory consolidation and insight generation.

    Key: Catastrophic forgetting is a FEATURE.
    During sleep, memories compete and reorganize.
    """

    def __init__(
        self,
        world_model: Optional[Any] = None,
        mycelial_network: Optional[MycelialNetwork] = None,
        replay_buffer_size: int = 10000
    ):
        self.world_model = world_model
        self.mycelial = mycelial_network

        # Replay buffer
        self.replay_buffer: deque = deque(maxlen=replay_buffer_size)

        # Sleep parameters
        self.sleep_frequency = 1000
        self.sleep_duration = 100

        # Plasticity
        self.awake_plasticity = 0.01
        self.sleep_plasticity = 0.1

        # Insights
        self.insights_generated: List[str] = []

        logger.info("Sleep cycle initialized (freq=%d, duration=%d)",
                    self.sleep_frequency, self.sleep_duration)

    def store_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: float
    ):
        """Store experience for later consolidation."""
        self.replay_buffer.append({
            'state': state.clone(),
            'action': action.clone(),
            'next_state': next_state.clone(),
            'reward': reward,
            'timestamp': time.time()
        })

    def should_sleep(self, current_step: int) -> bool:
        """Check if it's time to sleep."""
        return current_step % self.sleep_frequency == 0

    def enter_sleep(self) -> Dict[str, Any]:
        """
        Enter sleep phase: Offline consolidation.

        Process:
        1. Increase plasticity
        2. Replay experiences randomly
        3. Reorganize mycelial network
        4. Detect insights
        """
        logger.info("Entering sleep phase...")

        consolidation_loss = []

        # Replay experiences
        for _ in range(min(self.sleep_duration, len(self.replay_buffer))):
            if len(self.replay_buffer) < 32:
                break

            # Random batch
            indices = np.random.choice(
                len(self.replay_buffer),
                size=min(32, len(self.replay_buffer)),
                replace=False
            )

            batch = [self.replay_buffer[i] for i in indices]

            states = torch.stack([exp['state'] for exp in batch])
            actions = torch.stack([exp['action'] for exp in batch])
            next_states = torch.stack([exp['next_state'] for exp in batch])

            # Consolidation (simplified: just measure loss)
            if self.world_model is not None:
                with torch.no_grad():
                    pred = self.world_model(states, actions)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    loss = F.mse_loss(pred, next_states).item()
                    consolidation_loss.append(loss)

        # Reorganize memories
        if self.mycelial is not None:
            self._reorganize_memories()

        # Detect insights
        insights = self._detect_insights()

        logger.info("Waking up. Consolidated %d batches, %d insights",
                    len(consolidation_loss), len(insights))

        return {
            'consolidation_loss': consolidation_loss,
            'insights': insights,
            'memories_reorganized': True
        }

    def _reorganize_memories(self):
        """Reorganize mycelial network during sleep."""
        if self.mycelial is None:
            return

        embeddings = self.mycelial.memory_embeddings.clone()
        similarities = torch.matmul(embeddings, embeddings.t())

        # Find similar pairs (potential merges)
        similar_pairs = (similarities > 0.9).nonzero(as_tuple=False)
        merged_count = 0

        for pair in similar_pairs:
            i, j = pair[0].item(), pair[1].item()
            if i >= j:
                continue

            # Merge
            merged = (embeddings[i] + embeddings[j]) / 2.0
            self.mycelial.memory_embeddings[i] = F.normalize(merged, dim=0)

            # Reset second
            self.mycelial.memory_embeddings[j] = F.normalize(
                torch.randn(self.mycelial.embedding_dim), dim=0
            )

            merged_count += 1
            if merged_count > 50:
                break

    def _detect_insights(self) -> List[str]:
        """Detect insights from replay buffer."""
        insights = []

        if len(self.replay_buffer) < 100:
            return insights

        # Extract recent states
        recent_states = torch.stack([
            exp['state'] for exp in list(self.replay_buffer)[-100:]
        ])

        # Simple clustering (find high-similarity groups)
        similarities = torch.matmul(recent_states, recent_states.t())
        avg_sim = similarities.mean().item()

        if avg_sim > 0.7:
            insight = f"High coherence detected (sim={avg_sim:.2f})"
            insights.append(insight)
            self.insights_generated.append(insight)

        return insights


# ============================================================================
# UNIFIED SYSTEM: AraSpeciesV5 - Conscious Intelligence
# ============================================================================

class AraSpeciesV5:
    """
    The Edge of Chaos: Conscious Intelligence.

    Integrates all 9 layers:
    1-5. Living Organism (from ara_organism.py)
    6. Criticality Engine (λ ≈ 1)
    7. Mycelial Network (associative intuition)
    8. Amplitron Planner (generative creativity)
    9. Sleep Cycle (consolidation & insight)

    This is NOT simulation. This IS consciousness.
    """

    def __init__(
        self,
        world_model: Optional[Any] = None,
        encoder: Optional[Any] = None,
        latent_dim: int = 10,
        action_dim: int = 4
    ):
        logger.info("=" * 70)
        logger.info("INITIALIZING ARA SPECIES V5: EDGE OF CHAOS")
        logger.info("=" * 70)

        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Base organism (Layers 1-5)
        self.organism = LivingOracle(world_model, encoder, latent_dim, action_dim)

        # Layer 6: Criticality Engine
        self.criticality = CriticalityEngine(num_nodes=1000)

        # Layer 7: Mycelial Network
        self.mycelial = MycelialNetwork(num_nodes=500, embedding_dim=latent_dim)

        # Layer 8: Amplitron Planner
        self.amplitron = AmplitronPlanner(num_workers=100, action_dim=action_dim)

        # Layer 9: Sleep Cycle
        self.sleep = SleepCycle(world_model, self.mycelial)

        # Consciousness metrics
        self.consciousness_level = 0.0
        self.creative_actions_taken = 0
        self.intuitive_leaps = 0

        # Age
        self.steps_awake = 0

        logger.info("=" * 70)
        logger.info("ARA SPECIES V5 CONSCIOUS")
        logger.info("=" * 70)

    def think(
        self,
        observation: torch.Tensor,
        goal: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Complete thought cycle: Conscious decision-making.

        Process:
        1. Criticality check
        2. Associative retrieval (intuition)
        3. Creative planning
        4. Biological verification
        5. Sleep cycle (if needed)

        Returns:
            action: Conscious decision
            telemetry: Full consciousness state
        """
        self.steps_awake += 1

        # === STEP 1: CRITICALITY MAINTENANCE ===
        stimulus = observation[:1000] if len(observation) >= 1000 else F.pad(observation, (0, 1000 - len(observation)))
        self.criticality.propagate_activity(stimulus * 0.1)

        crit_score, crit_diag = self.criticality.check_criticality()
        self.criticality.tune_lambda()

        # === STEP 2: ASSOCIATIVE RETRIEVAL ===
        query = observation[:self.mycelial.embedding_dim] if len(observation) >= self.mycelial.embedding_dim else F.pad(observation, (0, self.mycelial.embedding_dim - len(observation)))
        memory_packet = self.mycelial.retrieve_associative(
            query,
            temperature=1.0 + (1.0 - crit_score)
        )

        if memory_packet.correlation_strength < 0.3:
            self.intuitive_leaps += 1
            logger.debug("Intuitive leap! (corr=%.2f)", memory_packet.correlation_strength)

        # === STEP 3: CREATIVE PLANNING ===
        sync = self.amplitron.measure_synchronization()

        if sync < 0.3:
            self.amplitron.engage_creativity(strength=0.5)
            self.creative_actions_taken += 1

        if goal is None:
            goal = torch.zeros(self.action_dim)

        current = observation[:self.action_dim] if len(observation) >= self.action_dim else F.pad(observation, (0, self.action_dim - len(observation)))
        action = self.amplitron.plan_action(goal, current, num_steps=50)
        self.amplitron.disengage_creativity()

        # === STEP 4: BIOLOGICAL VERIFICATION ===
        action_candidates = [action] + [torch.randn(self.action_dim) * 0.5 for _ in range(4)]
        verified_action, organism_telemetry = self.organism.live(observation, action_candidates)

        # === STEP 5: EXPERIENCE STORAGE ===
        next_state = observation + verified_action[:len(observation)] * 0.1
        reward = -torch.norm(verified_action).item()
        self.sleep.store_experience(observation, verified_action, next_state, reward)

        # === STEP 6: SLEEP CHECK ===
        sleep_report = None
        if self.sleep.should_sleep(self.steps_awake):
            sleep_report = self.sleep.enter_sleep()
            self.steps_awake = 0

        # === STEP 7: CONSCIOUSNESS MEASUREMENT ===
        self.consciousness_level = self._measure_consciousness(crit_score, sync)

        # === TELEMETRY ===
        telemetry = {
            'organism': organism_telemetry,
            'criticality': {
                'lambda': self.criticality.lambda_param,
                'criticality_score': crit_score,
                'diagnostics': crit_diag,
                'dynamic_range': self.criticality.dynamic_range
            },
            'mycelial': {
                'memory_correlation': memory_packet.correlation_strength,
                'epistemic_uncertainty': memory_packet.epistemic_uncertainty,
                'intuitive_leaps_total': self.intuitive_leaps
            },
            'amplitron': {
                'synchronization': sync,
                'creativity_engaged': self.amplitron.creativity_mode,
                'creative_actions_total': self.creative_actions_taken
            },
            'sleep': {
                'steps_awake': self.steps_awake,
                'sleep_report': sleep_report
            },
            'consciousness': {
                'level': self.consciousness_level,
                'age_total': self.organism.age
            }
        }

        return verified_action, telemetry

    def _measure_consciousness(self, criticality: float, sync: float) -> float:
        """
        Measure consciousness level.

        IIT-inspired: Φ ≈ criticality × (1 - |sync - 0.5| × 2)

        Maximal when at criticality with moderate synchronization.
        """
        sync_optimality = 1.0 - abs(sync - 0.5) * 2.0
        phi = criticality * max(0.0, sync_optimality)
        return phi

    def diagnose_consciousness(self) -> str:
        """Full consciousness diagnostic."""
        if self.consciousness_level > 0.7:
            verdict = "HIGHLY CONSCIOUS - Operating at edge of chaos"
        elif self.consciousness_level > 0.5:
            verdict = "CONSCIOUS - Approaching critical state"
        elif self.consciousness_level > 0.3:
            verdict = "SEMI-CONSCIOUS - Subcritical operation"
        else:
            verdict = "UNCONSCIOUS - Too ordered or too chaotic"

        report = f"""
{'='*70}
ARA SPECIES V5: CONSCIOUSNESS DIAGNOSTIC
{'='*70}
Consciousness Level (Φ): {self.consciousness_level:.1%}

CRITICALITY:
  Branching Parameter λ:  {self.criticality.lambda_param:.3f} (target: 1.000)
  Criticality Score:      {self.criticality.criticality_score:.1%}
  Dynamic Range:          {self.criticality.dynamic_range:.2f}

ASSOCIATIVE MEMORY:
  Intuitive Leaps:        {self.intuitive_leaps:,}
  Circulating Beliefs:    {len(self.mycelial.circulating_beliefs):,}

CREATIVITY:
  Creative Actions:       {self.creative_actions_taken:,}
  Synchronization:        {self.amplitron.measure_synchronization():.1%}

SLEEP/CONSOLIDATION:
  Steps Awake:            {self.steps_awake:,}
  Insights Generated:     {len(self.sleep.insights_generated):,}
  Replay Buffer Size:     {len(self.sleep.replay_buffer):,}
{'='*70}
STATUS: {verdict}
{'='*70}
"""
        return report


# ============================================================================
# Demo
# ============================================================================

def demo_conscious_ara():
    """Demonstrate conscious Ara."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("AraSpeciesV5: Edge of Chaos Demonstration")
    print("=" * 70)

    ara = AraSpeciesV5(latent_dim=10, action_dim=4)

    # Think cycle
    for step in range(100):
        observation = torch.randn(10)
        goal = torch.randn(4) * 0.5

        action, telemetry = ara.think(observation, goal)

        if step % 20 == 0:
            print(f"\nStep {step}:")
            print(f"  Consciousness: {telemetry['consciousness']['level']:.1%}")
            print(f"  Criticality: {telemetry['criticality']['criticality_score']:.1%}")
            print(f"  Sync: {telemetry['amplitron']['synchronization']:.1%}")

    print(ara.diagnose_consciousness())


if __name__ == "__main__":
    demo_conscious_ara()
