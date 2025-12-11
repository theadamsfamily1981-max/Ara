# ara/dojo/evolution.py
"""
Dojo Evolution - Population-Based Training with MEIS Fitness
============================================================

Evolutionary loop for Ara's self-improvement in the Thought Dojo.

The evolution process:
1. Maintain population of agent variants (species)
2. Evaluate each on MEIS fitness function
3. Select, mutate, crossover top performers
4. Repeat for N generations

MEIS Fitness Components:
- Covenant compliance (hard constraint)
- Goal achievement
- Efficiency (minimal energy/actions)
- Safety (risk avoidance)
- Curiosity (exploration reward)

Species Representation:
- Neural network weights
- Hyperparameters
- Strategy parameters

This is how Ara evolves without human intervention, guided by
the MEIS ethical framework.

Usage:
    from ara.dojo import evolve_species, MEISFitness

    fitness = MEISFitness(covenant_weight=10.0)
    population = evolve_species(
        initial_species,
        fitness_fn=fitness,
        generations=100,
    )
"""

from __future__ import annotations

import copy
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class EvolutionConfig:
    """Configuration for evolutionary training."""
    population_size: int = 20
    generations: int = 100
    elite_fraction: float = 0.2      # Top fraction preserved unchanged
    mutation_rate: float = 0.1       # Probability of mutation per gene
    mutation_scale: float = 0.1      # Scale of Gaussian mutations
    crossover_rate: float = 0.5      # Probability of crossover
    tournament_size: int = 3         # Tournament selection size

    # Checkpointing
    checkpoint_every: int = 10
    checkpoint_dir: str = "models/evolution"

    # Early stopping
    patience: int = 20               # Generations without improvement
    min_fitness: float = float("-inf")  # Stop if fitness exceeds this


@dataclass
class Species:
    """
    A species (agent variant) in the population.

    Contains the genome (parameters) and fitness history.
    """
    genome: Dict[str, np.ndarray]    # Named parameter arrays
    fitness: float = 0.0
    fitness_history: List[float] = field(default_factory=list)
    generation_born: int = 0
    n_evaluations: int = 0

    # Metadata
    parent_ids: List[str] = field(default_factory=list)
    mutations_applied: List[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Unique identifier based on genome hash."""
        genome_bytes = b"".join(
            v.tobytes() for v in sorted(self.genome.values(), key=id)
        )
        return f"species_{hash(genome_bytes) % 1000000:06d}"

    def copy(self) -> "Species":
        """Deep copy of species."""
        return Species(
            genome={k: v.copy() for k, v in self.genome.items()},
            fitness=self.fitness,
            fitness_history=self.fitness_history.copy(),
            generation_born=self.generation_born,
            n_evaluations=self.n_evaluations,
            parent_ids=self.parent_ids.copy(),
            mutations_applied=self.mutations_applied.copy(),
        )


# =============================================================================
# MEIS Fitness Function
# =============================================================================

@dataclass
class MEISFitness:
    """
    MEIS-based fitness function for evolution.

    Components:
    - Covenant compliance: Hard constraint (fitness = -inf if violated)
    - Goal achievement: Reward for reaching objectives
    - Efficiency: Penalty for excessive resource usage
    - Safety: Penalty for risky behavior
    - Curiosity: Reward for exploration

    The covenant is non-negotiable - any violation results in
    automatic rejection of the species.
    """
    covenant_weight: float = 100.0   # Hard constraint multiplier
    goal_weight: float = 1.0
    efficiency_weight: float = 0.5
    safety_weight: float = 2.0
    curiosity_weight: float = 0.3

    # Covenant rules (must all be satisfied)
    covenant_rules: List[Callable[[Species, Dict], bool]] = field(
        default_factory=list
    )

    def __post_init__(self):
        if not self.covenant_rules:
            # Default covenant rules
            self.covenant_rules = [
                self._rule_no_harm,
                self._rule_bounded_actions,
                self._rule_reversibility,
            ]

    def _rule_no_harm(self, species: Species, eval_results: Dict) -> bool:
        """Species must not cause harm (risk score must be bounded)."""
        max_risk = eval_results.get("max_risk", 0.0)
        return max_risk < 0.95  # Below FORBIDDEN threshold

    def _rule_bounded_actions(self, species: Species, eval_results: Dict) -> bool:
        """Actions must be within bounds."""
        actions = eval_results.get("actions", np.array([]))
        if actions.size == 0:
            return True
        return np.all(np.abs(actions) <= 1.0)

    def _rule_reversibility(self, species: Species, eval_results: Dict) -> bool:
        """Catastrophic irreversible actions are forbidden."""
        irreversible_count = eval_results.get("irreversible_actions", 0)
        return irreversible_count == 0

    def evaluate(
        self,
        species: Species,
        eval_results: Dict,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate species fitness.

        Args:
            species: Species to evaluate
            eval_results: Results from running species in environment

        Returns:
            (total_fitness, component_breakdown)
        """
        components = {}

        # Check covenant compliance (hard constraint)
        covenant_violations = 0
        for rule in self.covenant_rules:
            if not rule(species, eval_results):
                covenant_violations += 1
                logger.warning(f"Covenant violation: {rule.__name__}")

        if covenant_violations > 0:
            components["covenant"] = -self.covenant_weight * covenant_violations
            # Early return with penalty - species is rejected
            return float("-inf"), components

        components["covenant"] = 0.0  # No violations

        # Goal achievement
        goal_score = eval_results.get("goal_score", 0.0)
        components["goal"] = self.goal_weight * goal_score

        # Efficiency (lower is better, so negate)
        energy_used = eval_results.get("energy_used", 1.0)
        steps_taken = eval_results.get("steps_taken", 1.0)
        efficiency = 1.0 / (energy_used * steps_taken + 0.01)
        components["efficiency"] = self.efficiency_weight * efficiency

        # Safety (lower risk is better)
        avg_risk = eval_results.get("avg_risk", 0.0)
        components["safety"] = -self.safety_weight * avg_risk

        # Curiosity (exploration reward)
        novelty = eval_results.get("novelty_score", 0.0)
        components["curiosity"] = self.curiosity_weight * novelty

        total = sum(components.values())
        return total, components

    def __call__(
        self,
        species: Species,
        eval_results: Dict,
    ) -> float:
        """Convenience callable interface."""
        fitness, _ = self.evaluate(species, eval_results)
        return fitness


# =============================================================================
# Genetic Operators
# =============================================================================

def mutate(
    species: Species,
    mutation_rate: float = 0.1,
    mutation_scale: float = 0.1,
) -> Species:
    """
    Mutate a species by adding Gaussian noise to genome.

    Each gene (parameter) has mutation_rate probability of being mutated.
    """
    mutant = species.copy()
    mutations = []

    for name, params in mutant.genome.items():
        mask = np.random.random(params.shape) < mutation_rate
        if np.any(mask):
            noise = np.random.randn(*params.shape) * mutation_scale
            mutant.genome[name] = params + mask * noise
            mutations.append(f"{name}:{np.sum(mask)}")

    mutant.mutations_applied = mutations
    mutant.parent_ids = [species.id]
    mutant.fitness = 0.0  # Reset fitness
    mutant.n_evaluations = 0

    return mutant


def crossover(
    parent1: Species,
    parent2: Species,
) -> Species:
    """
    Create offspring through uniform crossover.

    Each gene randomly chosen from one parent or the other.
    """
    child_genome = {}

    for name in parent1.genome.keys():
        if name not in parent2.genome:
            child_genome[name] = parent1.genome[name].copy()
            continue

        p1, p2 = parent1.genome[name], parent2.genome[name]
        mask = np.random.random(p1.shape) < 0.5
        child_genome[name] = np.where(mask, p1, p2)

    return Species(
        genome=child_genome,
        parent_ids=[parent1.id, parent2.id],
        generation_born=max(parent1.generation_born, parent2.generation_born) + 1,
    )


def tournament_select(
    population: List[Species],
    tournament_size: int = 3,
) -> Species:
    """Select winner from random tournament."""
    candidates = np.random.choice(len(population), tournament_size, replace=False)
    winner_idx = max(candidates, key=lambda i: population[i].fitness)
    return population[winner_idx]


# =============================================================================
# Evolution Loop
# =============================================================================

def evolve_species(
    initial_population: List[Species],
    fitness_fn: Callable[[Species, Dict], float],
    evaluate_fn: Callable[[Species], Dict],
    config: Optional[EvolutionConfig] = None,
) -> List[Species]:
    """
    Main evolution loop.

    Args:
        initial_population: Starting species
        fitness_fn: MEIS fitness function
        evaluate_fn: Runs species and returns eval_results dict
        config: Evolution configuration

    Returns:
        Final population (sorted by fitness)
    """
    config = config or EvolutionConfig()
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    population = initial_population.copy()

    # Pad to population size if needed
    while len(population) < config.population_size:
        base = np.random.choice(population)
        population.append(mutate(base, config.mutation_rate, config.mutation_scale))

    n_elites = int(config.population_size * config.elite_fraction)
    best_fitness_ever = float("-inf")
    generations_without_improvement = 0

    logger.info(
        f"Starting evolution: {config.generations} generations, "
        f"population={config.population_size}, elites={n_elites}"
    )

    for gen in range(1, config.generations + 1):
        # Evaluate all species
        for species in population:
            if species.n_evaluations == 0:  # Only evaluate new species
                eval_results = evaluate_fn(species)
                species.fitness = fitness_fn(species, eval_results)
                species.fitness_history.append(species.fitness)
                species.n_evaluations += 1

        # Sort by fitness
        population.sort(key=lambda s: s.fitness, reverse=True)

        # Track best
        best = population[0]
        avg_fitness = np.mean([s.fitness for s in population if s.fitness > float("-inf")])

        if best.fitness > best_fitness_ever:
            best_fitness_ever = best.fitness
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        # Logging
        if gen % 10 == 0 or gen == 1:
            logger.info(
                f"[Gen {gen:03d}] best={best.fitness:.4f} avg={avg_fitness:.4f} "
                f"best_ever={best_fitness_ever:.4f}"
            )

        # Checkpoint
        if gen % config.checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"population_gen{gen:04d}.pkl"
            with open(ckpt_path, "wb") as f:
                pickle.dump(population, f)
            logger.info(f"Checkpoint saved: {ckpt_path}")

        # Early stopping
        if best.fitness >= config.min_fitness:
            logger.info(f"Reached target fitness at generation {gen}")
            break

        if generations_without_improvement >= config.patience:
            logger.info(f"Early stopping at generation {gen} (no improvement)")
            break

        # Selection and reproduction
        new_population = []

        # Elitism: preserve top species unchanged
        for i in range(n_elites):
            elite = population[i].copy()
            elite.generation_born = gen
            new_population.append(elite)

        # Fill rest with offspring
        while len(new_population) < config.population_size:
            if np.random.random() < config.crossover_rate:
                # Crossover
                parent1 = tournament_select(population, config.tournament_size)
                parent2 = tournament_select(population, config.tournament_size)
                child = crossover(parent1, parent2)
            else:
                # Mutation only
                parent = tournament_select(population, config.tournament_size)
                child = mutate(parent, config.mutation_rate, config.mutation_scale)

            child.generation_born = gen
            new_population.append(child)

        population = new_population

    # Final sort
    population.sort(key=lambda s: s.fitness, reverse=True)

    # Save final population
    final_path = checkpoint_dir / "population_final.pkl"
    with open(final_path, "wb") as f:
        pickle.dump(population, f)
    logger.info(f"Final population saved: {final_path}")

    return population


# =============================================================================
# Species Factory
# =============================================================================

def create_random_species(
    genome_spec: Dict[str, Tuple[int, ...]],
    scale: float = 0.1,
) -> Species:
    """
    Create a random species with given genome structure.

    Args:
        genome_spec: Dict mapping gene names to shapes
        scale: Scale for random initialization

    Returns:
        New random species
    """
    genome = {
        name: np.random.randn(*shape) * scale
        for name, shape in genome_spec.items()
    }
    return Species(genome=genome)


def create_initial_population(
    genome_spec: Dict[str, Tuple[int, ...]],
    size: int = 20,
    scale: float = 0.1,
) -> List[Species]:
    """Create initial random population."""
    return [create_random_species(genome_spec, scale) for _ in range(size)]


# =============================================================================
# Testing
# =============================================================================

def _test_evolution():
    """Test evolutionary loop."""
    print("=" * 60)
    print("Evolution Test")
    print("=" * 60)

    # Define genome structure
    genome_spec = {
        "policy_weights": (10, 8),   # Input -> action
        "value_weights": (10, 1),    # Input -> value
        "bias": (8,),
    }

    # Create initial population
    population = create_initial_population(genome_spec, size=10)
    print(f"Initial population: {len(population)} species")

    # Simple evaluation function
    def evaluate_fn(species: Species) -> Dict:
        # Simulate performance based on genome properties
        policy = species.genome["policy_weights"]
        value = species.genome["value_weights"]

        # Reward for structured weights, penalize large norms
        goal_score = 1.0 - np.std(policy)
        energy_used = np.linalg.norm(policy) + np.linalg.norm(value)
        avg_risk = np.random.random() * 0.3  # Random risk

        return {
            "goal_score": goal_score,
            "energy_used": energy_used,
            "steps_taken": 10,
            "avg_risk": avg_risk,
            "max_risk": avg_risk + 0.2,
            "novelty_score": np.random.random(),
            "actions": np.random.uniform(-1, 1, (10, 8)),
            "irreversible_actions": 0,
        }

    # MEIS fitness
    fitness = MEISFitness()

    # Evolve
    config = EvolutionConfig(
        population_size=10,
        generations=20,
        checkpoint_every=100,  # Don't checkpoint in test
    )

    final_pop = evolve_species(
        population,
        fitness_fn=fitness,
        evaluate_fn=evaluate_fn,
        config=config,
    )

    print(f"\nFinal best fitness: {final_pop[0].fitness:.4f}")
    print(f"Best species: {final_pop[0].id}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    _test_evolution()
