#!/usr/bin/env python3
# ara/dojo/arena_core.py
"""
Optimized Arena Core for Thought Dojo / Ara
============================================

Goals:
- Maximize Ara cycles/second on Threadripper 5955WX
- Exploit batched HDC → VAE latent projection
- Keep encoder hot in cache (shared across agents)
- Keep per-core execution deterministic & isolated

This is the high-performance evaluation loop that:
1. Collects observations from vectorized environments
2. Batch-encodes HDC → latent via shared VAE
3. Lets each agent plan/imagine in latent space
4. Steps the arena and accumulates fitness

Interfaces are defined as Protocols so you can adapt your existing classes
(AraSpeciesV2, Arena, HDC encoder) without hard coupling.

Usage:
    from ara.dojo import OptimizedArenaCore, ArenaCoreConfig

    core = OptimizedArenaCore(
        config=ArenaCoreConfig(core_id=4, batch_size=256),
        latent_encoder=vae_encoder,
    )

    fitnesses = core.evaluate_population(population, arena, episodes=5)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np

logger = logging.getLogger(__name__)

# Optional torch import
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# =============================================================================
# Protocols (Interfaces)
# =============================================================================

@runtime_checkable
class HDCEncoder(Protocol):
    """
    Optional hardware/software HDC encoder.

    If your Arena already returns obs["hdc_vector"], you don't need this.

    encode_batch:
        obs_batch: list[obs_dict]
        returns:   Tensor [batch, h_dim] (float32 or binary mapped to {0,1})
    """

    def encode_batch(self, obs_batch: Sequence[Dict[str, Any]]) -> Any:
        """Encode a batch of observations to hypervectors."""
        ...


@runtime_checkable
class LatentEncoder(Protocol):
    """
    VAE-like encoder: maps HDC hypervectors → latent z.

    Expected shapes:
        input:  [batch, h_dim]
        output: either z                [batch, latent_dim]
                or (z, mu, log_var, ..)

    We handle both cases; the first element is assumed to be z if a tuple.
    """

    def encode(self, h_batch: Any) -> Any:
        """Encode hypervectors to latent space."""
        ...


@runtime_checkable
class PlannerAgent(Protocol):
    """
    Minimal interface for an Ara species compatible with this core.

    You extend your AraSpecies to implement:

        def select_action_from_latent(self, z_t: Tensor) -> Tensor

    where z_t is [latent_dim], and return an action tensor [action_dim].
    """

    def select_action_from_latent(self, z_t: Any) -> Any:
        """Select action given current latent state."""
        ...


@runtime_checkable
class VecArena(Protocol):
    """
    Vectorized Arena interface (one env per agent).

    reset(population) → observations: list[obs]
    step(actions)     → (next_obs, rewards, dones, infos)

    Each obs dict SHOULD contain:
        - "hdc_vector": np.ndarray / torch.Tensor [h_dim]
      OR
        - raw sensor fields that your HDCEncoder knows how to consume.
    """

    def reset(self, population: Sequence[PlannerAgent]) -> List[Dict[str, Any]]:
        """Reset environments for population."""
        ...

    def step(
        self,
        actions: Sequence[np.ndarray],
    ) -> Tuple[
        List[Dict[str, Any]],  # next_obs
        np.ndarray,            # rewards [num_envs]
        np.ndarray,            # dones   [num_envs] bool
        List[Dict[str, Any]],  # infos   [num_envs]
    ]:
        """Step environments with actions."""
        ...


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ArenaCoreConfig:
    """
    Configuration for OptimizedArenaCore.

    Attributes:
        core_id: Physical CPU core to pin this process to (None to skip pinning)
        batch_size: Number of agents to encode per VAE call (128-512 recommended)
        device: "cpu" (recommended for Threadripper) or "cuda"
        use_no_grad: Wrap inference in torch.no_grad() to avoid autograd overhead
        hdc_dim: HDC dimensionality for sanity checking (10_000 default)
        max_steps_per_episode: Safety limit on episode length
        log_every_n_steps: Log progress every N steps (0 to disable)
    """
    core_id: Optional[int] = None
    batch_size: int = 256
    device: str = "cpu"
    use_no_grad: bool = True
    hdc_dim: Optional[int] = 10_000
    max_steps_per_episode: int = 1000
    log_every_n_steps: int = 100


@dataclass
class EvaluationResult:
    """Result of population evaluation."""
    fitnesses: List[float]
    total_steps: int
    episodes_completed: int
    steps_per_second: float = 0.0
    per_agent_stats: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Optimized Arena Core
# =============================================================================

class OptimizedArenaCore:
    """
    High-performance control loop for a population of Ara-like agents.

    Responsibilities:
    - Build batched HDC tensor from env observations
    - Run batched VAE encoder → latent z_t
    - Call each agent's imagination/planner in latent space
    - Step the VecArena, accumulate fitness

    This class does NOT handle evolution/mutation; it just evaluates.
    """

    def __init__(
        self,
        config: ArenaCoreConfig,
        latent_encoder: LatentEncoder,
        hdc_encoder: Optional[HDCEncoder] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for OptimizedArenaCore")

        self.config = config
        self.latent_encoder = latent_encoder
        self.hdc_encoder = hdc_encoder

        self.device = torch.device(config.device)

        # Put encoder on the right device and eval mode
        if isinstance(self.latent_encoder, nn.Module):
            self.latent_encoder.to(self.device)
            self.latent_encoder.eval()

        # Pin this process to a core (for cache locality & deterministic latency)
        if self.config.core_id is not None:
            self._pin_to_core(self.config.core_id)

        # Torch threading: 1 per core for many-core setups
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        logger.info(
            f"OptimizedArenaCore initialized: "
            f"device={config.device}, batch_size={config.batch_size}, "
            f"core_id={config.core_id}"
        )

    def _pin_to_core(self, core_id: int) -> None:
        """Pin this process to a specific CPU core."""
        try:
            os.sched_setaffinity(0, {core_id})
            logger.info(f"Pinned to core {core_id}")
        except AttributeError:
            # Not available on all platforms (e.g., Windows)
            logger.warning("CPU pinning not available on this platform")
        except OSError as e:
            logger.warning(f"Could not pin to core {core_id}: {e}")

    # =========================================================================
    # Public API
    # =========================================================================

    def evaluate_population(
        self,
        population: Sequence[PlannerAgent],
        arena: VecArena,
        episodes: int = 1,
        on_step_hook: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a population of agents in the given VecArena.

        Args:
            population: list of agents (one per environment)
            arena: vectorized environment implementation
            episodes: number of episodes to run per agent
            on_step_hook: optional callback per step: fn(step_idx, context_dict)

        Returns:
            EvaluationResult with fitnesses and statistics
        """
        import time

        num_agents = len(population)
        fitness = np.zeros(num_agents, dtype=np.float64)
        per_agent_stats = [{"rewards": [], "steps": 0} for _ in range(num_agents)]

        total_steps = 0
        start_time = time.perf_counter()

        for ep in range(episodes):
            obs_batch = arena.reset(population)
            done_flags = np.zeros(num_agents, dtype=bool)
            step_idx = 0

            while not done_flags.all() and step_idx < self.config.max_steps_per_episode:
                # 1) Build HDC batch: [batch, h_dim]
                hdc_batch = self._extract_hdc_batch(obs_batch)

                # 2) Batched latent projection via VAE encoder
                z_batch = self._encode_latent_batch(hdc_batch)

                # 3) Per-agent imagination/planning in latent space
                actions: List[np.ndarray] = []
                for i, agent in enumerate(population):
                    if done_flags[i]:
                        # Env finished – action ignored, but keep placeholder shape
                        actions.append(np.zeros(1, dtype=np.float32))
                        continue

                    z_i = z_batch[i]  # [latent_dim]
                    action_tensor = agent.select_action_from_latent(z_i)
                    actions.append(self._to_numpy(action_tensor))

                # 4) Step arena
                next_obs_batch, rewards, dones, infos = arena.step(actions)

                # 5) Update fitness + done flags
                for i in range(num_agents):
                    if not done_flags[i]:
                        fitness[i] += rewards[i]
                        per_agent_stats[i]["rewards"].append(float(rewards[i]))
                        per_agent_stats[i]["steps"] += 1

                done_flags |= dones.astype(bool)
                obs_batch = next_obs_batch

                if on_step_hook is not None:
                    on_step_hook(
                        step_idx,
                        {
                            "episode": ep,
                            "actions": actions,
                            "rewards": rewards,
                            "dones": done_flags.copy(),
                            "infos": infos,
                        },
                    )

                step_idx += 1
                total_steps += num_agents

                if (
                    self.config.log_every_n_steps > 0
                    and step_idx % self.config.log_every_n_steps == 0
                ):
                    active = (~done_flags).sum()
                    logger.debug(f"Episode {ep}, step {step_idx}: {active} active")

        elapsed = time.perf_counter() - start_time
        steps_per_second = total_steps / elapsed if elapsed > 0 else 0.0

        logger.info(
            f"Evaluation complete: {total_steps} steps in {elapsed:.2f}s "
            f"({steps_per_second:.0f} steps/s)"
        )

        return EvaluationResult(
            fitnesses=fitness.tolist(),
            total_steps=total_steps,
            episodes_completed=episodes * num_agents,
            steps_per_second=steps_per_second,
            per_agent_stats=per_agent_stats,
        )

    def evaluate_single(
        self,
        agent: PlannerAgent,
        arena: VecArena,
        episodes: int = 1,
    ) -> float:
        """Evaluate a single agent (convenience wrapper)."""
        result = self.evaluate_population([agent], arena, episodes)
        return result.fitnesses[0]

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _extract_hdc_batch(
        self,
        obs_batch: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        """
        Build a [batch, h_dim] HDC tensor from observations.

        Priority:
            1. If obs["hdc_vector"] exists, use it directly.
            2. Else, if hdc_encoder is provided, call encode_batch(obs_batch).
        """
        # Case 1: Arena already gives pre-encoded HDC vectors
        if obs_batch and "hdc_vector" in obs_batch[0]:
            hv_list: List[np.ndarray] = []
            for obs in obs_batch:
                h = obs["hdc_vector"]
                if isinstance(h, torch.Tensor):
                    h = h.cpu().numpy()
                hv_list.append(np.asarray(h, dtype=np.float32))

            hdc = np.stack(hv_list, axis=0)  # [batch, h_dim]

        # Case 2: Use HDCEncoder to map raw obs → hypervector
        elif self.hdc_encoder is not None:
            hdc_tensor = self.hdc_encoder.encode_batch(obs_batch)
            if isinstance(hdc_tensor, torch.Tensor):
                hdc = hdc_tensor.cpu().numpy()
            else:
                hdc = np.asarray(hdc_tensor, dtype=np.float32)

        else:
            raise ValueError(
                "Observations do not contain 'hdc_vector' and no HDCEncoder was provided.\n"
                "Either adapt Arena to provide obs['hdc_vector'] or pass an HDCEncoder."
            )

        if self.config.hdc_dim is not None and hdc.shape[1] != self.config.hdc_dim:
            raise ValueError(
                f"HDC dim mismatch: expected {self.config.hdc_dim}, got {hdc.shape[1]}"
            )

        return torch.from_numpy(hdc).to(self.device, non_blocking=True)

    def _encode_latent_batch(self, hdc_batch: torch.Tensor) -> torch.Tensor:
        """Batched call into the VAE encoder, with chunking if needed."""
        if self.config.use_no_grad:
            ctx = torch.no_grad()
        else:
            class _NullCtx:
                def __enter__(self):
                    return None

                def __exit__(self, *exc):
                    return False

            ctx = _NullCtx()

        with ctx:
            batch_size = hdc_batch.size(0)
            if batch_size <= self.config.batch_size:
                z_full = self._encode_latent_chunk(hdc_batch)
            else:
                chunks = []
                for start in range(0, batch_size, self.config.batch_size):
                    end = min(start + self.config.batch_size, batch_size)
                    h_chunk = hdc_batch[start:end]
                    z_chunk = self._encode_latent_chunk(h_chunk)
                    chunks.append(z_chunk)
                z_full = torch.cat(chunks, dim=0)

        return z_full  # [batch, latent_dim]

    def _encode_latent_chunk(self, h_chunk: torch.Tensor) -> torch.Tensor:
        """
        Encode a single chunk of HDC → latent.

        Handles encoder.encode(...) returning:
            - z
            - (z, mu, log_var, ...)
        """
        out = self.latent_encoder.encode(h_chunk)

        if isinstance(out, torch.Tensor):
            z = out
        elif isinstance(out, tuple):
            # Assume first element of tuple-like output is z
            z = out[0]
        else:
            z = out

        if z.dim() == 1:
            z = z.unsqueeze(0)

        return z

    @staticmethod
    def _to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert a torch tensor to a contiguous float32 numpy array."""
        if isinstance(x, np.ndarray):
            return x.astype(np.float32, copy=False)

        if x.is_cuda:
            x = x.cpu()
        return x.detach().numpy().astype(np.float32, copy=False)


# =============================================================================
# Ara Species Base Class
# =============================================================================

class AraSpeciesBase:
    """
    Base class for Ara species that work with OptimizedArenaCore.

    Subclass this and implement select_action_from_latent().

    Example:
        class MyAra(AraSpeciesBase):
            def __init__(self, world_model, planner):
                self.world_model = world_model
                self.planner = planner

            def select_action_from_latent(self, z_t):
                action, _ = self.planner.plan(z_t)
                return action
    """

    def select_action_from_latent(
        self,
        z_t: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Select action given current latent state.

        Args:
            z_t: Current latent state [latent_dim]
            goal: Optional goal state [latent_dim]

        Returns:
            Action tensor [action_dim]
        """
        raise NotImplementedError("Subclass must implement select_action_from_latent")


class AraSpeciesV2(AraSpeciesBase):
    """
    Ara species with world model and imagination planner.

    This is the default implementation that uses:
    - World model for dynamics prediction
    - MPC planner for action selection via imagination
    """

    def __init__(
        self,
        world_model: Any,
        planner: Any,
        default_goal: Optional[np.ndarray] = None,
    ):
        """
        Initialize Ara species.

        Args:
            world_model: Dynamics model for z' = f(z, u)
            planner: MPC planner (e.g., DojoPlanner)
            default_goal: Default goal in latent space
        """
        self.world_model = world_model
        self.planner = planner
        self.default_goal = default_goal

        # Stats
        self.total_plans = 0
        self.total_imagination_steps = 0

    def select_action_from_latent(
        self,
        z_t: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Select action via imagination/planning in latent space.

        Called by OptimizedArenaCore with a precomputed latent state.
        """
        # Convert to numpy for planner
        if isinstance(z_t, torch.Tensor):
            z_np = z_t.detach().cpu().numpy()
        else:
            z_np = np.asarray(z_t)

        # Use default goal if none provided
        if goal is None and self.default_goal is not None:
            goal_np = self.default_goal
        elif goal is not None:
            if isinstance(goal, torch.Tensor):
                goal_np = goal.detach().cpu().numpy()
            else:
                goal_np = np.asarray(goal)
        else:
            goal_np = None

        # Plan via imagination
        plan = self.planner.plan(z_np, goal=goal_np)
        action = plan.first_action

        # Track stats
        self.total_plans += 1
        self.total_imagination_steps += len(plan.trajectory)

        # Return as tensor
        return torch.from_numpy(action.astype(np.float32))


# =============================================================================
# Simple Test Arena
# =============================================================================

class SimpleTestArena:
    """
    Simple test arena for verifying OptimizedArenaCore.

    Provides random HDC vectors and basic step logic.
    """

    def __init__(
        self,
        num_envs: int,
        hdc_dim: int = 10_000,
        episode_length: int = 100,
    ):
        self.num_envs = num_envs
        self.hdc_dim = hdc_dim
        self.episode_length = episode_length
        self.steps = np.zeros(num_envs, dtype=int)

    def reset(
        self,
        population: Sequence[PlannerAgent],
    ) -> List[Dict[str, Any]]:
        """Reset all environments."""
        self.steps = np.zeros(self.num_envs, dtype=int)
        return self._make_obs()

    def step(
        self,
        actions: Sequence[np.ndarray],
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step all environments."""
        self.steps += 1

        # Random rewards
        rewards = np.random.randn(self.num_envs).astype(np.float32)

        # Done when episode length reached
        dones = (self.steps >= self.episode_length).astype(bool)

        # Info dicts
        infos = [{"step": int(s)} for s in self.steps]

        return self._make_obs(), rewards, dones, infos

    def _make_obs(self) -> List[Dict[str, Any]]:
        """Generate random HDC observations."""
        return [
            {"hdc_vector": np.random.randn(self.hdc_dim).astype(np.float32)}
            for _ in range(self.num_envs)
        ]


# =============================================================================
# Testing
# =============================================================================

def _test_arena_core():
    """Test the optimized arena core."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping test")
        return

    print("=" * 60)
    print("OptimizedArenaCore Test")
    print("=" * 60)

    # Create a simple encoder
    class SimpleEncoder(nn.Module):
        def __init__(self, hdc_dim: int = 10_000, latent_dim: int = 10):
            super().__init__()
            self.fc = nn.Linear(hdc_dim, latent_dim)

        def encode(self, h):
            return self.fc(h)

    # Create a simple agent
    class SimpleAgent(AraSpeciesBase):
        def __init__(self, action_dim: int = 4):
            self.action_dim = action_dim

        def select_action_from_latent(self, z_t, goal=None):
            # Random action based on latent
            return torch.randn(self.action_dim)

    # Setup
    hdc_dim = 1000  # Small for testing
    latent_dim = 10
    num_agents = 8

    encoder = SimpleEncoder(hdc_dim, latent_dim)
    config = ArenaCoreConfig(
        batch_size=4,
        hdc_dim=hdc_dim,
        log_every_n_steps=0,
    )

    core = OptimizedArenaCore(config, encoder)

    # Create population and arena
    population = [SimpleAgent() for _ in range(num_agents)]
    arena = SimpleTestArena(num_agents, hdc_dim, episode_length=50)

    # Evaluate
    result = core.evaluate_population(population, arena, episodes=2)

    print(f"\nFitnesses: {[f'{f:.2f}' for f in result.fitnesses]}")
    print(f"Total steps: {result.total_steps}")
    print(f"Steps/second: {result.steps_per_second:.0f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _test_arena_core()
