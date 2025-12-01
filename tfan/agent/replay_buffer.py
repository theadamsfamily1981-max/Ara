#!/usr/bin/env python
"""
Replay Buffer for AEPO

Stores transitions for off-policy learning.

Usage:
    buffer = ReplayBuffer(capacity=10000)

    # Store transition
    buffer.add(obs, action, reward, next_obs, done)

    # Sample batch
    batch = buffer.sample(batch_size=256)
"""

import numpy as np
from typing import Dict, Tuple, NamedTuple
from collections import deque
import random


class Transition(NamedTuple):
    """Single transition."""
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """
    Experience replay buffer for AEPO.

    Stores transitions and samples batches for training.
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """
        Add transition to buffer.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Episode done
        """
        transition = Transition(obs, action, reward, next_obs, done)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            batch: Dict with arrays of obs, actions, rewards, etc.
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Sample random transitions
        transitions = random.sample(self.buffer, batch_size)

        # Unpack into arrays
        obs = np.array([t.obs for t in transitions])
        actions = np.array([t.action for t in transitions])
        rewards = np.array([t.reward for t in transitions])
        next_obs = np.array([t.next_obs for t in transitions])
        dones = np.array([t.done for t in transitions])

        return {
            'obs': obs,
            'actions': actions,
            'rewards': rewards,
            'next_obs': next_obs,
            'dones': dones
        }

    def __len__(self) -> int:
        """Get number of transitions in buffer."""
        return len(self.buffer)

    def clear(self):
        """Clear all transitions."""
        self.buffer.clear()


class EpisodeBuffer:
    """
    Buffer for storing complete episodes.

    Used for on-policy algorithms.
    """

    def __init__(self):
        self.episodes = []
        self.current_episode = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool
    ):
        """Add transition to current episode."""
        transition = Transition(obs, action, reward, next_obs, done)
        self.current_episode.append(transition)

        if done:
            # Episode finished, store it
            self.episodes.append(self.current_episode)
            self.current_episode = []

    def get_episodes(self) -> list:
        """Get all complete episodes."""
        return self.episodes

    def compute_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """
        Compute GAE advantages for all episodes.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda

        Returns:
            Dict with observations, actions, advantages
        """
        all_obs = []
        all_actions = []
        all_advantages = []

        for episode in self.episodes:
            # Extract episode data
            rewards = np.array([t.reward for t in episode])

            # Compute returns (simple Monte Carlo for now)
            returns = np.zeros_like(rewards)
            running_return = 0.0

            for t in reversed(range(len(rewards))):
                running_return = rewards[t] + gamma * running_return
                returns[t] = running_return

            # Normalize advantages
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Collect
            all_obs.extend([t.obs for t in episode])
            all_actions.extend([t.action for t in episode])
            all_advantages.extend(advantages)

        return {
            'obs': np.array(all_obs),
            'actions': np.array(all_actions),
            'advantages': np.array(all_advantages)
        }

    def clear(self):
        """Clear all episodes."""
        self.episodes.clear()
        self.current_episode.clear()

    def __len__(self) -> int:
        """Get number of complete episodes."""
        return len(self.episodes)
