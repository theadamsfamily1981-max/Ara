#!/usr/bin/env python
"""
Tool Environment for AEPO

Simulates tool-use environment where agent decides whether to call tools.

Tools:
- TTW: Tripwire VFE check (alignment monitoring)
- PGU: Proof Generation Unit (SMT solver)
- HTTP: External API calls

Each tool has cost (latency) and benefit (improves reward if relevant).
Agent learns to call tools only when benefit > cost.

Usage:
    env = ToolEnv(num_tools=3)

    obs = env.reset()
    action = policy.get_action(obs)  # 0=call, 1=skip
    next_obs, reward, done, info = env.step(action)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class ToolAction(IntEnum):
    """Tool use actions."""
    CALL = 0
    SKIP = 1


@dataclass
class ToolConfig:
    """Configuration for a tool."""
    name: str
    cost: float  # Latency cost
    benefit: float  # Reward benefit if relevant
    relevance_prob: float  # Probability tool is relevant


class ToolEnv:
    """
    Environment for tool-use decision making.

    Agent observes context and decides whether to call each tool.
    Reward = task_reward - tool_costs
    """

    def __init__(
        self,
        num_tools: int = 3,
        obs_dim: int = 256,
        episode_length: int = 100
    ):
        """
        Initialize tool environment.

        Args:
            num_tools: Number of available tools
            obs_dim: Observation dimension
            episode_length: Steps per episode
        """
        self.num_tools = num_tools
        self.obs_dim = obs_dim
        self.episode_length = episode_length

        # Define tools
        self.tools = [
            ToolConfig(name="TTW", cost=0.005, benefit=0.1, relevance_prob=0.2),
            ToolConfig(name="PGU", cost=0.05, benefit=0.3, relevance_prob=0.3),
            ToolConfig(name="HTTP", cost=0.1, benefit=0.5, relevance_prob=0.4),
        ]

        # State
        self.step_count = 0
        self.current_obs = None
        self.tool_relevant = None  # Which tools are relevant this episode

    def reset(self) -> np.ndarray:
        """
        Reset environment.

        Returns:
            obs: Initial observation [obs_dim]
        """
        self.step_count = 0

        # Random observation
        self.current_obs = np.random.randn(self.obs_dim).astype(np.float32)

        # Determine which tools are relevant this episode
        self.tool_relevant = [
            np.random.random() < tool.relevance_prob
            for tool in self.tools
        ]

        return self.current_obs

    def step(
        self,
        action: int,
        tool_idx: int = 0
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment.

        Args:
            action: ToolAction (CALL or SKIP)
            tool_idx: Which tool to act on

        Returns:
            obs: Next observation
            reward: Reward
            done: Episode done
            info: Additional info
        """
        self.step_count += 1

        tool = self.tools[tool_idx]
        is_relevant = self.tool_relevant[tool_idx]

        # Compute reward
        if action == ToolAction.CALL:
            # Pay cost
            reward = -tool.cost

            # Get benefit if relevant
            if is_relevant:
                reward += tool.benefit

            tool_called = True
        else:
            # Skip - no cost, no benefit
            reward = 0.0

            # But lose potential benefit if tool was relevant
            if is_relevant:
                reward -= tool.benefit * 0.5  # Penalty for missing relevant tool

            tool_called = False

        # Next observation (random)
        next_obs = np.random.randn(self.obs_dim).astype(np.float32)
        self.current_obs = next_obs

        # Check if done
        done = self.step_count >= self.episode_length

        info = {
            'tool_name': tool.name,
            'tool_called': tool_called,
            'is_relevant': is_relevant,
            'step': self.step_count
        }

        return next_obs, reward, done, info

    def get_obs(self) -> np.ndarray:
        """Get current observation."""
        return self.current_obs


class ToolOracle:
    """
    Oracle that knows optimal policy.

    Used for baseline comparison.
    """

    def __init__(self, env: ToolEnv):
        self.env = env

    def get_action(self, tool_idx: int) -> int:
        """
        Get optimal action for tool.

        Args:
            tool_idx: Tool index

        Returns:
            action: CALL if relevant, SKIP otherwise
        """
        is_relevant = self.env.tool_relevant[tool_idx]

        if is_relevant:
            return ToolAction.CALL
        else:
            return ToolAction.SKIP
