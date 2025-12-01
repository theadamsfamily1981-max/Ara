#!/usr/bin/env python
"""
AEPO - Adaptive Entropy-regularized Policy Optimization

Entropy-regularized policy for tool-use control.
Learns when to call tools (TTW/PGU/HTTP) vs skip to minimize calls
while maintaining task reward.

Hard gates:
- Tool-call count −50% vs baseline
- Reward within −1% of baseline
- Stable entropy curve (no collapse) across 10 seeds

Architecture:
- Policy network: obs → logits for [call, skip]
- Entropy regularization: prevents collapse to deterministic policy
- Adaptive target entropy: adjusts based on training dynamics

Usage:
    policy = AEPO(obs_dim=256, ent_coef=0.02, target_entropy=0.7)

    # Training step
    loss, info = policy.loss(obs, advantages)
    loss.backward()
    optimizer.step()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class AEPOConfig:
    """Configuration for AEPO policy."""
    obs_dim: int = 256
    hidden_dim: int = 256
    ent_coef: float = 0.02  # Entropy regularization coefficient
    target_entropy: float = 0.7  # Target entropy (bits)
    adaptive_ent: bool = True  # Adaptive entropy target
    clip_ratio: float = 0.2  # PPO-style clipping


class AEPO(nn.Module):
    """
    Adaptive Entropy-regularized Policy Optimization.

    Binary policy: call tool or skip tool.
    """

    def __init__(
        self,
        obs_dim: Optional[int] = None,
        config: Optional[AEPOConfig] = None
    ):
        """
        Initialize AEPO policy.

        Args:
            obs_dim: Observation dimension
            config: AEPO configuration
        """
        super().__init__()

        self.config = config or AEPOConfig()

        if obs_dim is not None:
            self.config.obs_dim = obs_dim

        # Policy network: obs → [call, skip] logits
        self.pi = nn.Sequential(
            nn.Linear(self.config.obs_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, 2)  # Binary action
        )

        # Entropy coefficient (can be adaptive)
        self.ent_coef = self.config.ent_coef
        self.target_entropy = self.config.target_entropy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: obs → action logits.

        Args:
            obs: Observation tensor [batch, obs_dim]

        Returns:
            logits: Action logits [batch, 2]
        """
        return self.pi(obs)

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> tuple:
        """
        Sample action from policy.

        Args:
            obs: Observation [batch, obs_dim]
            deterministic: If True, use argmax instead of sampling

        Returns:
            action: Action indices [batch]
            log_prob: Log probability [batch]
        """
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.multinomial(probs, num_samples=1).squeeze(-1)

        log_prob = F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)

        return action, log_prob

    def loss(
        self,
        obs: torch.Tensor,
        advantages: Dict[str, torch.Tensor]
    ) -> tuple:
        """
        Compute AEPO loss.

        Args:
            obs: Observations [batch, obs_dim]
            advantages: Dict with 'action' and 'adv' tensors

        Returns:
            loss: Total loss
            info: Dict with loss components
        """
        logits = self.forward(obs)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # Entropy
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        # Policy loss (weighted by advantages)
        action = advantages["action"]
        adv = advantages["adv"]

        selected_log_probs = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        policy_loss = -(selected_log_probs * adv).mean()

        # Entropy regularization
        # Positive coefficient encourages high entropy
        ent_loss = -self.ent_coef * entropy

        # Total loss
        total_loss = policy_loss + ent_loss

        info = {
            "entropy": entropy.item(),
            "policy_loss": policy_loss.item(),
            "ent_loss": ent_loss.item(),
            "total_loss": total_loss.item()
        }

        return total_loss, info

    def update_ent_coef(self, current_entropy: float):
        """
        Adaptively update entropy coefficient.

        Args:
            current_entropy: Current policy entropy
        """
        if not self.config.adaptive_ent:
            return

        # If entropy too low, increase coefficient
        # If entropy too high, decrease coefficient
        entropy_error = self.target_entropy - current_entropy

        # Simple adaptive update
        self.ent_coef = max(0.001, min(0.1, self.ent_coef + 0.001 * entropy_error))
