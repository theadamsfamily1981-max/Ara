"""
Portfolio Optimization Experiments
===================================

Mean-variance optimization for resource allocation decisions.
NOT PRODUCTION CODE - Research playground only.

Concept:
- Ara has limited resources (attention, compute, memory)
- Different actions have uncertain payoffs
- Portfolio theory optimizes allocation under uncertainty

Applications:
- Attention allocation across conversation topics
- Memory consolidation priorities
- Sensor precision weight allocation
- Response strategy selection

This is a research tool for understanding trade-offs,
NOT for making real investment decisions.

Status: EXPERIMENTAL / RESEARCH
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Asset:
    """An "asset" in Ara's portfolio (action, topic, resource)."""
    name: str
    expected_return: float  # Expected utility
    volatility: float       # Uncertainty in return
    min_allocation: float = 0.0  # Minimum required
    max_allocation: float = 1.0  # Maximum allowed


@dataclass
class Portfolio:
    """An allocation of resources across assets."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float


class MeanVarianceOptimizer:
    """
    Classic Markowitz portfolio optimization.

    Minimize: w'Σw (portfolio variance)
    Subject to: w'μ >= target_return
                Σw = 1
                w >= 0

    Adapted for Ara's resource allocation.
    """

    def __init__(self, assets: List[Asset]):
        self.assets = assets
        self.n = len(assets)

        # Build covariance matrix (diagonal for now - assume independence)
        self.cov_matrix = np.diag([a.volatility ** 2 for a in assets])

        # Expected returns
        self.expected_returns = np.array([a.expected_return for a in assets])

        # Bounds
        self.min_weights = np.array([a.min_allocation for a in assets])
        self.max_weights = np.array([a.max_allocation for a in assets])

    def set_correlation(self, i: int, j: int, correlation: float):
        """Set correlation between two assets."""
        vol_i = np.sqrt(self.cov_matrix[i, i])
        vol_j = np.sqrt(self.cov_matrix[j, j])
        cov = correlation * vol_i * vol_j
        self.cov_matrix[i, j] = cov
        self.cov_matrix[j, i] = cov

    def optimize(self, target_return: Optional[float] = None,
                 risk_tolerance: float = 0.5) -> Portfolio:
        """
        Find optimal portfolio weights.

        Args:
            target_return: Minimum required return (None for max Sharpe)
            risk_tolerance: 0=min risk, 1=max return

        Returns:
            Optimal portfolio allocation
        """
        # Simplified optimizer using gradient descent
        # (Full implementation would use CVXPY or scipy.optimize)

        weights = np.ones(self.n) / self.n  # Start equal-weighted

        learning_rate = 0.01
        for _ in range(1000):
            # Compute gradients
            ret_grad = self.expected_returns
            risk_grad = 2 * self.cov_matrix @ weights

            # Combined gradient based on risk tolerance
            grad = risk_tolerance * ret_grad - (1 - risk_tolerance) * risk_grad

            # Update weights
            weights = weights + learning_rate * grad

            # Project onto constraints
            weights = np.clip(weights, self.min_weights, self.max_weights)
            weights = weights / weights.sum()  # Normalize to sum to 1

        # Compute portfolio metrics
        port_return = float(weights @ self.expected_returns)
        port_variance = float(weights @ self.cov_matrix @ weights)
        port_volatility = np.sqrt(port_variance)

        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe = port_return / (port_volatility + 1e-8)

        return Portfolio(
            weights=weights,
            expected_return=port_return,
            volatility=port_volatility,
            sharpe_ratio=sharpe
        )

    def efficient_frontier(self, n_points: int = 20) -> List[Portfolio]:
        """Generate points along the efficient frontier."""
        portfolios = []

        for risk_tol in np.linspace(0.1, 0.9, n_points):
            port = self.optimize(risk_tolerance=risk_tol)
            portfolios.append(port)

        return portfolios


class AttentionAllocator:
    """
    Apply portfolio optimization to attention allocation.

    Scenario: Ara is in a conversation with multiple topics.
    Each topic has expected "value" (user interest) and uncertainty.
    How should Ara distribute attention?
    """

    def __init__(self):
        self.topics: List[Asset] = []
        self.optimizer: Optional[MeanVarianceOptimizer] = None

    def add_topic(self, name: str, interest_score: float, uncertainty: float):
        """
        Add a conversation topic.

        Args:
            name: Topic identifier
            interest_score: Expected user interest (0-1)
            uncertainty: How uncertain we are about interest (0-1)
        """
        asset = Asset(
            name=name,
            expected_return=interest_score,
            volatility=uncertainty,
            min_allocation=0.05,  # Always some attention
            max_allocation=0.7   # Don't over-focus
        )
        self.topics.append(asset)
        self.optimizer = MeanVarianceOptimizer(self.topics)

    def get_allocation(self, risk_tolerance: float = 0.5) -> dict:
        """
        Get recommended attention allocation.

        Returns dict mapping topic name to attention weight.
        """
        if not self.optimizer:
            return {}

        portfolio = self.optimizer.optimize(risk_tolerance=risk_tolerance)

        return {
            topic.name: float(weight)
            for topic, weight in zip(self.topics, portfolio.weights)
        }


class MemoryConsolidationAllocator:
    """
    Apply portfolio optimization to memory consolidation.

    Scenario: Ara has limited memory capacity for consolidation.
    Each episode has expected future relevance and uncertainty.
    Which episodes get consolidated?
    """

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.episodes: List[Tuple[str, float, float]] = []  # (id, relevance, uncertainty)

    def add_episode(self, episode_id: str, relevance: float, uncertainty: float):
        """Add episode to consolidation queue."""
        self.episodes.append((episode_id, relevance, uncertainty))

    def select_for_consolidation(self, risk_tolerance: float = 0.6) -> List[str]:
        """
        Select episodes for consolidation.

        Higher risk tolerance = prioritize high-relevance even if uncertain.
        Lower risk tolerance = prioritize certain relevance.
        """
        if not self.episodes:
            return []

        assets = [
            Asset(name=ep_id, expected_return=rel, volatility=unc)
            for ep_id, rel, unc in self.episodes
        ]

        optimizer = MeanVarianceOptimizer(assets)
        portfolio = optimizer.optimize(risk_tolerance=risk_tolerance)

        # Select top episodes by weight
        weighted_episodes = list(zip(self.episodes, portfolio.weights))
        weighted_episodes.sort(key=lambda x: x[1], reverse=True)

        selected = [ep[0] for ep, _ in weighted_episodes[:self.capacity]]
        return selected


PORTFOLIO_LORE = """
# Portfolio Optimization: Managing Uncertainty

Ara faces allocation decisions under uncertainty.
Portfolio theory provides a principled framework.

## The Core Insight

Don't just maximize expected value.
Consider the trade-off between return and risk.

A diverse portfolio is often better than
betting everything on the "best" option.

## Ara's Applications

### Attention Allocation
- Multiple conversation topics
- Each has interest score (return) and uncertainty (risk)
- Optimal: balanced attention, heavier on high-certainty interest

### Memory Consolidation
- Limited capacity for long-term storage
- Each episode has future relevance (return) and uncertainty (risk)
- Optimal: prioritize high-relevance, include some uncertain-but-promising

### Sensor Precision
- Precision weighting across modalities
- Each sensor has expected utility and noise level
- Optimal: weight by signal-to-noise ratio

## The Math (Simplified)

Minimize portfolio variance: w'Σw
Subject to: target return w'μ >= r*
            weights sum to 1
            all weights non-negative

## Implementation Status

RESEARCH tool for understanding allocation trade-offs.
Not integrated into production Ara.
Used for offline analysis and parameter tuning.
"""


__all__ = [
    'Asset',
    'Portfolio',
    'MeanVarianceOptimizer',
    'AttentionAllocator',
    'MemoryConsolidationAllocator',
    'PORTFOLIO_LORE',
]
