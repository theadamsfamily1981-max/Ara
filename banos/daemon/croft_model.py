#!/usr/bin/env python3
"""
CROFT MODEL - User Preference Predictor
========================================

Bio-Affective Neuromorphic Operating System
A tiny neural network that learns what Croft likes.

This model is trained nightly by the Dreamer from accumulated
user feedback (EpisodicMemory.user_preferences). It learns to
predict:
1. Expected user rating for a (context, tool, style) triple
2. Probability of each friction flag
3. Optimal latency/token budget

The model is intentionally small (<1MB) so it can:
- Train in seconds on CPU
- Run inference in <1ms
- Be updated incrementally with new data

Usage:
    model = CroftModel()

    # Train from preference data
    model.train_from_preferences(episodic_memory)

    # Predict user satisfaction
    pred = model.predict(context_type="debugging", tool="grep", style="verbose")
    # => {'rating': 0.7, 'friction_risk': {'too_slow': 0.3, ...}, 'confidence': 0.8}

    # Get recommended approach
    rec = model.recommend(context_type="debugging")
    # => {'tool': 'grep', 'style': 'concise', 'expected_rating': 0.8}
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import PyTorch, fall back to simple model if unavailable
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using fallback preference model")

import numpy as np


# =============================================================================
# Vocabulary for encoding
# =============================================================================

# Common context types
CONTEXT_TYPES = [
    "unknown",
    "interaction",      # General chat
    "code_review",      # Reviewing code
    "debugging",        # Finding bugs
    "explanation",      # Explaining concepts
    "generation",       # Generating code/text
    "refactoring",      # Improving code
    "testing",          # Writing tests
    "documentation",    # Writing docs
    "exploration",      # Exploring codebase
    "planning",         # Planning work
]

# Common tools/approaches
TOOLS = [
    "unknown",
    "direct_answer",    # Just answer the question
    "search_first",     # Search codebase then answer
    "read_file",        # Read specific files
    "grep",             # Search for patterns
    "web_search",       # Search the web
    "multi_step",       # Complex multi-step reasoning
    "ask_clarify",      # Ask clarifying questions
    "show_code",        # Show code examples
    "explain_steps",    # Explain step by step
]

# Common styles
STYLES = [
    "unknown",
    "concise",          # Brief, to the point
    "verbose",          # Detailed explanation
    "technical",        # Technical language
    "friendly",         # Casual, friendly
    "formal",           # Formal language
    "step_by_step",     # Numbered steps
    "code_heavy",       # Mostly code
    "narrative",        # Story-like flow
]

# Friction flags we can predict
FRICTION_FLAGS = [
    "had_to_repeat",
    "missed_intent",
    "too_slow",
    "too_verbose",
    "too_terse",
    "wrong_tool",
    "wrong_style",
    "interrupted",
    "ignored_context",
    "over_promised",
]


def _vocab_to_idx(items: List[str]) -> Dict[str, int]:
    """Create vocabulary index."""
    return {item: idx for idx, item in enumerate(items)}


CONTEXT_IDX = _vocab_to_idx(CONTEXT_TYPES)
TOOL_IDX = _vocab_to_idx(TOOLS)
STYLE_IDX = _vocab_to_idx(STYLES)
FRICTION_IDX = _vocab_to_idx(FRICTION_FLAGS)


# =============================================================================
# PyTorch Model
# =============================================================================

if TORCH_AVAILABLE:

    class CroftNetwork(nn.Module):
        """
        Small neural network for preference prediction.

        Architecture:
        - Embedding layers for context, tool, style
        - Concat embeddings
        - 2-layer MLP
        - Multi-head output: rating, friction probs, latency

        Total params: ~15K (intentionally tiny)
        """

        def __init__(
            self,
            embed_dim: int = 16,
            hidden_dim: int = 32,
            n_contexts: int = len(CONTEXT_TYPES),
            n_tools: int = len(TOOLS),
            n_styles: int = len(STYLES),
            n_frictions: int = len(FRICTION_FLAGS),
        ):
            super().__init__()

            # Embeddings
            self.context_embed = nn.Embedding(n_contexts, embed_dim)
            self.tool_embed = nn.Embedding(n_tools, embed_dim)
            self.style_embed = nn.Embedding(n_styles, embed_dim)

            # MLP
            input_dim = embed_dim * 3
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)

            # Output heads
            self.rating_head = nn.Linear(hidden_dim, 1)  # Predict rating [-1, 1]
            self.friction_head = nn.Linear(hidden_dim, n_frictions)  # Friction probs
            self.latency_head = nn.Linear(hidden_dim, 1)  # Optimal latency

            # Dropout for regularization
            self.dropout = nn.Dropout(0.1)

        def forward(
            self,
            context_idx: torch.Tensor,
            tool_idx: torch.Tensor,
            style_idx: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass.

            Args:
                context_idx: [batch] context type indices
                tool_idx: [batch] tool indices
                style_idx: [batch] style indices

            Returns:
                Dict with 'rating', 'friction', 'latency' tensors
            """
            # Embed
            ctx_emb = self.context_embed(context_idx)
            tool_emb = self.tool_embed(tool_idx)
            style_emb = self.style_embed(style_idx)

            # Concat
            x = torch.cat([ctx_emb, tool_emb, style_emb], dim=-1)

            # MLP
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))

            # Outputs
            rating = torch.tanh(self.rating_head(x))  # [-1, 1]
            friction = torch.sigmoid(self.friction_head(x))  # [0, 1] per flag
            latency = F.softplus(self.latency_head(x))  # > 0

            return {
                'rating': rating.squeeze(-1),
                'friction': friction,
                'latency': latency.squeeze(-1),
            }


    class PreferenceDataset(Dataset):
        """Dataset for training from preference data."""

        def __init__(self, data: List[Dict[str, Any]]):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            return {
                'context_idx': item['context_idx'],
                'tool_idx': item['tool_idx'],
                'style_idx': item['style_idx'],
                'rating': item['rating'],
                'friction': item['friction'],
                'latency': item['latency'],
                'weight': item.get('weight', 1.0),
            }


# =============================================================================
# Main Model Class
# =============================================================================

@dataclass
class CroftModelConfig:
    """Configuration for the Croft model."""
    embed_dim: int = 16
    hidden_dim: int = 32
    learning_rate: float = 0.01
    epochs: int = 50
    batch_size: int = 32
    model_path: str = "/var/lib/banos/croft_model.pt"


class CroftModel:
    """
    User preference predictor.

    Learns from interaction history to predict what the user will like.
    """

    def __init__(self, config: Optional[CroftModelConfig] = None):
        self.config = config or CroftModelConfig()
        self._network = None
        self._optimizer = None
        self._last_trained = 0.0

        # Fallback: simple preference table for when PyTorch unavailable
        self._preference_table: Dict[Tuple[str, str, str], Dict[str, float]] = {}

        # Load saved model if exists
        if TORCH_AVAILABLE:
            self._init_network()
            self._load_if_exists()

    def _init_network(self) -> None:
        """Initialize the neural network."""
        if not TORCH_AVAILABLE:
            return

        self._network = CroftNetwork(
            embed_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
        )
        self._optimizer = torch.optim.Adam(
            self._network.parameters(),
            lr=self.config.learning_rate,
        )

    def _load_if_exists(self) -> None:
        """Load model from disk if it exists."""
        if not TORCH_AVAILABLE:
            return

        path = Path(self.config.model_path)
        if path.exists():
            try:
                state = torch.load(path, map_location='cpu')
                self._network.load_state_dict(state['model'])
                self._last_trained = state.get('timestamp', 0.0)
                logger.info(f"Loaded Croft model from {path}")
            except Exception as e:
                logger.warning(f"Failed to load Croft model: {e}")

    def save(self) -> None:
        """Save model to disk."""
        if not TORCH_AVAILABLE or self._network is None:
            return

        path = Path(self.config.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'model': self._network.state_dict(),
            'timestamp': time.time(),
            'config': {
                'embed_dim': self.config.embed_dim,
                'hidden_dim': self.config.hidden_dim,
            }
        }
        torch.save(state, path)
        logger.info(f"Saved Croft model to {path}")

    def train_from_preferences(
        self,
        episodic_memory,  # EpisodicMemory instance
        min_samples: int = 10,
    ) -> Dict[str, float]:
        """
        Train the model from preference data in EpisodicMemory.

        Args:
            episodic_memory: EpisodicMemory instance with preference data
            min_samples: Minimum samples needed to train

        Returns:
            Dict with training metrics
        """
        # Get preference data from database
        data = self._extract_training_data(episodic_memory)

        if len(data) < min_samples:
            logger.info(f"Not enough data to train ({len(data)} < {min_samples})")
            return {'status': 'skipped', 'reason': 'insufficient_data', 'n_samples': len(data)}

        if TORCH_AVAILABLE:
            return self._train_pytorch(data)
        else:
            return self._train_fallback(data)

    def _extract_training_data(self, episodic_memory) -> List[Dict[str, Any]]:
        """Extract training data from episodic memory."""
        data = []

        # Query preference patterns
        cursor = episodic_memory._conn.execute("""
            SELECT context_type, tool_used, style_used,
                   ema_rating, ema_latency_s, friction_history, total_count
            FROM user_preferences
            WHERE total_count >= 2
        """)

        for row in cursor:
            ctx_type, tool, style, rating, latency, friction_json, count = row

            # Convert to indices
            ctx_idx = CONTEXT_IDX.get(ctx_type, 0)
            tool_idx = TOOL_IDX.get(tool, 0)
            style_idx = STYLE_IDX.get(style, 0)

            # Parse friction history
            friction_counts = json.loads(friction_json) if friction_json else {}

            # Convert friction counts to probabilities
            total_friction = sum(friction_counts.values()) + 1  # +1 smoothing
            friction_probs = np.zeros(len(FRICTION_FLAGS))
            for flag, cnt in friction_counts.items():
                if flag in FRICTION_IDX:
                    friction_probs[FRICTION_IDX[flag]] = cnt / total_friction

            data.append({
                'context_idx': ctx_idx,
                'tool_idx': tool_idx,
                'style_idx': style_idx,
                'rating': rating,
                'friction': friction_probs,
                'latency': latency,
                'weight': min(count / 10.0, 1.0),  # Weight by sample count
            })

        return data

    def _train_pytorch(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train with PyTorch."""
        dataset = PreferenceDataset(data)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        self._network.train()
        total_loss = 0.0
        n_batches = 0

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            for batch in loader:
                self._optimizer.zero_grad()

                # Forward pass
                outputs = self._network(
                    torch.tensor(batch['context_idx']),
                    torch.tensor(batch['tool_idx']),
                    torch.tensor(batch['style_idx']),
                )

                # Compute loss
                rating_loss = F.mse_loss(
                    outputs['rating'],
                    torch.tensor(batch['rating'], dtype=torch.float32),
                )
                friction_loss = F.binary_cross_entropy(
                    outputs['friction'],
                    torch.tensor(np.stack(batch['friction']), dtype=torch.float32),
                )
                latency_loss = F.mse_loss(
                    outputs['latency'],
                    torch.tensor(batch['latency'], dtype=torch.float32),
                )

                # Weighted loss
                loss = rating_loss + 0.5 * friction_loss + 0.1 * latency_loss
                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            total_loss = epoch_loss / max(1, len(loader))

        self._network.eval()
        self._last_trained = time.time()
        self.save()

        return {
            'status': 'trained',
            'n_samples': len(data),
            'n_epochs': self.config.epochs,
            'final_loss': total_loss,
        }

    def _train_fallback(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train simple lookup table when PyTorch unavailable."""
        self._preference_table.clear()

        for item in data:
            ctx = CONTEXT_TYPES[item['context_idx']]
            tool = TOOLS[item['tool_idx']]
            style = STYLES[item['style_idx']]

            key = (ctx, tool, style)
            self._preference_table[key] = {
                'rating': item['rating'],
                'latency': item['latency'],
                'friction': item['friction'].tolist(),
            }

        self._last_trained = time.time()
        return {
            'status': 'trained_fallback',
            'n_patterns': len(self._preference_table),
        }

    def predict(
        self,
        context_type: str,
        tool: str,
        style: str,
    ) -> Dict[str, Any]:
        """
        Predict user satisfaction for a (context, tool, style) triple.

        Args:
            context_type: Type of context
            tool: Tool/approach to use
            style: Response style

        Returns:
            Dict with:
            - rating: Expected rating [-1, 1]
            - friction_risk: Dict of friction flag -> probability
            - latency_target: Suggested latency (seconds)
            - confidence: How confident we are (based on training data)
        """
        ctx_idx = CONTEXT_IDX.get(context_type, 0)
        tool_idx = TOOL_IDX.get(tool, 0)
        style_idx = STYLE_IDX.get(style, 0)

        if TORCH_AVAILABLE and self._network is not None:
            return self._predict_pytorch(ctx_idx, tool_idx, style_idx)
        else:
            return self._predict_fallback(context_type, tool, style)

    def _predict_pytorch(
        self,
        ctx_idx: int,
        tool_idx: int,
        style_idx: int,
    ) -> Dict[str, Any]:
        """Predict using PyTorch model."""
        self._network.eval()

        with torch.no_grad():
            outputs = self._network(
                torch.tensor([ctx_idx]),
                torch.tensor([tool_idx]),
                torch.tensor([style_idx]),
            )

        rating = outputs['rating'].item()
        friction = outputs['friction'][0].numpy()
        latency = outputs['latency'].item()

        # Convert friction to dict
        friction_risk = {
            FRICTION_FLAGS[i]: float(friction[i])
            for i in range(len(FRICTION_FLAGS))
            if friction[i] > 0.1  # Only include significant risks
        }

        # Confidence based on how much we've trained
        confidence = min(1.0, self._last_trained / 86400.0) if self._last_trained > 0 else 0.1

        return {
            'rating': rating,
            'friction_risk': friction_risk,
            'latency_target': latency,
            'confidence': confidence,
        }

    def _predict_fallback(
        self,
        context_type: str,
        tool: str,
        style: str,
    ) -> Dict[str, Any]:
        """Predict using simple lookup table."""
        key = (context_type, tool, style)

        if key in self._preference_table:
            pref = self._preference_table[key]
            friction_arr = pref.get('friction', [0.0] * len(FRICTION_FLAGS))
            friction_risk = {
                FRICTION_FLAGS[i]: friction_arr[i]
                for i in range(len(FRICTION_FLAGS))
                if i < len(friction_arr) and friction_arr[i] > 0.1
            }
            return {
                'rating': pref['rating'],
                'friction_risk': friction_risk,
                'latency_target': pref.get('latency', 2.0),
                'confidence': 0.5,
            }

        # No data - return neutral prediction
        return {
            'rating': 0.0,
            'friction_risk': {},
            'latency_target': 2.0,
            'confidence': 0.0,
        }

    def recommend(
        self,
        context_type: str,
        exclude_tools: Optional[List[str]] = None,
        exclude_styles: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Recommend the best (tool, style) for a context type.

        Args:
            context_type: What kind of task is this
            exclude_tools: Tools to avoid
            exclude_styles: Styles to avoid

        Returns:
            Dict with recommended tool, style, and expected rating
        """
        exclude_tools = exclude_tools or []
        exclude_styles = exclude_styles or []

        best_rating = -2.0
        best_tool = "direct_answer"
        best_style = "concise"
        best_friction = 1.0

        # Try all combinations
        for tool in TOOLS[1:]:  # Skip "unknown"
            if tool in exclude_tools:
                continue
            for style in STYLES[1:]:  # Skip "unknown"
                if style in exclude_styles:
                    continue

                pred = self.predict(context_type, tool, style)

                # Score: high rating, low friction
                friction_score = sum(pred['friction_risk'].values())
                score = pred['rating'] - 0.5 * friction_score

                if score > best_rating:
                    best_rating = pred['rating']
                    best_tool = tool
                    best_style = style
                    best_friction = friction_score

        return {
            'tool': best_tool,
            'style': best_style,
            'expected_rating': best_rating,
            'expected_friction': best_friction,
            'context_type': context_type,
        }

    def get_friction_warnings(
        self,
        context_type: str,
        tool: str,
        style: str,
        threshold: float = 0.3,
    ) -> List[str]:
        """
        Get warnings about likely friction for an approach.

        Args:
            context_type: Task context
            tool: Planned tool
            style: Planned style
            threshold: Minimum probability to warn

        Returns:
            List of friction flags that are likely to occur
        """
        pred = self.predict(context_type, tool, style)
        warnings = [
            flag for flag, prob in pred['friction_risk'].items()
            if prob >= threshold
        ]
        return warnings


# =============================================================================
# Integration with Dreamer
# =============================================================================

def train_croft_model_nightly(
    episodic_memory,
    model_path: str = "/var/lib/banos/croft_model.pt",
) -> Dict[str, Any]:
    """
    Train the Croft model as part of nightly Dreamer consolidation.

    Call this after dream() to update the preference model.
    """
    config = CroftModelConfig(model_path=model_path)
    model = CroftModel(config)

    result = model.train_from_preferences(episodic_memory)
    logger.info(f"Croft model training: {result}")

    return result


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Croft Model - User Preference Predictor")
    parser.add_argument("--train", action="store_true", help="Train from episodic memory")
    parser.add_argument("--predict", type=str, nargs=3, metavar=("CONTEXT", "TOOL", "STYLE"),
                       help="Predict rating for context/tool/style")
    parser.add_argument("--recommend", type=str, metavar="CONTEXT",
                       help="Recommend best tool/style for context")
    args = parser.parse_args()

    model = CroftModel()

    if args.predict:
        context, tool, style = args.predict
        pred = model.predict(context, tool, style)
        print(f"Prediction for ({context}, {tool}, {style}):")
        print(f"  Rating: {pred['rating']:.2f}")
        print(f"  Latency target: {pred['latency_target']:.1f}s")
        print(f"  Friction risks: {pred['friction_risk']}")
        print(f"  Confidence: {pred['confidence']:.2f}")

    elif args.recommend:
        rec = model.recommend(args.recommend)
        print(f"Recommendation for '{args.recommend}':")
        print(f"  Tool: {rec['tool']}")
        print(f"  Style: {rec['style']}")
        print(f"  Expected rating: {rec['expected_rating']:.2f}")

    elif args.train:
        try:
            from episodic_memory import get_episodic_memory
            mem = get_episodic_memory()
            result = model.train_from_preferences(mem)
            print(f"Training result: {result}")
        except Exception as e:
            print(f"Training failed: {e}")
    else:
        parser.print_help()
