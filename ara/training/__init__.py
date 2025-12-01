"""
ARA Training Package

Unified training system for all ARA components:
- TFAN transformer training
- SNN spiking network training
- HRRL agent training
- TGSFN criticality-controlled training
- Pareto-optimal TP-RL training (NEW)

Key Features:
- Multi-objective Pareto optimization (Accuracy, Energy, Latency, Stability)
- L1/L2/L3 closed-loop training with interoception
- CXL hardware-aware training with latency constraints
- Unified pipeline for end-to-end autonomous training

Usage:
    from ara.training import Trainer, TrainingConfig

    trainer = Trainer(config)
    trainer.train()

    # Or use Pareto training
    from ara.training import ParetoTrainer, run_pareto_training

    results = run_pareto_training(num_episodes=500)
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable
from datetime import datetime
import logging
import time

logger = logging.getLogger("ara.training")

# Add parent paths for imports
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Try numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Import training components
try:
    from training.train import main as train_main
    from training.data import DataConfig
except ImportError:
    train_main = None
    DataConfig = None

# Import TFAN trainer
try:
    from tfan.trainer import TFANTrainer
except ImportError:
    TFANTrainer = None

# Import agent training
try:
    from hrrl_agent.loops import OnlineLoop, SleepLoop, DualLoopTrainer
except ImportError:
    OnlineLoop = None
    SleepLoop = None
    DualLoopTrainer = None


class UnifiedTrainer:
    """
    Unified trainer that can handle different training backends.

    Backends:
        - 'tfan': Train TFAN transformer model
        - 'snn': Train spiking neural network
        - 'agent': Train HRRL agent
        - 'tgsfn': Train TGSFN with criticality control
    """

    def __init__(
        self,
        backend: str = "tfan",
        config: dict = None,
        device: str = "auto",
    ):
        """
        Initialize unified trainer.

        Args:
            backend: Training backend ('tfan', 'snn', 'agent', 'tgsfn')
            config: Configuration dictionary
            device: Device to use ('auto', 'cuda', 'cpu', 'mps')
        """
        self.backend = backend
        self.config = config or {}
        self.device = device

        # Initialize appropriate trainer
        self._trainer = None
        self._init_trainer()

    def _init_trainer(self):
        """Initialize the appropriate trainer based on backend."""
        if self.backend == "tfan" and TFANTrainer is not None:
            self._trainer = TFANTrainer(**self.config)
        elif self.backend == "agent" and DualLoopTrainer is not None:
            self._trainer = DualLoopTrainer(**self.config)
        else:
            # Default to no-op
            self._trainer = None

    def train(self, **kwargs):
        """Run training."""
        if self._trainer is None:
            raise RuntimeError(f"Trainer not available for backend: {self.backend}")

        if hasattr(self._trainer, 'train'):
            return self._trainer.train(**kwargs)
        elif hasattr(self._trainer, 'run'):
            return self._trainer.run(**kwargs)
        else:
            raise RuntimeError("Trainer has no train() or run() method")

    def evaluate(self, **kwargs):
        """Run evaluation."""
        if self._trainer is None:
            raise RuntimeError(f"Trainer not available for backend: {self.backend}")

        if hasattr(self._trainer, 'evaluate'):
            return self._trainer.evaluate(**kwargs)
        else:
            raise RuntimeError("Trainer has no evaluate() method")

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        if self._trainer and hasattr(self._trainer, 'save_checkpoint'):
            self._trainer.save_checkpoint(path)

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        if self._trainer and hasattr(self._trainer, 'load_checkpoint'):
            self._trainer.load_checkpoint(path)


# Convenience function
def train(
    backend: str = "tfan",
    config: dict = None,
    **kwargs
):
    """
    Convenience function for training.

    Args:
        backend: Training backend
        config: Configuration dictionary
        **kwargs: Additional training arguments

    Returns:
        Training results
    """
    trainer = UnifiedTrainer(backend=backend, config=config)
    return trainer.train(**kwargs)


# =============================================================================
# PARETO-OPTIMAL TRAINING
# =============================================================================

# Import Ara components for Pareto training
try:
    from ara.tprl import (
        TPRLTrainer,
        TrainingConfig as TPRLConfig,
        MaskEnvironment,
        TPRLAgent,
    )
    TPRL_AVAILABLE = True
except ImportError:
    TPRL_AVAILABLE = False
    TPRLTrainer = None

try:
    from ara.interoception import (
        InteroceptionCore,
        L1BodyState,
        L2PerceptionState,
    )
    INTEROCEPTION_AVAILABLE = True
except ImportError:
    INTEROCEPTION_AVAILABLE = False
    InteroceptionCore = None

try:
    from ara.cxl_control import (
        ControlPlane,
        hls_l3_metacontrol,
    )
    CXL_AVAILABLE = True
except ImportError:
    CXL_AVAILABLE = False
    ControlPlane = None


@dataclass
class ParetoObjective:
    """Single objective in Pareto optimization."""
    name: str
    weight: float = 1.0
    target: float = 1.0
    minimize: bool = False  # True = lower is better
    constraint_min: Optional[float] = None
    constraint_max: Optional[float] = None

    def score(self, value: float) -> float:
        """Compute normalized score for this objective."""
        if self.minimize:
            return max(0, 1.0 - value / (self.target + 1e-8))
        else:
            return min(1.0, value / (self.target + 1e-8))


@dataclass
class ParetoConfig:
    """Configuration for Pareto optimization."""
    objectives: List[ParetoObjective] = field(default_factory=lambda: [
        ParetoObjective("accuracy", weight=1.0, target=0.95),
        ParetoObjective("energy", weight=0.5, target=0.1, minimize=True),
        ParetoObjective("latency_us", weight=0.3, target=200.0, minimize=True),
        ParetoObjective("stability", weight=0.3, target=0.9),
    ])
    num_episodes: int = 500
    max_steps_per_episode: int = 500
    validate_every: int = 50
    checkpoint_dir: str = "./checkpoints/pareto"
    learning_rate: float = 0.001
    gamma: float = 0.99
    early_stopping_patience: int = 100


@dataclass
class ParetoResult:
    """Result from Pareto training."""
    episode: int
    objectives: Dict[str, float]
    pareto_score: float
    is_pareto_optimal: bool = False
    tprl_metrics: Dict[str, float] = field(default_factory=dict)
    interoception_pad: Optional[Dict] = None
    cxl_latency_us: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode": self.episode,
            "objectives": self.objectives,
            "pareto_score": self.pareto_score,
            "tprl": self.tprl_metrics,
            "pad": self.interoception_pad,
        }


class ParetoFront:
    """Maintains Pareto-optimal solutions."""

    def __init__(self, objectives: List[ParetoObjective]):
        self.objectives = objectives
        self.solutions: List[ParetoResult] = []

    def dominates(self, a: ParetoResult, b: ParetoResult) -> bool:
        """Check if a dominates b."""
        better_or_equal = 0
        strictly_better = 0
        for obj in self.objectives:
            va = a.objectives.get(obj.name, 0)
            vb = b.objectives.get(obj.name, 0)
            if obj.minimize:
                va, vb = -va, -vb
            if va >= vb:
                better_or_equal += 1
            if va > vb:
                strictly_better += 1
        return better_or_equal == len(self.objectives) and strictly_better > 0

    def add(self, result: ParetoResult) -> bool:
        """Add solution if non-dominated."""
        for sol in self.solutions:
            if self.dominates(sol, result):
                return False
        self.solutions = [s for s in self.solutions if not self.dominates(result, s)]
        result.is_pareto_optimal = True
        self.solutions.append(result)
        return True

    def get_best(self) -> Optional[ParetoResult]:
        """Get best by weighted score."""
        if not self.solutions:
            return None
        return max(self.solutions, key=lambda r: r.pareto_score)


class ParetoTrainer:
    """
    Pareto-optimal trainer combining TP-RL, Interoception, and CXL.

    Optimizes:
    - Accuracy: SNN performance
    - Energy: Spike rate / cost
    - Latency: CXL control loop latency
    - Stability: Topology stability
    """

    def __init__(self, config: Optional[ParetoConfig] = None):
        self.config = config or ParetoConfig()

        # Initialize TP-RL
        if TPRL_AVAILABLE:
            tprl_config = TPRLConfig(
                num_episodes=self.config.num_episodes,
                max_steps_per_episode=self.config.max_steps_per_episode,
                validate_every=self.config.validate_every,
                learning_rate=self.config.learning_rate,
                checkpoint_dir=self.config.checkpoint_dir,
            )
            self.tprl = TPRLTrainer(tprl_config)
        else:
            self.tprl = None

        # Initialize Interoception
        if INTEROCEPTION_AVAILABLE:
            self.interoception = InteroceptionCore(population_size=32)
        else:
            self.interoception = None

        # Pareto front
        self.pareto_front = ParetoFront(self.config.objectives)
        self.training_history: List[ParetoResult] = []
        self.current_episode = 0
        self.best_score = float('-inf')

        logger.info(f"ParetoTrainer initialized: TPRL={TPRL_AVAILABLE}, Intero={INTEROCEPTION_AVAILABLE}")

    def _compute_score(self, objectives: Dict[str, float]) -> float:
        """Compute weighted Pareto score."""
        total = 0.0
        weights = 0.0
        for obj in self.config.objectives:
            if obj.name in objectives:
                total += obj.weight * obj.score(objectives[obj.name])
                weights += obj.weight
        return total / weights if weights > 0 else 0.0

    def _run_step(self) -> ParetoResult:
        """Run single training step."""
        self.current_episode += 1

        # Run TP-RL episode if available
        tprl_metrics = {}
        if self.tprl:
            result = self.tprl._run_episode(explore=True)
            self.tprl.agent.update_policy()
            self.tprl._decay_exploration()
            tprl_metrics = {
                "accuracy": result.accuracy,
                "energy": result.energy,
                "stability": result.stability,
                "density": result.density,
            }
        else:
            # Simulated metrics
            import random
            tprl_metrics = {
                "accuracy": 0.5 + 0.4 * random.random(),
                "energy": 0.1 + 0.1 * random.random(),
                "stability": 0.8 + 0.15 * random.random(),
                "density": 0.03 + 0.01 * random.random(),
            }

        # Interoception processing
        pad_dict = None
        if self.interoception:
            l1 = L1BodyState(
                heart_rate=70 + tprl_metrics["energy"] * 30,
                muscle_tension=0.2 + tprl_metrics["energy"] * 0.3,
            )
            l2 = L2PerceptionState(
                audio_valence=tprl_metrics["accuracy"] - 0.5,
                audio_arousal=1 - tprl_metrics["stability"],
            )
            pad = self.interoception.process_with_layers(l1, l2)
            pad_dict = pad.to_dict()

        # CXL latency
        cxl_latency = None
        if CXL_AVAILABLE and hls_l3_metacontrol:
            start = time.perf_counter()
            hls_l3_metacontrol(0, 0.5, 0.5)
            cxl_latency = (time.perf_counter() - start) * 1e6

        # Build objectives
        objectives = {
            "accuracy": tprl_metrics["accuracy"],
            "energy": tprl_metrics["energy"],
            "stability": tprl_metrics["stability"],
            "latency_us": cxl_latency or 10.0,
        }

        score = self._compute_score(objectives)

        return ParetoResult(
            episode=self.current_episode,
            objectives=objectives,
            pareto_score=score,
            tprl_metrics=tprl_metrics,
            interoception_pad=pad_dict,
            cxl_latency_us=cxl_latency,
        )

    def train(
        self,
        callback: Optional[Callable[[int, ParetoResult], bool]] = None,
    ) -> Dict[str, Any]:
        """Run Pareto-optimal training."""
        start_time = time.time()
        no_improvement = 0

        logger.info(f"Starting Pareto training for {self.config.num_episodes} episodes")

        try:
            while self.current_episode < self.config.num_episodes:
                result = self._run_step()
                self.training_history.append(result)
                self.pareto_front.add(result)

                if result.pareto_score > self.best_score + 0.001:
                    self.best_score = result.pareto_score
                    no_improvement = 0
                else:
                    no_improvement += 1

                # Logging
                if self.current_episode % 10 == 0:
                    logger.info(
                        f"Episode {self.current_episode}: pareto={result.pareto_score:.4f}, "
                        f"acc={result.objectives['accuracy']:.3f}, front={len(self.pareto_front.solutions)}"
                    )

                # Early stopping
                if no_improvement >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at episode {self.current_episode}")
                    break

                if callback and not callback(self.current_episode, result):
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted")

        elapsed = time.time() - start_time
        best = self.pareto_front.get_best()

        return {
            "total_episodes": self.current_episode,
            "elapsed_seconds": elapsed,
            "best_pareto_score": self.best_score,
            "best_result": best.to_dict() if best else None,
            "pareto_front_size": len(self.pareto_front.solutions),
        }


def run_pareto_training(
    num_episodes: int = 500,
    accuracy_weight: float = 1.0,
    energy_weight: float = 0.5,
    latency_weight: float = 0.3,
    checkpoint_dir: str = "./checkpoints/pareto",
) -> Dict[str, Any]:
    """
    Convenience function for Pareto-optimal training.

    Args:
        num_episodes: Number of episodes
        accuracy_weight: Weight for accuracy objective
        energy_weight: Weight for energy objective
        latency_weight: Weight for latency objective
        checkpoint_dir: Directory for checkpoints

    Returns:
        Training results

    Example:
        results = run_pareto_training(num_episodes=100)
        print(f"Best score: {results['best_pareto_score']}")
    """
    config = ParetoConfig(
        objectives=[
            ParetoObjective("accuracy", weight=accuracy_weight, target=0.95),
            ParetoObjective("energy", weight=energy_weight, target=0.1, minimize=True),
            ParetoObjective("latency_us", weight=latency_weight, target=200.0, minimize=True),
            ParetoObjective("stability", weight=0.3, target=0.9),
        ],
        num_episodes=num_episodes,
        checkpoint_dir=checkpoint_dir,
    )

    trainer = ParetoTrainer(config)
    return trainer.train()


__all__ = [
    # Original exports
    "UnifiedTrainer",
    "train",
    "TFANTrainer",
    "OnlineLoop",
    "SleepLoop",
    "DualLoopTrainer",
    "DataConfig",
    # Pareto training exports
    "ParetoObjective",
    "ParetoConfig",
    "ParetoResult",
    "ParetoFront",
    "ParetoTrainer",
    "run_pareto_training",
]
