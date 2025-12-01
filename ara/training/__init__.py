"""
ARA Training Package

Unified training system for all ARA components:
- TFAN transformer training
- SNN spiking network training
- HRRL agent training
- TGSFN criticality-controlled training

Usage:
    from ara.training import Trainer, TrainingConfig

    trainer = Trainer(config)
    trainer.train()
"""

import sys
from pathlib import Path

# Add parent paths for imports
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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


__all__ = [
    "UnifiedTrainer",
    "train",
    "TFANTrainer",
    "OnlineLoop",
    "SleepLoop",
    "DualLoopTrainer",
    "DataConfig",
]
