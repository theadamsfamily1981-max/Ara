#!/usr/bin/env python3
"""
TGSFN-01 Training Script - Blueprint Protocol Implementation

Constructs and trains the Thermodynamic-Geometric Spiking Field Network (TGSFN)
integrated with TFAN-7B architecture.

4-Stage Construction:
    Stage 1: Architecture (TFAN-7B with SSA + TLS masks)
    Stage 2: Curriculum (QUANTA 70/20/10 mixture)
    Stage 3: Training (Thermodynamic regularization + AEPO)
    Stage 4: Validation (C-DP criticality checking)

Loss Function:
    L_total = L_CE + λ_diss * Π_q + L_AEPO

Where:
    L_CE    - Cross-entropy language modeling loss
    Π_q     - Entropy production (thermodynamic cost)
    L_AEPO  - Adaptive entropy policy optimization loss

Usage:
    python scripts/train_tgsfn_01.py --config configs/7b/quanta_focus.yaml
    python scripts/train_tgsfn_01.py --debug --max-steps 100
"""

import argparse
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import yaml

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import numpy as np

# Project imports
from grok_tgsfn.config import ThermoConfig
from grok_tgsfn.thermodynamics import ThermodynamicMonitor
from tfan.agent.aepo import AEPO, AEPOConfig
from analysis.criticality import validate_cdp_scaling, CriticalityResult
from analysis.avalanches import AvalancheAnalyzer, extract_avalanches

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

@dataclass
class TGSFN01Config:
    """Configuration for TGSFN-01 training."""

    # Model
    model_name: str = "tfan-7b"
    n_layers: int = 32
    n_heads: int = 32
    d_model: int = 4096
    d_ff: int = 11008
    vocab_size: int = 32000
    max_seq_length: int = 4096

    # SSA (Selective Sparse Attention)
    ssa_keep_ratio: float = 0.33
    ssa_local_window: int = 128
    ssa_hops: int = 2
    tls_alpha: float = 0.7

    # Dataset
    focus: str = "quanta"
    split_weights: List[float] = field(default_factory=lambda: [0.7, 0.2, 0.1])

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 50000
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    weight_decay: float = 0.01

    # Thermodynamics
    lambda_diss: float = 0.1
    lambda_geom: float = 0.01
    tau_m: float = 10.0

    # AEPO
    aepo_ent_coef: float = 0.02
    aepo_target_entropy: float = 0.7
    aepo_adaptive: bool = True

    # Criticality validation
    validate_every: int = 1000
    cdp_tolerance: float = 0.2  # γ_sT within 2.0 ± 0.2

    # Gates
    min_accuracy: float = 0.85
    max_latency_p95: float = 200.0
    max_epr_cv: float = 0.15

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def from_yaml(cls, path: str) -> "TGSFN01Config":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        config = cls()

        # Model
        if "model" in data:
            for k, v in data["model"].items():
                if hasattr(config, k):
                    setattr(config, k, v)

        # Dataset
        if "dataset" in data:
            config.focus = data["dataset"].get("focus", config.focus)
            config.split_weights = data["dataset"].get("split_weights", config.split_weights)

        # Training
        if "training" in data:
            config.batch_size = data["training"].get("batch_size", config.batch_size)
            config.learning_rate = data["training"].get("learning_rate", config.learning_rate)
            config.max_steps = data["training"].get("max_steps", config.max_steps)

        # Gates
        if "gates" in data:
            config.min_accuracy = data["gates"].get("min_accuracy", config.min_accuracy)
            config.max_latency_p95 = data["gates"].get("max_latency_p95", config.max_latency_p95)
            config.max_epr_cv = data["gates"].get("max_epr_cv", config.max_epr_cv)

        return config


# ============================================================
# Stage 1: Architecture (TFAN-7B with SSA + TLS)
# ============================================================

def init_tfan7b_model(config: TGSFN01Config) -> nn.Module:
    """
    Initialize TFAN-7B model with Selective Sparse Attention.

    Uses TLS (Topological Landmark Selection) for O(N log N) attention.
    """
    try:
        from tfan.models.tfan7b.modeling_tfan7b import TFANForCausalLM, TFANConfig

        model_config = TFANConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,
            num_hidden_layers=config.n_layers,
            num_attention_heads=config.n_heads,
            intermediate_size=config.d_ff,
            max_position_embeddings=config.max_seq_length,
            # SSA configuration
            attention_impl="ssa_radial_v1",
            ssa_keep_ratio=config.ssa_keep_ratio,
            ssa_local=config.ssa_local_window,
            ssa_hops=config.ssa_hops,
            tls_alpha=config.tls_alpha,
        )

        model = TFANForCausalLM(model_config)
        model = model.to(config.device)

        logger.info(
            f"Initialized TFAN-7B: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params"
        )
        logger.info(
            f"SSA config: keep_ratio={config.ssa_keep_ratio}, "
            f"local={config.ssa_local_window}, hops={config.ssa_hops}"
        )

        return model

    except ImportError as e:
        logger.warning(f"TFAN-7B not available: {e}")
        logger.warning("Creating placeholder model for demonstration")

        # Placeholder model for when full TFAN isn't available
        class PlaceholderTFAN(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.embed = nn.Embedding(config.vocab_size, config.d_model)
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(
                        d_model=config.d_model,
                        nhead=min(config.n_heads, 8),
                        dim_feedforward=min(config.d_ff, 2048),
                        batch_first=True,
                    )
                    for _ in range(min(config.n_layers, 4))
                ])
                self.lm_head = nn.Linear(config.d_model, config.vocab_size)
                self.config = config

            def forward(self, input_ids, attention_mask=None, labels=None, return_activations=False):
                x = self.embed(input_ids)
                activations = [x]
                for layer in self.layers:
                    x = layer(x)
                    activations.append(x)
                logits = self.lm_head(x)

                loss = None
                if labels is not None:
                    loss = nn.CrossEntropyLoss()(
                        logits.view(-1, logits.size(-1)),
                        labels.view(-1)
                    )

                if return_activations:
                    return {"loss": loss, "logits": logits, "activations": activations}
                return {"loss": loss, "logits": logits}

        return PlaceholderTFAN(config).to(config.device)


# ============================================================
# Stage 2: Curriculum (QUANTA 70/20/10)
# ============================================================

class QUANTADataset(Dataset):
    """
    QUANTA curriculum dataset with weighted sampling.

    Mixture:
        70% quanta_train - domain-specific data
        20% general_math - reasoning/logic
        10% topology_problems - structural tasks
    """

    def __init__(
        self,
        config: TGSFN01Config,
        split: str = "train",
        max_samples: int = 100000,
    ):
        self.config = config
        self.split = split
        self.max_length = config.max_seq_length

        # In production, load actual datasets
        # For now, create synthetic data
        self.data = self._create_synthetic_data(max_samples)

        logger.info(f"Created QUANTA dataset: {len(self.data)} samples")

    def _create_synthetic_data(self, n: int) -> List[Dict]:
        """Create synthetic training data."""
        data = []
        weights = self.config.split_weights

        for i in range(n):
            # Sample category based on weights
            r = np.random.random()
            if r < weights[0]:
                category = "quanta"
            elif r < weights[0] + weights[1]:
                category = "math"
            else:
                category = "topology"

            # Create random token sequence
            length = np.random.randint(64, self.max_length)
            tokens = np.random.randint(0, self.config.vocab_size, size=length)

            data.append({
                "input_ids": torch.tensor(tokens[:-1]),
                "labels": torch.tensor(tokens[1:]),
                "category": category,
            })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": item["input_ids"],
            "labels": item["labels"],
        }


def create_dataloader(config: TGSFN01Config, split: str = "train") -> DataLoader:
    """Create dataloader with QUANTA curriculum."""
    dataset = QUANTADataset(config, split)

    # Collate function with padding
    def collate_fn(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

        for i, item in enumerate(batch):
            seq_len = len(item["input_ids"])
            input_ids[i, :seq_len] = item["input_ids"]
            labels[i, :seq_len] = item["labels"]

        return {"input_ids": input_ids, "labels": labels}

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )


# ============================================================
# Stage 3: Training Loop (Thermodynamic + AEPO)
# ============================================================

class TGSFN01Trainer:
    """
    TGSFN-01 Trainer with thermodynamic regularization and AEPO.

    Optimizes:
        L_total = L_CE + λ_diss * Π_q + L_AEPO

    Where Π_q is entropy production from ThermodynamicMonitor.
    """

    def __init__(self, config: TGSFN01Config):
        self.config = config
        self.device = torch.device(config.device)

        # Stage 1: Initialize model
        logger.info("Stage 1: Initializing TFAN-7B architecture...")
        self.model = init_tfan7b_model(config)

        # Initialize thermodynamic monitor
        thermo_config = ThermoConfig(
            lambda_diss=config.lambda_diss,
            lambda_geom=config.lambda_geom,
            device=config.device,
        )
        self.thermo_monitor = ThermodynamicMonitor(thermo_config)
        logger.info(f"Initialized ThermodynamicMonitor (λ_diss={config.lambda_diss})")

        # Initialize AEPO
        aepo_config = AEPOConfig(
            obs_dim=config.d_model,
            ent_coef=config.aepo_ent_coef,
            target_entropy=config.aepo_target_entropy,
            adaptive_ent=config.aepo_adaptive,
        )
        self.aepo = AEPO(config=aepo_config).to(self.device)
        logger.info(f"Initialized AEPO (target_entropy={config.aepo_target_entropy})")

        # Optimizer
        self.optimizer = optim.AdamW(
            list(self.model.parameters()) + list(self.aepo.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # LR scheduler with warmup
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
        )

        # Stage 2: Create dataloader
        logger.info("Stage 2: Loading QUANTA curriculum...")
        self.train_loader = create_dataloader(config)

        # Metrics tracking
        self.step = 0
        self.metrics_history = []
        self.Pi_q_history = []
        self.avalanche_history = []

        # Avalanche analyzer for criticality
        self.avalanche_analyzer = AvalancheAnalyzer()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step with thermodynamic cost.

        Returns:
            Dictionary of metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass with activations
        outputs = self.model(
            input_ids,
            labels=labels,
            return_activations=True,
        )

        ce_loss = outputs["loss"] if outputs["loss"] is not None else torch.tensor(0.0)
        activations = outputs.get("activations", [])

        # Compute thermodynamic cost (Π_q)
        Pi_q = torch.tensor(0.0, device=self.device)
        if activations:
            # Stack activations for thermodynamic analysis
            act_tensor = torch.stack([a.mean(dim=1) for a in activations[-3:]], dim=1)

            # Approximate spikes as high-activation events
            threshold = act_tensor.std() * 2
            spikes = (act_tensor.abs() > threshold).float()

            # Compute entropy production
            dissipation = self.thermo_monitor.compute_spike_dissipation(
                membrane_potentials=act_tensor,
                spikes=spikes,
                tau_m=self.config.tau_m,
            )
            Pi_q = dissipation

        # AEPO loss (policy entropy regularization)
        aepo_loss = torch.tensor(0.0, device=self.device)
        if activations:
            # Use mean activation as observation
            obs = activations[-1].mean(dim=1)  # [batch, d_model]
            logits = self.aepo(obs)
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            aepo_loss = -self.config.aepo_ent_coef * entropy

        # Total loss
        total_loss = ce_loss + self.config.lambda_diss * Pi_q + aepo_loss

        # Backward pass
        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            self.scheduler.step()

        # Track metrics
        metrics = {
            "loss": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            "ce_loss": ce_loss.item() if isinstance(ce_loss, torch.Tensor) else ce_loss,
            "Pi_q": Pi_q.item() if isinstance(Pi_q, torch.Tensor) else Pi_q,
            "aepo_loss": aepo_loss.item() if isinstance(aepo_loss, torch.Tensor) else aepo_loss,
            "lr": self.scheduler.get_last_lr()[0],
        }

        self.Pi_q_history.append(metrics["Pi_q"])

        return metrics

    def validate_criticality(self) -> Optional[CriticalityResult]:
        """
        Stage 4: Validate C-DP criticality.

        Checks that γ_sT ≈ 2.0 (within tolerance).
        """
        self.model.eval()

        # Collect activations for avalanche analysis
        all_activations = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader):
                if batch_idx >= 10:  # Sample from 10 batches
                    break

                input_ids = batch["input_ids"].to(self.device)
                outputs = self.model(input_ids, return_activations=True)

                if "activations" in outputs:
                    # Get last layer activations
                    act = outputs["activations"][-1].cpu().numpy()
                    all_activations.append(act)

        if not all_activations:
            logger.warning("No activations collected for criticality validation")
            return None

        # Stack activations: [batch, seq, d_model] -> [total_time, neurons]
        activations = np.concatenate(all_activations, axis=0)
        activations = activations.reshape(-1, activations.shape[-1])

        # Extract avalanches
        # Threshold at 2 std above mean
        threshold = activations.mean() + 2 * activations.std()
        binary = (activations > threshold).astype(int)

        # Extract avalanches from binary activity
        avalanches = extract_avalanches(binary, threshold=0.5)

        if len(avalanches) < 50:
            logger.warning(f"Only {len(avalanches)} avalanches detected - insufficient for analysis")
            return None

        sizes = np.array([a.size for a in avalanches])
        durations = np.array([a.duration for a in avalanches])
        N = activations.shape[-1]

        # Validate C-DP scaling
        result = validate_cdp_scaling(sizes, durations, N, verbose=True)

        # Log results
        logger.info(f"Criticality validation:")
        logger.info(f"  τ = {result.tau.exponent:.3f} ± {result.tau.sigma:.3f}")
        logger.info(f"  α = {result.alpha.exponent:.3f} ± {result.alpha.sigma:.3f}")
        logger.info(f"  γ_sT = {result.gamma_sT:.3f} ± {result.gamma_sT_err:.3f}")
        logger.info(f"  C-DP distance = {result.cdp_distance:.3f}")
        logger.info(f"  Is critical: {result.is_critical}")

        # Adjust training if not critical
        if not result.is_critical:
            if result.gamma_sT < 1.8:
                # Subcritical - reduce regularization
                logger.warning("System appears SUBCRITICAL - reducing λ_diss")
                self.config.lambda_diss *= 0.9
            elif result.gamma_sT > 2.2:
                # Supercritical - increase regularization
                logger.warning("System appears SUPERCRITICAL - increasing λ_diss")
                self.config.lambda_diss *= 1.1

        return result

    def train(self):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Starting TGSFN-01 Training")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Max steps: {self.config.max_steps}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"λ_diss: {self.config.lambda_diss}")
        logger.info("=" * 60)

        start_time = time.time()
        data_iter = iter(self.train_loader)

        for self.step in range(1, self.config.max_steps + 1):
            # Get batch (with cycling)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Training step
            metrics = self.train_step(batch)
            self.metrics_history.append(metrics)

            # Log progress
            if self.step % 100 == 0:
                elapsed = time.time() - start_time
                avg_loss = np.mean([m["loss"] for m in self.metrics_history[-100:]])
                avg_Pi_q = np.mean([m["Pi_q"] for m in self.metrics_history[-100:]])

                logger.info(
                    f"Step {self.step}/{self.config.max_steps} | "
                    f"Loss: {avg_loss:.4f} | Π_q: {avg_Pi_q:.4f} | "
                    f"LR: {metrics['lr']:.2e} | Time: {elapsed:.1f}s"
                )

            # Criticality validation
            if self.step % self.config.validate_every == 0:
                logger.info("-" * 40)
                logger.info("Running criticality validation...")
                result = self.validate_criticality()
                if result:
                    self.avalanche_history.append({
                        "step": self.step,
                        "gamma_sT": result.gamma_sT,
                        "is_critical": result.is_critical,
                        "lambda_diss": self.config.lambda_diss,
                    })
                logger.info("-" * 40)

        # Final summary
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)

        # Final criticality check
        final_result = self.validate_criticality()
        if final_result:
            logger.info(f"Final γ_sT = {final_result.gamma_sT:.3f}")
            logger.info(f"Final C-DP membership: {final_result.is_critical}")

        return {
            "metrics": self.metrics_history,
            "avalanches": self.avalanche_history,
            "final_criticality": final_result,
        }


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TGSFN-01 Training - Blueprint Protocol Implementation"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lambda-diss", type=float, default=None,
        help="Override λ_diss (thermodynamic regularization)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode (fewer steps, more logging)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    if args.config:
        config = TGSFN01Config.from_yaml(args.config)
    else:
        config = TGSFN01Config()

    # Apply overrides
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lambda_diss:
        config.lambda_diss = args.lambda_diss
    if args.device:
        config.device = args.device

    # Debug mode
    if args.debug:
        config.max_steps = min(config.max_steps, 100)
        config.validate_every = 50
        config.batch_size = min(config.batch_size, 4)

    # Run training
    print("\n" + "=" * 60)
    print("  TGSFN-01 Blueprint Protocol - Construction Starting")
    print("=" * 60)
    print(f"\n  Model: {config.model_name}")
    print(f"  Focus: {config.focus}")
    print(f"  Split weights: {config.split_weights}")
    print(f"  λ_diss: {config.lambda_diss}")
    print(f"  Device: {config.device}")
    print("\n" + "=" * 60 + "\n")

    trainer = TGSFN01Trainer(config)
    results = trainer.train()

    print("\n" + "=" * 60)
    print("  TGSFN-01 Blueprint Protocol - Construction Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
