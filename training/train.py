"""
Training script for TF-A-N 7B with FDT, PGU, topology, and emotion integration.

Usage:
    # QUANTA-focused training (default)
    python training/train.py --config configs/7b/quanta_focus.yaml

    # Generic baseline training
    python training/train.py --config configs/7b/quanta_focus.yaml \
                              --data configs/7b/datasets/generic_base.yaml

    # CI quick smoke test
    python training/train.py --config configs/ci/ci_quick.yaml
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import json
from pathlib import Path
from typing import Dict, Optional
import time
import numpy as np

# TF-A-N imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tfan.models.tfan7b import TFANConfig, TFANForCausalLM, count_parameters
from training.optimizer import create_optimizer
from training.scheduler import get_cosine_schedule_with_warmup
from training.fdt_controller import FDTControllerWithEmotion
from training.data import DummyDataset, create_dataloader

# Optional imports
try:
    from tfan.pgu import ProofGatedUpdater
    HAS_PGU = True
except ImportError:
    HAS_PGU = False
    print("Warning: PGU not available")

try:
    import deepspeed
    HAS_DEEPSPEED = True
except ImportError:
    HAS_DEEPSPEED = False
    print("Warning: DeepSpeed not available")


class TFANTrainer:
    """
    Trainer for TF-A-N 7B with all TF-A-N components.

    Integrates:
    - FDT controller for EPR-CV ≤ 0.15
    - Emotion-based policy modulation
    - Optional PGU for formal verification
    - Optional topology regularization
    - Gradient clipping and mixed precision

    Args:
        model: TFANForCausalLM instance
        config: TFANConfig
        optimizer: Optimizer
        scheduler: LR scheduler
        device: Device for training
        use_fdt: Whether to use FDT controller
        use_pgu: Whether to use PGU
        use_topo: Whether to use topology regularization
        lambda_topo: Topology loss weight
        grad_clip: Gradient clipping value
    """

    def __init__(
        self,
        model: TFANForCausalLM,
        config: TFANConfig,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        use_fdt: bool = True,
        use_pgu: bool = False,
        use_topo: bool = False,
        lambda_topo: float = 0.1,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.use_fdt = use_fdt
        self.use_pgu = use_pgu and HAS_PGU
        self.use_topo = use_topo
        self.lambda_topo = lambda_topo
        self.grad_clip = grad_clip

        # FDT controller
        if self.use_fdt:
            self.fdt_controller = FDTControllerWithEmotion(
                emotion_weight=0.3,
                arousal_to_temp=True,
                valence_to_lr=True,
            )
        else:
            self.fdt_controller = None

        # PGU
        if self.use_pgu:
            self.pgu = ProofGatedUpdater(mode="soft")
        else:
            self.pgu = None

        # Metrics
        self.step_count = 0
        self.epoch_count = 0

    def compute_grad_variance(self) -> float:
        """
        Compute gradient variance across all parameters.

        Returns:
            grad_variance: Variance of gradients
        """
        grad_norms = []
        for param in self.model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        if len(grad_norms) == 0:
            return 0.0

        return float(np.var(grad_norms))

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        compute_metrics: bool = True,
    ) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch: Input batch with 'input_ids' and 'labels'
            compute_metrics: Whether to compute full metrics

        Returns:
            metrics: Dictionary of training metrics
        """
        self.model.train()

        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=self.use_topo,
            return_dict=True,
        )

        loss = outputs["loss"]

        # Add topology loss if enabled
        if self.use_topo and "topology_landscapes" in outputs:
            # Compute topology loss (stub, integrate with tfan/topo.py)
            topo_loss = torch.tensor(0.0, device=self.device)
            loss = loss + self.lambda_topo * topo_loss

        # Backward pass
        loss.backward()

        # Compute gradient variance for FDT
        grad_variance = self.compute_grad_variance() if self.use_fdt else 0.0

        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        # Get current LR
        base_lr = self.optimizer.param_groups[0]["lr"]

        # FDT controller step
        if self.use_fdt:
            # Get emotion if available
            emotion = None
            if "emotion" in outputs:
                emotion = {
                    "valence": outputs["emotion"]["valence"].mean().item(),
                    "arousal": outputs["emotion"]["arousal"].mean().item(),
                }

            # FDT step
            fdt_outputs = self.fdt_controller.step(
                loss=loss.item(),
                grad_variance=grad_variance,
                base_lr=base_lr,
                base_temp=self.model.temperature,
                emotion=emotion,
            )

            # Update LR (modulate optimizer)
            modulated_lr = fdt_outputs["lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = modulated_lr

            # Update temperature
            self.model.set_temperature(fdt_outputs["temperature"])

            epr_cv = fdt_outputs["epr_cv"]
        else:
            epr_cv = 0.0

        # PGU verification (optional)
        if self.use_pgu:
            # Create update payload (simplified)
            update_payload = {
                "step": self.step_count,
                "loss": loss.item(),
                "epr_cv": epr_cv,
            }
            pgu_result = self.pgu.verify_update(update_payload, is_safety_critical=False)

            if not pgu_result.proven and self.pgu.mode == "hard":
                # Veto update (skip optimizer step)
                self.optimizer.zero_grad()
                print(f"Step {self.step_count}: PGU veto (not proven)")
                return {"loss": loss.item(), "pgu_veto": True}

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        self.step_count += 1

        # Collect metrics
        metrics = {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            "step": self.step_count,
        }

        if self.use_fdt:
            metrics.update({
                "epr": fdt_outputs["epr"],
                "epr_cv": epr_cv,
                "temperature": self.model.temperature,
            })

        if self.use_pgu and self.pgu is not None:
            metrics["pgu_cache_hits"] = len(self.pgu.cache)

        return metrics

    def validate_gates(self) -> tuple[bool, Dict[str, bool]]:
        """
        Validate all TF-A-N gates.

        Returns:
            all_pass: Whether all gates pass
            results: Dictionary of gate results
        """
        results = {}

        # EPR-CV gate
        if self.use_fdt:
            fdt_metrics = self.fdt_controller.get_metrics()
            epr_cv = fdt_metrics.get("epr_cv", 0.0)
            results["epr_cv_gate"] = epr_cv <= 0.15
        else:
            results["epr_cv_gate"] = True

        all_pass = all(results.values())
        return all_pass, results

    def save_checkpoint(self, checkpoint_dir: str):
        """
        Save training checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_dir)

        # Save optimizer
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_dir, "optimizer.pt"),
        )

        # Save scheduler
        if self.scheduler is not None:
            torch.save(
                self.scheduler.state_dict(),
                os.path.join(checkpoint_dir, "scheduler.pt"),
            )

        # Save FDT state
        if self.use_fdt:
            torch.save(
                self.fdt_controller.fdt.epr_history,
                os.path.join(checkpoint_dir, "fdt_history.pt"),
            )

        # Save training state
        state = {
            "step_count": self.step_count,
            "epoch_count": self.epoch_count,
        }
        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(state, f)

        print(f"Checkpoint saved to {checkpoint_dir}")


def main(args):
    """Main training function."""

    # Load config
    config = TFANConfig.from_json_file(args.config)
    print(f"Loaded config from {args.config}")

    # Create model
    print("Creating model...")
    model = TFANForCausalLM(config)

    # Count parameters
    param_counts = count_parameters(model)
    print(f"\nModel parameter count:")
    print(f"  Total: {param_counts['total']:,} ({param_counts['total_billions']:.2f}B)")
    print(f"  Trainable: {param_counts['trainable']:,}")

    # Check if within target range (6.8-7.2B)
    if not (6.8e9 <= param_counts['total'] <= 7.2e9):
        print(f"WARNING: Param count {param_counts['total_billions']:.2f}B outside target range [6.8B, 7.2B]")
    else:
        print(f"✓ Param count within target range")

    # Move to device
    device = torch.device(args.device)
    model = model.to(device)

    # Convert to bf16 if requested
    if args.bf16:
        model = model.to(torch.bfloat16)
        print("Converted model to bfloat16")

    # Create optimizer
    optimizer = create_optimizer(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # Create data loader
    print("Creating data loader...")
    dataset = DummyDataset(
        vocab_size=config.vocab_size,
        seq_length=args.seq_length,
    )
    dataloader = create_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IterableDataset
        num_workers=0,
    )

    # Create trainer
    print("Creating trainer...")
    trainer = TFANTrainer(
        model=model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_fdt=args.use_fdt,
        use_pgu=args.use_pgu,
        use_topo=args.use_topo,
        lambda_topo=args.lambda_topo,
        grad_clip=args.grad_clip,
    )

    # Training loop
    print(f"\nStarting training for {args.max_steps} steps...")
    print(f"Batch size: {args.batch_size}, Seq length: {args.seq_length}")

    data_iter = iter(dataloader)
    start_time = time.time()

    for step in range(args.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Training step
        metrics = trainer.training_step(batch)

        # Log
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            throughput = (step + 1) / elapsed if elapsed > 0 else 0

            log_str = f"Step {step}/{args.max_steps} | Loss: {metrics['loss']:.4f} | LR: {metrics['lr']:.2e}"
            if args.use_fdt:
                log_str += f" | EPR-CV: {metrics['epr_cv']:.4f} | Temp: {metrics['temperature']:.2f}"
            log_str += f" | {throughput:.2f} steps/s"
            print(log_str)

        # Validate gates
        if step % args.gate_check_interval == 0 and step > 0:
            all_pass, results = trainer.validate_gates()
            print(f"\nGate validation at step {step}:")
            for gate, passed in results.items():
                status = "✓ PASS" if passed else "✗ FAIL"
                print(f"  {gate}: {status}")
            print()

        # Save checkpoint
        if step % args.save_interval == 0 and step > 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            trainer.save_checkpoint(checkpoint_dir)

    print(f"\nTraining completed in {time.time() - start_time:.2f}s")

    # Final checkpoint
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_checkpoint(final_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TF-A-N 7B model")

    # Model config (defaults to QUANTA-focused training)
    parser.add_argument("--config", type=str, default="configs/7b/quanta_focus.yaml")

    # Training args
    parser.add_argument("--output_dir", type=str, default="checkpoints/tfan7b")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_length", type=int, default=2048)

    # Optimizer args
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # TF-A-N components
    parser.add_argument("--use_fdt", action="store_true", default=True)
    parser.add_argument("--use_pgu", action="store_true", default=False)
    parser.add_argument("--use_topo", action="store_true", default=False)
    parser.add_argument("--lambda_topo", type=float, default=0.1)

    # Device args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bf16", action="store_true", default=True)

    # Logging args
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--gate_check_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=5000)

    args = parser.parse_args()

    main(args)
