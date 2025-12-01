# tfan/backends/snn_emu.py
"""
SNN emulation backend with low-rank masked synapses.

Achieves 97-99% parameter reduction via:
1. Topological sparsity (TLS masks)
2. Low-rank factorization W = M ⊙ (U V^T)
3. Temporal basis sharing
4. Event-driven processing
"""

import torch
from torch import nn
from typing import Dict, Any
import numpy as np

from .base import Backend, BackendHooks


class SNNHooks(BackendHooks):
    """
    Hooks for SNN emulation with numerical stability and monitoring.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

        # Numerical stability params
        self.grad_clip = cfg.get('training', {}).get('grad_clip', 1.0)
        self.use_spectral_norm = cfg.get('snn', {}).get('use_spectral_norm', False)
        self.surrogate_beta_schedule = cfg.get('snn', {}).get('surrogate_beta_schedule', None)

        # Spike monitoring
        self.spike_rate_ema = None
        self.ema_alpha = 0.9

        # FDT integration
        self.fdt_controller = None
        self.use_fdt = cfg.get('tfan', {}).get('use_fdt', True)

        if self.use_fdt:
            try:
                from training.fdt_controller import FDTControllerWithEmotion
                fdt_cfg = cfg.get('tfan', {}).get('fdt', {})
                self.fdt_controller = FDTControllerWithEmotion(
                    kp=fdt_cfg.get('kp', 0.30),
                    ki=fdt_cfg.get('ki', 0.02),
                    kd=fdt_cfg.get('kd', 0.10),
                    ema_alpha=fdt_cfg.get('ema_alpha', 0.85),
                    target_epr_cv=fdt_cfg.get('target_epr_cv', 0.15),
                )
            except ImportError:
                print("⚠ FDT controller not available for SNN")
                self.fdt_controller = None

    def before_step(self, model: nn.Module):
        """
        Pre-step operations:
        - Global gradient clipping
        - Spectral normalization on U, V
        - Gradient sanity checks
        """
        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

        # Spectral norm on low-rank factors (optional)
        if self.use_spectral_norm:
            for name, param in model.named_parameters():
                if 'U' in name or 'V' in name:
                    # Apply spectral normalization
                    with torch.no_grad():
                        u_norm = torch.linalg.matrix_norm(param, ord=2)
                        if u_norm > 1.0:
                            param.data /= u_norm

        # Check for NaN/Inf gradients
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"⚠ NaN/Inf detected in gradients at step {self.step_count}")
                    param.grad.zero_()

    def after_step(self, model: nn.Module, aux: Dict[str, Any]):
        """
        Post-step operations:
        - Spike rate EMA
        - FDT PID updates (spike-rate aware)
        - Metric collection
        """
        super().after_step(model, aux)

        # Update spike rate EMA
        if 'spike_rate' in aux:
            spike_rate = aux['spike_rate']
            if self.spike_rate_ema is None:
                self.spike_rate_ema = spike_rate
            else:
                self.spike_rate_ema = (
                    self.ema_alpha * self.spike_rate_ema +
                    (1 - self.ema_alpha) * spike_rate
                )
            aux['spike_rate_ema'] = self.spike_rate_ema

        # FDT updates (if enabled)
        if self.fdt_controller is not None and 'loss' in aux:
            # Compute gradient variance
            grad_variance = 0.0
            count = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_variance += p.grad.var().item()
                    count += 1
            if count > 0:
                grad_variance /= count

            # FDT step with spike-rate modulation
            base_lr = self.cfg.get('training', {}).get('learning_rate', 1.5e-3)

            # Map spike rate to temperature (higher rate → lower threshold)
            spike_rate = aux.get('spike_rate_ema', 0.1)
            base_temp = 1.0 / (1.0 + spike_rate)  # Adaptive threshold

            fdt_outputs = self.fdt_controller.step(
                loss=aux['loss'],
                grad_variance=grad_variance,
                base_lr=base_lr,
                base_temp=base_temp
            )

            # Store metrics
            aux['epr_cv'] = self.fdt_controller.epr_cv
            aux['lr_multiplier'] = fdt_outputs.get('lr_multiplier', 1.0)
            aux['v_th_multiplier'] = fdt_outputs.get('temp_multiplier', 1.0)

    def log(self, step: int, aux: Dict[str, Any]):
        """Log SNN-specific metrics."""
        if step % 100 == 0:
            metrics = []
            if 'loss' in aux:
                metrics.append(f"loss={aux['loss']:.4f}")
            if 'spike_rate' in aux:
                metrics.append(f"spike_rate={aux['spike_rate']:.3f}")
            if 'spike_sparsity' in aux:
                metrics.append(f"sparsity={aux['spike_sparsity']:.2%}")
            if 'epr_cv' in aux:
                metrics.append(f"epr_cv={aux['epr_cv']:.4f}")
            if 'active_events' in aux:
                metrics.append(f"events={aux['active_events']}")

            if metrics:
                print(f"Step {step}: {', '.join(metrics)}")


class SNNBackend(Backend):
    """
    SNN emulation backend with low-rank masked synapses.

    Features:
    - 97-99% parameter reduction vs dense
    - TLS-based sparse connectivity
    - Event-driven processing
    - Surrogate gradient training
    - FDT homeostasis integration
    """

    def build_model(self) -> nn.Module:
        """
        Build SNN model with low-rank masked synapses.

        Returns:
            model: SNN with LIFLayerLowRank components
        """
        from tfan.snn import (
            LIFLayerLowRank,
            LowRankMaskedSynapse,
            build_uniform_random_mask,
            EventDrivenStepper,
        )

        model_cfg = self.cfg.get('model', {})
        snn_cfg = self.cfg.get('snn', model_cfg)

        N = snn_cfg.get('N', 4096)
        r = snn_cfg.get('lowrank_rank', 32)
        k_per_row = snn_cfg.get('k_per_row', 64)
        v_th = snn_cfg.get('v_th', 1.0)
        alpha = snn_cfg.get('alpha', 0.95)
        surrogate_scale = snn_cfg.get('surrogate_scale', 0.3)

        device = self.cfg.get('device', 'cpu')
        dtype = torch.float16 if self.cfg.get('dtype') == 'float16' else torch.float32

        # Build TLS mask (use random for initialization)
        # In production, this would use actual TLS scores from topology
        print(f"Building SNN mask: N={N}, k={k_per_row}")
        mask = build_uniform_random_mask(N=N, k_per_row=k_per_row, seed=42)

        # Create LIF layer
        lif = LIFLayerLowRank(
            N=N,
            r=r,
            synapse_cls=LowRankMaskedSynapse,
            mask_csr=mask,
            v_th=v_th,
            alpha=alpha,
            surrogate_scale=surrogate_scale,
            dtype=dtype,
            device=device
        )

        # Wrap in model with forward signature
        class SNNModel(nn.Module):
            def __init__(self, lif_layer, time_steps=256):
                super().__init__()
                self.lif = lif_layer
                self.time_steps = time_steps
                self.stepper = EventDrivenStepper(
                    lif_layer,
                    sparsity_threshold=0.75
                )

            def forward(self, x):
                """
                Forward pass with time unrolling.

                Args:
                    x: Input [batch, N] or [batch, time, N]

                Returns:
                    output: Accumulated spikes
                    aux: Metrics dict
                """
                batch_size = x.shape[0]

                # Initialize state
                v, s = self.lif.init_state(batch=batch_size, device=x.device)

                # If input is [batch, N], tile for time
                if x.dim() == 2:
                    x = x.unsqueeze(1).repeat(1, self.time_steps, 1)

                # Unroll in time
                spike_accumulator = torch.zeros_like(s)
                total_spikes = 0
                total_events = 0

                for t in range(self.time_steps):
                    # External input at this timestep
                    ext_input = x[:, t, :]

                    # Step with event-driven optimization
                    v, s = self.stepper.step(v, s, external_input=ext_input)

                    # Accumulate
                    spike_accumulator += s
                    total_spikes += s.sum().item()
                    total_events += (s > 0).sum().item()

                # Compute metrics
                spike_rate = total_spikes / (batch_size * self.lif.N * self.time_steps)
                spike_sparsity = 1.0 - spike_rate

                aux = {
                    'spike_rate': spike_rate,
                    'spike_sparsity': spike_sparsity,
                    'active_events': total_events,
                    'time_steps': self.time_steps,
                }

                return spike_accumulator, aux

        model = SNNModel(lif, time_steps=snn_cfg.get('time_steps', 256))

        # Print summary
        summary = lif.summary()
        print(f"\n{'='*60}")
        print(f"SNN Model Summary")
        print(f"{'='*60}")
        print(f"N: {summary['N']}")
        print(f"Rank: {summary['rank']}")
        print(f"Parameter reduction: {summary['reduction_pct']:.2f}%")
        print(f"Avg degree: {summary['avg_degree']:.1f}")
        print(f"Degree fraction: {summary['degree_frac']:.4f}")
        print(f"v_th: {summary['v_th']:.3f}")
        print(f"alpha: {summary['alpha']:.3f}")
        print(f"{'='*60}\n")

        return model

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Build AdamW optimizer with SNN-specific settings."""
        train_cfg = self.cfg.get('training', {})

        lr = train_cfg.get('learning_rate', 1.5e-3)  # Higher for SNN
        weight_decay = train_cfg.get('weight_decay', 0.01)
        betas = train_cfg.get('adam_betas', (0.9, 0.95))

        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )

    def build_hooks(self) -> BackendHooks:
        """Build SNN hooks with stability and monitoring."""
        return SNNHooks(self.cfg)

    def summary(self) -> Dict[str, Any]:
        """Return SNN-specific summary."""
        base_summary = super().summary()

        # Add SNN-specific stats
        if hasattr(self.model, 'lif'):
            lif_summary = self.model.lif.summary()
            base_summary.update({
                'param_reduction_pct': lif_summary['reduction_pct'],
                'avg_degree': lif_summary['avg_degree'],
                'degree_frac': lif_summary['degree_frac'],
            })

        return base_summary


__all__ = ['SNNBackend']
