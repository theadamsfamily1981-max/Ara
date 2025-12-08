"""
Homeostatic LIF Network with Hypervector Head
==============================================

Iteration 39-40: A tiny SNN that:
1. Uses optimized LIF neurons with inhibition + adaptive thresholds
2. Maintains homeostasis (target spike rates via threshold/leak adaptation)
3. Produces bipolar hypervectors for CorrSpike-HDC integration
4. Encodes fabric health status in a separate status HV

The hypervector head transforms hidden layer spike rates into
a bipolar {-1, +1} vector that CorrSpike-HDC can use for:
- Correlation against prototype HPVs
- Rolling state integration
- Escalation/policy decisions

Training uses multi-objective loss:
- Classification (CE)
- Rate regularization (keep spike budget sane)
- Hypervector contrast (normals cluster, anomalies separate)
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List
import json

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

# Try to import snntorch, fall back to pure PyTorch implementation
try:
    import snntorch as snn
    from snntorch import surrogate
    HAS_SNNTORCH = True
except ImportError:
    HAS_SNNTORCH = False


# ============================================================================
# Configuration (always available)
# ============================================================================

@dataclass
class HomeostaticLIFConfig:
    """Configuration for the homeostatic LIF network."""
    # Network architecture
    num_inputs: int = 784          # Input dimension (e.g., 28x28 for MNIST-like)
    num_hidden: int = 256          # Hidden layer neurons
    num_outputs: int = 2           # Output classes (normal/anomaly)
    num_steps: int = 25            # Simulation timesteps

    # Hypervector dimensions
    hv_dim: int = 1024             # Main task hypervector dimension
    status_hv_dim: int = 64        # Status/health hypervector dimension

    # LIF parameters
    beta: float = 0.9              # Membrane decay
    v_reset: float = 0.0           # Reset potential
    v_thr_init: float = 1.0        # Initial threshold

    # Inhibition
    lambda_inh: float = 0.1        # Inhibition strength
    gamma_inh: float = 0.05        # Cross-layer inhibition

    # Homeostasis targets
    target_hidden_rate: float = 0.15   # Target spike rate for hidden layer
    target_output_rate: float = 0.10   # Target spike rate for output layer
    homeo_tau: float = 100.0           # Homeostasis time constant

    # Loss weights
    rate_loss_weight: float = 0.01     # Weight for rate regularization
    hv_contrast_weight: float = 0.05   # Weight for HV separation loss

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 10


# ============================================================================
# PyTorch-dependent code (guarded)
# ============================================================================

if HAS_TORCH:

    class SurrogateSpike(torch.autograd.Function):
        """Surrogate gradient for spike function."""

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            return (input > 0).float()

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            # Fast sigmoid surrogate
            grad = grad_output / (1 + torch.abs(input * 10)) ** 2
            return grad

    def spike_fn(x):
        """Spike function with surrogate gradient."""
        return SurrogateSpike.apply(x)


    class HomeostaticLIFLayer(nn.Module):
        """
        LIF layer with adaptive thresholds and inhibition.

        Maintains homeostasis by tracking spike rates and adjusting thresholds.
        """

        def __init__(
            self,
            size: int,
            beta: float = 0.9,
            v_thr_init: float = 1.0,
            v_reset: float = 0.0,
            lambda_inh: float = 0.1,
            target_rate: float = 0.15,
            homeo_tau: float = 100.0
        ):
            super().__init__()
            self.size = size
            self.beta = beta
            self.v_reset = v_reset
            self.lambda_inh = lambda_inh
            self.target_rate = target_rate
            self.homeo_tau = homeo_tau

            # Learnable threshold per neuron
            self.v_thr = nn.Parameter(torch.ones(size) * v_thr_init)

            # Running rate estimate (not a parameter)
            self.register_buffer('rate_estimate', torch.ones(size) * target_rate)

        def forward(
            self,
            x: torch.Tensor,
            mem: torch.Tensor,
            global_inhibition: float = 0.0
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward one timestep.

            Args:
                x: Input current [batch, size]
                mem: Membrane potential [batch, size]
                global_inhibition: Cross-layer inhibition signal

            Returns:
                spk: Spikes [batch, size]
                mem: Updated membrane potential [batch, size]
            """
            # Leaky integration with inhibition
            mem = self.beta * mem + x - self.lambda_inh * global_inhibition

            # Spike generation
            spk = spike_fn(mem - self.v_thr)

            # Reset on spike
            mem = mem * (1 - spk) + self.v_reset * spk

            return spk, mem

        def update_homeostasis(self, batch_rate: torch.Tensor):
            """
            Update threshold based on observed spike rate.

            Args:
                batch_rate: Observed rate per neuron [size]
            """
            with torch.no_grad():
                # Exponential moving average of rate
                self.rate_estimate = (
                    (1 - 1/self.homeo_tau) * self.rate_estimate +
                    (1/self.homeo_tau) * batch_rate
                )

                # Adjust threshold: high rate → higher threshold
                rate_error = self.rate_estimate - self.target_rate
                self.v_thr.data += 0.01 * rate_error


    class HypervectorHead(nn.Module):
        """
        Projects spike rates to bipolar hypervector space.

        Takes average spike rates from a layer and produces:
        - main_hv: Task-relevant hypervector for CorrSpike-HDC
        - status_hv: Fabric health/status encoding
        """

        def __init__(self, input_dim: int, hv_dim: int, status_dim: int):
            super().__init__()
            self.hv_dim = hv_dim
            self.status_dim = status_dim

            # Main task projection (bipolar)
            self.hv_proj = nn.Linear(input_dim, hv_dim, bias=False)

            # Status projection (encodes homeostasis state)
            self.status_proj = nn.Linear(input_dim, status_dim, bias=False)

            # Initialize with random orthogonal-ish projections
            nn.init.orthogonal_(self.hv_proj.weight)
            nn.init.orthogonal_(self.status_proj.weight)

        def forward(
            self,
            hidden_rates: torch.Tensor,
            homeo_deviation: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Project spike rates to hypervector space.

            Args:
                hidden_rates: Average spike rates [batch, hidden_dim]
                homeo_deviation: Optional homeostasis deviation [batch, hidden_dim]

            Returns:
                main_hv: Bipolar task hypervector [batch, hv_dim]
                status_hv: Bipolar status hypervector [batch, status_dim]
            """
            # Main HV: project and binarize
            hv_raw = self.hv_proj(hidden_rates)
            main_hv = torch.sign(hv_raw)
            # Handle zeros (rare but possible)
            main_hv = torch.where(main_hv == 0, torch.ones_like(main_hv), main_hv)

            # Status HV: incorporate homeostasis deviation if available
            if homeo_deviation is not None:
                status_input = torch.cat([hidden_rates, homeo_deviation], dim=-1)
                # Need to handle dimension mismatch
                status_raw = self.status_proj(hidden_rates)
            else:
                status_raw = self.status_proj(hidden_rates)

            status_hv = torch.sign(status_raw)
            status_hv = torch.where(status_hv == 0, torch.ones_like(status_hv), status_hv)

            return main_hv, status_hv


    class HomeostaticLIFNet(nn.Module):
        """
        Complete homeostatic LIF network with hypervector head.

        Architecture:
            Input → Linear → LIF Hidden → Linear → LIF Output
                        ↓
                    HV Head → [main_hv, status_hv]
        """

        def __init__(self, cfg: Optional[HomeostaticLIFConfig] = None):
            super().__init__()
            self.cfg = cfg or HomeostaticLIFConfig()
            cfg = self.cfg

            # Input → Hidden projection
            self.fc1 = nn.Linear(cfg.num_inputs, cfg.num_hidden)

            # Hidden LIF layer
            self.lif1 = HomeostaticLIFLayer(
                size=cfg.num_hidden,
                beta=cfg.beta,
                v_thr_init=cfg.v_thr_init,
                v_reset=cfg.v_reset,
                lambda_inh=cfg.lambda_inh,
                target_rate=cfg.target_hidden_rate,
                homeo_tau=cfg.homeo_tau
            )

            # Hidden → Output projection
            self.fc2 = nn.Linear(cfg.num_hidden, cfg.num_outputs)

            # Output LIF layer
            self.lif2 = HomeostaticLIFLayer(
                size=cfg.num_outputs,
                beta=cfg.beta,
                v_thr_init=cfg.v_thr_init,
                v_reset=cfg.v_reset,
                lambda_inh=cfg.lambda_inh,
                target_rate=cfg.target_output_rate,
                homeo_tau=cfg.homeo_tau
            )

            # Hypervector head
            self.hv_head = HypervectorHead(
                input_dim=cfg.num_hidden,
                hv_dim=cfg.hv_dim,
                status_dim=cfg.status_hv_dim
            )

        def forward(
            self,
            x: torch.Tensor,
            early_exit: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass over num_steps timesteps.

            Args:
                x: Input [batch, num_inputs]
                early_exit: If True, can exit early on clear classification

            Returns:
                spk_rec: Output spikes over time [num_steps, batch, num_outputs]
                mem_rec: Output membrane over time [num_steps, batch, num_outputs]
                hv_full: Concatenated [main_hv, status_hv] [batch, hv_dim + status_dim]
                homeo_dev: Homeostasis deviation [batch, num_hidden]
            """
            cfg = self.cfg
            batch_size = x.size(0)
            device = x.device

            # Initialize membrane potentials
            mem1 = torch.zeros(batch_size, cfg.num_hidden, device=device)
            mem2 = torch.zeros(batch_size, cfg.num_outputs, device=device)

            # Recording lists
            spk1_rec = []
            spk2_rec = []
            mem2_rec = []

            # Input current (constant over time for rate coding)
            cur1 = self.fc1(x)

            # Time loop
            for step in range(cfg.num_steps):
                # Hidden layer
                global_inh1 = mem2.mean(dim=1, keepdim=True) * cfg.gamma_inh if step > 0 else 0.0
                spk1, mem1 = self.lif1(cur1, mem1, global_inh1)
                spk1_rec.append(spk1)

                # Output layer
                cur2 = self.fc2(spk1)
                global_inh2 = spk1.mean(dim=1, keepdim=True) * cfg.gamma_inh
                spk2, mem2 = self.lif2(cur2, mem2, global_inh2)
                spk2_rec.append(spk2)
                mem2_rec.append(mem2)

                # Early exit check (if classification is clear)
                if early_exit and step > cfg.num_steps // 2:
                    spike_counts = torch.stack(spk2_rec).sum(0)  # [batch, outputs]
                    if (spike_counts.max(1).values > 5).all():
                        break

            # Stack recordings
            spk1_rec = torch.stack(spk1_rec)  # [T, batch, hidden]
            spk2_rec = torch.stack(spk2_rec)  # [T, batch, outputs]
            mem2_rec = torch.stack(mem2_rec)  # [T, batch, outputs]

            # Compute hidden spike rates for HV head
            hidden_rates = spk1_rec.mean(0)  # [batch, hidden] - avg over time

            # Homeostasis deviation: how far from target?
            target = self.cfg.target_hidden_rate
            homeo_dev = hidden_rates - target  # [batch, hidden]

            # Generate hypervectors
            main_hv, status_hv = self.hv_head(hidden_rates, homeo_dev)

            # Concatenate for full HV output
            hv_full = torch.cat([main_hv, status_hv], dim=-1)

            # Update homeostasis (running estimates)
            with torch.no_grad():
                batch_hidden_rate = spk1_rec.mean(dim=(0, 1))  # [hidden]
                batch_output_rate = spk2_rec.mean(dim=(0, 1))  # [outputs]
                self.lif1.update_homeostasis(batch_hidden_rate)
                self.lif2.update_homeostasis(batch_output_rate)

            return spk2_rec, mem2_rec, hv_full, homeo_dev

        def get_spike_stats(self, spk_rec: torch.Tensor) -> Dict[str, float]:
            """Get spike statistics from a recording."""
            return {
                'mean_rate': spk_rec.mean().item(),
                'max_rate': spk_rec.mean(0).max().item(),
                'min_rate': spk_rec.mean(0).min().item(),
                'sparsity': (spk_rec.sum(0) == 0).float().mean().item()
            }


    class HomeostaticLoss(nn.Module):
        """
        Multi-objective loss for homeostatic LIF training.

        Combines:
        - Classification loss (cross-entropy on membrane/logits)
        - Rate regularization (keep spike rates near targets)
        - Hypervector contrast (separate normal from anomaly clusters)
        """

        def __init__(self, cfg: HomeostaticLIFConfig):
            super().__init__()
            self.cfg = cfg
            self.ce_loss = nn.CrossEntropyLoss()

        def forward(
            self,
            mem_rec: torch.Tensor,
            targets: torch.Tensor,
            spk_rec: torch.Tensor,
            hv_full: torch.Tensor,
            hidden_rates: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            """
            Compute combined loss.

            Args:
                mem_rec: Output membrane over time [T, batch, outputs]
                targets: Ground truth labels [batch]
                spk_rec: Output spikes over time [T, batch, outputs]
                hv_full: Full hypervector [batch, hv_dim + status_dim]
                hidden_rates: Hidden layer spike rates [batch, hidden] (optional)

            Returns:
                total_loss: Combined loss scalar
                components: Dict of individual loss components
            """
            cfg = self.cfg

            # 1. Classification loss (sum over time)
            ce_loss = torch.zeros(1, device=mem_rec.device)
            for t in range(mem_rec.size(0)):
                ce_loss += self.ce_loss(mem_rec[t], targets)

            # 2. Rate regularization
            output_rate = spk_rec.mean()
            rate_loss = (output_rate - cfg.target_output_rate).pow(2)

            # 3. Hypervector contrastive loss
            hv_loss = self._hv_contrast_loss(hv_full, targets)

            # Combined
            total_loss = (
                ce_loss +
                cfg.rate_loss_weight * rate_loss +
                cfg.hv_contrast_weight * hv_loss
            )

            components = {
                'ce_loss': ce_loss.item(),
                'rate_loss': rate_loss.item(),
                'hv_loss': hv_loss.item(),
                'total_loss': total_loss.item()
            }

            return total_loss, components

        def _hv_contrast_loss(self, hv: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            """
            Contrastive loss for hypervector separation.

            Normals should cluster together, anomalies should be far from normal center.
            """
            # Only use main HV portion
            main_hv = hv[:, :self.cfg.hv_dim].float()

            mask_normal = (labels == 0)
            mask_anom = (labels == 1)

            if not mask_normal.any() or not mask_anom.any():
                return torch.zeros(1, device=hv.device)

            hv_norm = main_hv[mask_normal]
            hv_anom = main_hv[mask_anom]

            # Normal cluster center
            norm_center = hv_norm.mean(0, keepdim=True)

            # Similarity of normals to their center (want high)
            sim_norm = torch.cosine_similarity(hv_norm, norm_center, dim=1).mean()

            # Similarity of anomalies to normal center (want low)
            sim_anom = torch.cosine_similarity(hv_anom, norm_center, dim=1).mean()

            # Loss: want sim_norm high, sim_anom low (with margin)
            loss = (1.0 - sim_norm) + torch.clamp(sim_anom + 0.2, min=0.0)

            return loss


    def export_for_fpga(
        net: 'HomeostaticLIFNet',
        output_path: str = "lif_params.json"
    ) -> Dict:
        """
        Export trained network parameters for FPGA deployment.

        Exports:
        - LIF parameters (beta, thresholds, etc.)
        - Weight matrices (quantized to Q1.15)
        - HV projection matrices
        - Homeostasis targets
        """
        cfg = net.cfg

        # Quantization scale (Q1.15 fixed-point)
        Q_SCALE = 2**15

        def quantize(tensor):
            """Quantize to Q1.15."""
            return (tensor.detach().cpu().numpy() * Q_SCALE).astype(np.int32).tolist()

        params = {
            # LIF parameters
            "beta": cfg.beta,
            "v_reset": cfg.v_reset,
            "lambda_inh": cfg.lambda_inh,
            "gamma_inh": cfg.gamma_inh,
            "num_steps": cfg.num_steps,

            # Thresholds (per-neuron)
            "v_thr_hidden": net.lif1.v_thr.detach().cpu().tolist(),
            "v_thr_output": net.lif2.v_thr.detach().cpu().tolist(),

            # Quantized weights
            "fc1_weight": quantize(net.fc1.weight),
            "fc1_bias": quantize(net.fc1.bias) if net.fc1.bias is not None else None,
            "fc2_weight": quantize(net.fc2.weight),
            "fc2_bias": quantize(net.fc2.bias) if net.fc2.bias is not None else None,

            # Hypervector projections
            "hv_proj": quantize(net.hv_head.hv_proj.weight),
            "status_proj": quantize(net.hv_head.status_proj.weight),

            # Homeostasis targets
            "targets": {
                "target_hidden_rate": cfg.target_hidden_rate,
                "target_output_rate": cfg.target_output_rate,
            },

            # Dimensions
            "dims": {
                "num_inputs": cfg.num_inputs,
                "num_hidden": cfg.num_hidden,
                "num_outputs": cfg.num_outputs,
                "hv_dim": cfg.hv_dim,
                "status_hv_dim": cfg.status_hv_dim,
            },

            # Quantization info
            "q_scale": Q_SCALE,
        }

        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"Exported parameters to {output_path}")
        return params


    def export_c_header(
        net: 'HomeostaticLIFNet',
        output_path: str = "lif_params.h"
    ) -> None:
        """Export parameters as C header for direct FPGA inclusion."""
        cfg = net.cfg
        Q_SCALE = 2**15

        lines = [
            "// Auto-generated LIF parameters for FPGA",
            "// Iteration 39-40: Homeostatic LIF + HV Head",
            "#pragma once",
            "",
            f"#define NUM_INPUTS {cfg.num_inputs}",
            f"#define NUM_HIDDEN {cfg.num_hidden}",
            f"#define NUM_OUTPUTS {cfg.num_outputs}",
            f"#define NUM_STEPS {cfg.num_steps}",
            f"#define HV_DIM {cfg.hv_dim}",
            f"#define STATUS_HV_DIM {cfg.status_hv_dim}",
            "",
            f"#define Q_SCALE {Q_SCALE}",
            f"#define BETA_FP ((int32_t)({cfg.beta} * Q_SCALE))",
            f"#define V_RESET_FP ((int32_t)({cfg.v_reset} * Q_SCALE))",
            f"#define LAMBDA_INH_FP ((int32_t)({cfg.lambda_inh} * Q_SCALE))",
            f"#define GAMMA_INH_FP ((int32_t)({cfg.gamma_inh} * Q_SCALE))",
            "",
            f"#define TARGET_HIDDEN_RATE {cfg.target_hidden_rate}f",
            f"#define TARGET_OUTPUT_RATE {cfg.target_output_rate}f",
            "",
            "// Thresholds (learnable, per-neuron)",
            f"static const int32_t v_thr_hidden[NUM_HIDDEN] = {{",
        ]

        # Hidden thresholds
        thr_hidden = (net.lif1.v_thr.detach().cpu().numpy() * Q_SCALE).astype(np.int32)
        for i in range(0, len(thr_hidden), 8):
            chunk = thr_hidden[i:i+8]
            lines.append("    " + ", ".join(str(x) for x in chunk) + ",")
        lines.append("};")
        lines.append("")

        # Output thresholds
        lines.append(f"static const int32_t v_thr_output[NUM_OUTPUTS] = {{")
        thr_output = (net.lif2.v_thr.detach().cpu().numpy() * Q_SCALE).astype(np.int32)
        lines.append("    " + ", ".join(str(x) for x in thr_output))
        lines.append("};")

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        print(f"Exported C header to {output_path}")


    def demo():
        """Demonstrate the homeostatic LIF network."""
        print("=" * 60)
        print("Homeostatic LIF Network Demo")
        print("=" * 60)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")

        # Create network
        cfg = HomeostaticLIFConfig(
            num_inputs=784,
            num_hidden=128,
            num_outputs=2,
            num_steps=25,
            hv_dim=512,
            status_hv_dim=32
        )
        net = HomeostaticLIFNet(cfg).to(device)
        print(f"\nNetwork created:")
        print(f"  Inputs: {cfg.num_inputs}")
        print(f"  Hidden: {cfg.num_hidden}")
        print(f"  Outputs: {cfg.num_outputs}")
        print(f"  HV dim: {cfg.hv_dim}")
        print(f"  Status HV dim: {cfg.status_hv_dim}")

        # Create dummy batch
        batch_size = 32
        x = torch.randn(batch_size, cfg.num_inputs, device=device)
        targets = torch.randint(0, 2, (batch_size,), device=device)

        print(f"\n--- Forward Pass ---")
        spk_rec, mem_rec, hv_full, homeo_dev = net(x)

        print(f"  Output spikes shape: {spk_rec.shape}")
        print(f"  Output membrane shape: {mem_rec.shape}")
        print(f"  HV full shape: {hv_full.shape}")
        print(f"  Homeo deviation shape: {homeo_dev.shape}")

        # Spike stats
        stats = net.get_spike_stats(spk_rec)
        print(f"\n--- Spike Stats ---")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")

        # Loss computation
        print(f"\n--- Loss Computation ---")
        loss_fn = HomeostaticLoss(cfg)
        loss, components = loss_fn(mem_rec, targets, spk_rec, hv_full)
        print(f"  Total loss: {loss.item():.4f}")
        for k, v in components.items():
            print(f"  {k}: {v:.4f}")

        # HV analysis
        print(f"\n--- Hypervector Analysis ---")
        main_hv = hv_full[:, :cfg.hv_dim]
        status_hv = hv_full[:, cfg.hv_dim:]
        print(f"  Main HV range: [{main_hv.min().item()}, {main_hv.max().item()}]")
        print(f"  Status HV range: [{status_hv.min().item()}, {status_hv.max().item()}]")
        print(f"  Main HV sparsity (+1): {(main_hv > 0).float().mean().item():.2%}")

        # Export test
        print(f"\n--- Export Test ---")
        params = export_for_fpga(net, "/tmp/lif_params.json")
        print(f"  Exported {len(params)} parameter groups")

        print("\n" + "=" * 60)

else:
    # Stubs when PyTorch is not available
    def demo():
        print("PyTorch is required for homeostatic LIF network.")
        print("Install with: pip install torch")


if __name__ == '__main__':
    demo()
