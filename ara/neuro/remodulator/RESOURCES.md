# Predictive Coding & Active Inference Resources

Reference implementations, libraries, and tools for working with the
Brain Remodulator's theoretical foundations.

---

## Python Libraries

| Repository | Description | Key Features | License |
|------------|-------------|--------------|---------|
| [alec-tschantz/predcoding](https://github.com/alec-tschantz/predcoding) | Whittington & Bogacz (2017) implementation | Simple feedforward hierarchy; educational | MIT |
| [infer-actively/pypc](https://github.com/infer-actively/pypc) | General predictive coding with PyTorch | GPU acceleration; custom hierarchies | MIT |
| [bjornvz/PRECO](https://github.com/bjornvz/PRECO) | PCNs and PCGs (graph-structured) | Non-linear variants; research extensions | MIT |
| [ComputationalPsychiatry/pyhgf](https://github.com/ComputationalPsychiatry/pyhgf) | Hierarchical Gaussian filters | Computational psychiatry focus; model fitting | GPL-3.0 |
| [Bogacz-Group/PredictiveCoding](https://github.com/Bogacz-Group/PredictiveCoding) | Various PC architectures | Modular; MATLAB ports | MIT |
| [coxlab/prednet](https://github.com/coxlab/prednet) | Deep recurrent video prediction | Pre-trained models; temporal data | MIT |
| [BerenMillidge/PredictiveCodingBackprop](https://github.com/BerenMillidge/PredictiveCodingBackprop) | PC approximates backprop | Paper reproduction; deep learning bridge | MIT |
| [thebuckleylab/jpc](https://github.com/thebuckleylab/jpc) | JAX-based PC networks | High performance; JIT compilation | Apache-2.0 |
| [liukidar/pcax](https://github.com/liukidar/pcax) | Configurable PC networks in JAX | Tutorial notebooks; neuroscience tasks | MIT |

---

## Active Inference Libraries

| Repository | Description | Use Case |
|------------|-------------|----------|
| [infer-actively/pymdp](https://github.com/infer-actively/pymdp) | Active Inference in discrete state spaces | POMDP agents, decision-making |
| [SPFlow](https://github.com/SPFlow/SPFlow) | Sum-Product Networks | Probabilistic inference |
| [RxInfer.jl](https://github.com/biaslab/RxInfer.jl) | Reactive message passing (Julia) | Real-time Bayesian inference |

---

## Brain Remodulator Demos

This package includes working demonstrations:

```bash
# Predictive coding - precision effects
python -m ara.neuro.remodulator.predictive_coding_demo

# Interactive exploration
python -m ara.neuro.remodulator.predictive_coding_demo --interactive

# Active inference - curiosity and action
python -m ara.neuro.remodulator.active_inference

# Full remodulator simulation
python -m ara.neuro.remodulator.simulation --pattern all
```

---

## PyTorch Quick Example

Minimal predictive coding training loop using the `predictive-coding` library:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import predictive_coding as pc  # pip install predictive-coding

# Define hierarchical network with PC layers
model = nn.Sequential(
    nn.Linear(10, 256),
    pc.PCLayer(),  # Activity nodes
    nn.ReLU(),
    nn.Linear(256, 256),
    pc.PCLayer(),
    nn.ReLU(),
    nn.Linear(256, 784)  # Output (e.g., MNIST)
)

# Configure trainer
trainer = pc.PCTrainer(
    model,
    T=20,  # Inference iterations
    optimizer_x_fn=optim.Adam,
    optimizer_x_kwargs={'lr': 0.1},
    optimizer_p_fn=optim.Adam,
    optimizer_p_kwargs={'lr': 0.001},
)

# Training loop
def loss_fn(output, target):
    return 0.5 * (output - target).pow(2).sum()

for data, label in dataloader:
    labels_onehot = F.one_hot(label).float()
    trainer.train_on_batch(
        inputs=labels_onehot,
        loss_fn=loss_fn,
        loss_fn_kwargs={'_target': data},
    )
```

---

## Key Mathematical References

### Variational Free Energy (VFE)

```
F = E_q[ln q(s)] - E_q[ln p(o,s)]
  = D_KL[q(s) || p(s|o)] - ln p(o)
```

VFE is an upper bound on surprise. Minimizing F:
1. Maximizes model evidence ln p(o)
2. Minimizes divergence from true posterior

### Expected Free Energy (EFE)

```
G(π) = E_p[-ln p(o|π)] - E_p[H[p(s|o,π)]]
     = Pragmatic Value + Epistemic Value
```

- **Pragmatic**: Expected cost relative to preferences C
- **Epistemic**: Expected information gain (curiosity)

### POMDP Matrices

| Matrix | Definition | Role |
|--------|------------|------|
| **A** | P(o\|s) | Likelihood - perception |
| **B** | P(s'\|s,a) | Transition - prediction |
| **C** | log P(o) | Preferences - goals |
| **D** | P(s₀) | Initial prior |

---

## Related Reading

### Foundational
- Friston, K. (2010). The free-energy principle: A unified brain theory?
- Clark, A. (2013). Whatever next? Predictive brains, situated agents.
- Rao, R.P., & Ballard, D.H. (1999). Predictive coding in visual cortex.

### Clinical
- Adams, R.A., et al. (2013). Computational anatomy of psychosis.
- Lawson, R.P., et al. (2014). Adults with autism overestimate precision.
- Sterzer, P., et al. (2018). Predictive coding account of psychosis.

### Implementation
- Whittington, J.C., & Bogacz, R. (2017). Approximation of backprop.
- Millidge, B., et al. (2021). Predictive coding approximates backprop.
- Parr, T., & Friston, K. (2019). Generalised free energy and active inference.

---

## EEG Processing Tools

For implementing the Brain Remodulator's precision estimation:

| Tool | Purpose |
|------|---------|
| [MNE-Python](https://mne.tools/) | EEG/MEG analysis |
| [YASA](https://github.com/raphaelvallat/yasa) | Sleep/spectral analysis |
| [Neurodsp](https://github.com/neurodsp-tools/neurodsp) | Neural time series |
| [Fooof](https://github.com/fooof-tools/fooof) | Spectral parameterization |

---

*Last updated: 2024-12*
