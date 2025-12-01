# AEPO: Adaptive Entropy-regularized Policy Optimization

**Entropy-regularized tool-use policy for minimizing tool calls while maintaining performance.**

## Overview

AEPO learns when to call tools (TTW/PGU/HTTP) versus skipping them to minimize latency costs while maintaining task reward. The policy uses entropy regularization to prevent collapse to deterministic behavior and maintains exploration throughout training.

## Hard Gates

All AEPO implementations must meet these quantitative requirements:

| Metric | Threshold | Description |
|--------|-----------|-------------|
| **Tool-call reduction** | ≥ 50% | Reduction vs oracle baseline |
| **Reward delta** | ≤ 1% | Reward loss vs baseline |
| **Entropy stability** | No collapse | Entropy > 0.3 across seeds |

## Architecture

```
Observation (256-dim) → Policy Network → [CALL, SKIP] logits
                              ↓
                     Entropy Regularization
                              ↓
                      Adaptive Coefficient
```

### Components

1. **Policy Network** (`tfan/agent/aepo.py`)
   - 3-layer MLP: `obs_dim → hidden → hidden → 2`
   - Binary action space: CALL (0) or SKIP (1)
   - Entropy-regularized objective

2. **Tool Environment** (`tfan/agent/env_tools.py`)
   - Simulated environment with 3 tools: TTW, PGU, HTTP
   - Each tool has cost (latency) and benefit (reward if relevant)
   - Tool relevance determined per episode

3. **Replay Buffer** (`tfan/agent/replay_buffer.py`)
   - `EpisodeBuffer` for on-policy training
   - GAE (Generalized Advantage Estimation) computation
   - Episode storage and batching

## Installation

```bash
# Install TF-A-N
pip install -e .

# Verify AEPO components
python -c "from tfan.agent import AEPO, ToolEnv; print('AEPO ready')"
```

## Quick Start

### Training

```bash
# Train with default settings
python scripts/train_aepo.py --iterations 200 --seed 42

# Train with custom hyperparameters
python scripts/train_aepo.py \
    --iterations 500 \
    --episodes-per-iter 20 \
    --lr 1e-4 \
    --seed 123 \
    --log-dir artifacts/aepo_exp1
```

**Training outputs:**
- `artifacts/aepo/checkpoint_N.pt` - Intermediate checkpoints
- `artifacts/aepo/final.pt` - Final trained policy
- `artifacts/aepo/metrics.json` - Training metrics history
- `artifacts/aepo/final_report.json` - Gate verification results

### Evaluation

```bash
# Evaluate final checkpoint
python scripts/eval_aepo.py \
    --checkpoint artifacts/aepo/final.pt \
    --episodes 100

# Multi-seed evaluation for stability
python scripts/eval_aepo.py \
    --checkpoint artifacts/aepo/final.pt \
    --episodes 100 \
    --seeds 10 \
    --output artifacts/aepo/eval_results.json
```

## Usage Examples

### Basic Policy Usage

```python
import torch
from tfan.agent import AEPO, AEPOConfig

# Create policy
config = AEPOConfig(
    obs_dim=256,
    hidden_dim=256,
    ent_coef=0.02,
    target_entropy=0.7,
    adaptive_ent=True
)
policy = AEPO(config=config)

# Get action
obs = torch.randn(1, 256)
action, log_prob = policy.get_action(obs, deterministic=False)

print(f"Action: {action.item()}")  # 0 = CALL, 1 = SKIP
print(f"Log prob: {log_prob.item()}")
```

### Custom Training Loop

```python
import torch.optim as optim
from tfan.agent import AEPO, ToolEnv
from tfan.agent.replay_buffer import EpisodeBuffer

# Setup
policy = AEPO(obs_dim=256)
env = ToolEnv(num_tools=3)
optimizer = optim.Adam(policy.parameters(), lr=3e-4)
buffer = EpisodeBuffer()

# Collect episodes
for episode in range(10):
    obs = env.reset()
    done = False

    while not done:
        # Get action
        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
        action, _ = policy.get_action(obs_tensor)

        # Step environment
        tool_idx = env.step_count % env.num_tools
        next_obs, reward, done, info = env.step(action.item(), tool_idx)

        # Store transition
        buffer.add(obs, action.item(), reward, next_obs, done)
        obs = next_obs

# Compute advantages
batch = buffer.compute_advantages(gamma=0.99, gae_lambda=0.95)

# Train
obs = torch.from_numpy(batch['obs'])
actions = torch.from_numpy(batch['actions']).long()
advantages = torch.from_numpy(batch['advantages'])

loss, info = policy.loss(obs, {'action': actions, 'adv': advantages})
loss.backward()
optimizer.step()

print(f"Loss: {info['total_loss']:.3f}, Entropy: {info['entropy']:.3f}")
```

### Loading and Inference

```python
import torch
from tfan.agent import AEPO

# Load checkpoint
checkpoint = torch.load('artifacts/aepo/final.pt')
policy = AEPO(config=checkpoint['config'])
policy.load_state_dict(checkpoint['policy_state_dict'])
policy.eval()

# Inference
obs = torch.randn(1, 256)
with torch.no_grad():
    action, _ = policy.get_action(obs, deterministic=True)

print(f"Deterministic action: {action.item()}")
```

## Hyperparameter Guide

### Policy Configuration

```python
AEPOConfig(
    obs_dim=256,           # Observation dimension
    hidden_dim=256,        # Hidden layer size
    ent_coef=0.02,         # Entropy coefficient (higher = more exploration)
    target_entropy=0.7,    # Target entropy in bits
    adaptive_ent=True,     # Adaptive entropy coefficient
    clip_ratio=0.2         # PPO-style clipping (not currently used)
)
```

### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 3e-4 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `epochs_per_iter` | 4 | Training epochs per iteration |
| `batch_size` | 256 | Batch size |
| `episodes_per_iter` | 10 | Episodes to collect per iteration |

### Tuning Tips

- **High tool calls**: Increase `ent_coef` to encourage more skipping
- **Low reward**: Decrease `ent_coef` to be more greedy
- **Entropy collapse**: Enable `adaptive_ent=True` and increase `target_entropy`
- **Unstable training**: Reduce `lr` or increase `batch_size`

## Testing

```bash
# Run AEPO integration tests
pytest tests/test_aepo_integration.py -v

# Run specific test
pytest tests/test_aepo_integration.py::TestAEPOPolicy::test_forward_pass -v

# Run with coverage
pytest tests/test_aepo_integration.py --cov=tfan.agent --cov-report=html
```

## Gate Verification

After training, verify all hard gates:

```python
from scripts.eval_aepo import load_checkpoint, AEPOEvaluator
from tfan.agent import ToolEnv

# Load policy
policy = load_checkpoint('artifacts/aepo/final.pt')
env = ToolEnv(num_tools=3)

# Evaluate
evaluator = AEPOEvaluator(policy, env)
policy_metrics = evaluator.evaluate_policy(num_episodes=100)
baseline_metrics = evaluator.evaluate_baseline(num_episodes=100)
entropy = evaluator.compute_entropy(num_episodes=100)

# Verify gates
gates = evaluator.verify_gates(policy_metrics, baseline_metrics, entropy)

for gate_name, gate_info in gates.items():
    if gate_name != 'all_pass':
        print(f"{gate_name}: {gate_info['value']:.3f} "
              f"({'PASS' if gate_info['pass'] else 'FAIL'})")
```

## Integration with TF-A-N

AEPO can be integrated into the main TF-A-N training loop to adaptively control tool usage:

```python
from tfan import TFANConfig, TFANTrainer
from tfan.agent import AEPO, AEPOConfig

# Create AEPO policy
aepo_config = AEPOConfig(obs_dim=model.config.hidden_size)
tool_policy = AEPO(config=aepo_config)

# In training loop
def should_call_tool(context_embedding):
    """Decide whether to call tool based on AEPO policy."""
    obs = torch.from_numpy(context_embedding).unsqueeze(0)

    with torch.no_grad():
        action, _ = tool_policy.get_action(obs, deterministic=False)

    return action.item() == 0  # CALL = 0

# Use in TTW/PGU calls
if should_call_tool(context_emb):
    result = ttw.check_alignment(...)
else:
    # Skip tool call
    pass
```

## Troubleshooting

### Issue: Entropy collapses to near-zero

**Solution:**
- Enable `adaptive_ent=True`
- Increase `target_entropy` (try 0.8-1.0)
- Increase initial `ent_coef` (try 0.05-0.1)

### Issue: Tool calls not reducing

**Solution:**
- Increase training iterations
- Increase `ent_coef` to encourage exploration
- Check environment rewards are properly balanced

### Issue: Reward dropping below -1%

**Solution:**
- Decrease `ent_coef` to be less exploratory
- Increase `episodes_per_iter` for more data
- Check baseline is computed correctly

### Issue: Unstable training

**Solution:**
- Reduce learning rate (try 1e-4)
- Increase batch size (try 512)
- Add gradient clipping (already enabled at 0.5)

## Advanced Topics

### Custom Tool Environments

Create custom environments by subclassing `ToolEnv`:

```python
from tfan.agent.env_tools import ToolEnv, ToolConfig

class CustomToolEnv(ToolEnv):
    def __init__(self):
        super().__init__(num_tools=5, obs_dim=512)

        # Define custom tools
        self.tools = [
            ToolConfig(name="FastTool", cost=0.01, benefit=0.2, relevance_prob=0.5),
            ToolConfig(name="SlowTool", cost=0.1, benefit=0.8, relevance_prob=0.2),
            # ... more tools
        ]

    def _compute_reward(self, action, tool_idx):
        # Custom reward logic
        pass
```

### Multi-Agent AEPO

Train multiple AEPO agents for different tool types:

```python
# Separate policies for different tools
ttw_policy = AEPO(obs_dim=256, config=AEPOConfig(ent_coef=0.01))
pgu_policy = AEPO(obs_dim=256, config=AEPOConfig(ent_coef=0.05))

# Use appropriate policy
if tool_type == 'TTW':
    action, _ = ttw_policy.get_action(obs)
elif tool_type == 'PGU':
    action, _ = pgu_policy.get_action(obs)
```

## References

- **REINFORCE**: Williams, R. J. (1992). Simple statistical gradient-following algorithms
- **GAE**: Schulman et al. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation
- **Entropy Regularization**: Haarnoja et al. (2018). Soft Actor-Critic

## FAQ

**Q: Why binary actions instead of continuous?**
A: Tool usage is inherently discrete (call or skip). Binary actions simplify the policy and improve sample efficiency.

**Q: Can AEPO work with real tool APIs?**
A: Yes! Replace `ToolEnv` with a real environment that calls actual APIs and measures latency/results.

**Q: How to tune for production?**
A: Start with provided defaults, then tune based on your cost/reward tradeoffs. Monitor gate metrics in production.

**Q: What if I have more than 3 tools?**
A: Modify `ToolEnv.num_tools` and provide custom tool configs. The policy scales to any number of tools.

## License

MIT License - See LICENSE file for details.
