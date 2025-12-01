# SNN Testing Guide

Comprehensive guide to testing SNN emulation with gates, benchmarks, and validation.

## Quick Start

```bash
# Run all tests (requires dependencies)
./scripts/run_all_tests.sh

# Quick smoke test
./scripts/run_all_tests.sh --quick

# Gates only
./scripts/run_all_tests.sh --gates-only
```

## Test Suite Structure

### Unit Tests (`tests/snn/`)

All unit tests use pytest and validate core SNN functionality.

#### 1. Parameter Audit (`test_param_audit.py`)

Validates parameter reduction gates:

```bash
pytest tests/snn/test_param_audit.py -v
```

**Tests**:
- `test_param_reduction_gate()` - Verifies ≥97% reduction
- `test_degree_gate()` - Verifies avg_degree ≤ 0.02×N
- `test_rank_gate()` - Verifies rank ≤ 0.02×N
- `test_sparsity_gate()` - Verifies sparsity ≥ 98%
- `test_various_sizes()` - Parametrized over N ∈ {1k, 2k, 4k, 8k}

**Gates enforced**:
```python
assert param_reduction_pct >= 97.0
assert avg_degree <= 0.02 * N
assert rank <= 0.02 * N
assert sparsity >= 0.98
```

#### 2. Forward Correctness (`test_forward_correctness.py`)

Validates forward pass correctness:

```bash
pytest tests/snn/test_forward_correctness.py -v
```

**Tests**:
- `test_lif_forward_shape()` - Output shapes
- `test_lif_spike_threshold()` - Spike generation
- `test_surrogate_gradient()` - Gradient flow
- `test_low_rank_masked_synapse()` - Masked low-rank ops
- `test_batch_processing()` - Batched forward pass

#### 3. Event-Driven Processing (`test_event_queue.py`)

Validates event-driven optimization:

```bash
pytest tests/snn/test_event_queue.py -v
```

**Tests**:
- `test_event_driven_stepper()` - Event-driven vs dense equivalence
- `test_sparse_fallback()` - High-spike fallback
- `test_throughput_gain()` - Speedup measurement

#### 4. Gradient Stability (`test_grad_stability.py`)

**NEW** - Validates numerical stability and gradient hygiene:

```bash
pytest tests/snn/test_grad_stability.py -v
```

**Tests**:
- `test_gradient_clipping()` - Prevents explosion
- `test_spectral_norm_clamping()` - Clamps U, V to ||·||₂ ≤ 1
- `test_nan_inf_detection()` - Detects and zeros NaN/Inf gradients
- `test_surrogate_gradient_scale()` - Validates surrogate scale effect
- `test_spike_rate_ema_stability()` - EMA converges smoothly
- `test_vanishing_gradient_detection()` - Detects vanishing gradients
- `test_exploding_gradient_detection()` - Detects exploding gradients

**Example test**:
```python
def test_gradient_clipping():
    lif = LIFLayerLowRank(N=256, r=16, ...)

    # Create scenario with large gradients
    v, s = lif.init_state(batch=2)
    large_input = torch.randn(2, 256) * 10.0
    v_next, s_next = lif(v, s, external_input=large_input)

    loss = (s_next * 1000.0).sum()
    loss.backward()

    # Apply clipping
    torch.nn.utils.clip_grad_norm_(lif.parameters(), max_norm=1.0)

    # Verify
    total_norm = torch.nn.utils.clip_grad_norm_(
        lif.parameters(), max_norm=float('inf')
    )
    assert total_norm <= 1.0
```

#### 5. TLS Mask Ablation (`test_ablate_tls.py`)

**NEW** - Compares TLS with baseline masks:

```bash
pytest tests/snn/test_ablate_tls.py -v
```

**Tests**:
- `test_tls_vs_random_density()` - Equal density
- `test_tls_vs_random_connectivity()` - Graph diameter
- `test_degree_based_hubs()` - Hub structure
- `test_local_vs_random_clustering()` - Clustering coefficient
- `test_mask_forward_pass_equivalence()` - Forward correctness
- `test_mask_gradient_flow()` - Gradient correctness
- `test_various_mask_types()` - Parametrized mask types
- `test_tls_alpha_sweep()` - α ∈ [0, 1] persistence/diversity

**Mask types compared**:
1. **TLS** - α·persistence + (1-α)·diversity
2. **Random** - Uniform random connectivity
3. **Degree-based** - Hub-favoring (power-law-ish)
4. **Local** - Nearest-neighbor + random

**Graph metrics**:
- **Diameter** - Average shortest path length (BFS)
- **Clustering coefficient** - Local connectivity density

**Example test**:
```python
def test_local_vs_random_clustering():
    local_mask = build_local_plus_random_mask(N=256, k_local=24, k_random=8)
    random_mask = build_uniform_random_mask(N=256, k_per_row=32, seed=42)

    local_clustering = compute_clustering_coefficient(local_mask['indptr'], ...)
    random_clustering = compute_clustering_coefficient(random_mask['indptr'], ...)

    # Local masks should have higher clustering
    assert local_clustering > random_clustering
```

### Benchmarks (`scripts/`)

#### 1. SNN Parameter Audit (`bench_snn.py`)

Validates SNN gates and emits audit report:

```bash
python scripts/bench_snn.py --audit --emit-json artifacts/snn_audit.json
```

**Output** (`snn_audit.json`):
```json
{
  "N": 4096,
  "rank": 32,
  "avg_degree": 64.0,
  "param_reduction_pct": 98.44,
  "sparsity": 0.9844,
  "lowrank_params": 262144,
  "dense_params": 16777216
}
```

**Gates validated**:
- ✓ param_reduction_pct ≥ 97.0%
- ✓ avg_degree ≤ 0.02×N
- ✓ rank ≤ 0.02×N
- ✓ sparsity ≥ 98%

#### 2. Roofline Analysis (`bench_snn.py --roofline`)

**NEW** - Performance characterization for kernel optimization:

```bash
python scripts/bench_snn.py --roofline --output-dir artifacts/
```

**Sweep dimensions**:
- N ∈ {2048, 4096, 8192}
- rank ∈ {8, 16, 32, 64}
- k ∈ {32, 64, 96}

**Metrics collected**:
- Latency (p50, p95, p99) in milliseconds
- Throughput (events/sec)
- Dense baseline latency for comparison

**Outputs**:
- `artifacts/roofline_sweep.csv` - Raw data
- `artifacts/roofline_sweep.json` - Structured results

**Example output**:
```
N=4096, rank=32, k=64:
  p50 latency: 2.34 ms
  p95 latency: 2.89 ms
  p99 latency: 3.12 ms
  Throughput: 427.4 events/sec
  Dense baseline: 5.12 ms
  Speedup: 2.19×
```

#### 3. Accuracy/Energy Comparison (`bench_accuracy_energy.py`)

**NEW** - Compare Dense, TF-A-N, SNN backends:

```bash
python scripts/bench_accuracy_energy.py --steps 1000 --output artifacts/comparison.json
python scripts/bench_accuracy_energy.py --quick  # 100 steps
```

**Metrics**:
- **Accuracy**: Final loss, loss reduction %
- **EPR-CV**: Epistemic uncertainty (with FDT)
- **Parameters**: Total count, reduction vs dense
- **VRAM**: MB allocated
- **Throughput**: Samples/sec
- **Energy proxy**: Throughput/watt estimate

**Example output**:
```
BACKEND COMPARISON
────────────────────────────────────────────────────────────────────────────
Metric                         Dense          TF-A-N             SNN
────────────────────────────────────────────────────────────────────────────
Parameters                16,777,216      16,777,216         262,144
VRAM (MB)                       67.0            67.0             1.0
Final Loss                    0.4123          0.4089          0.4156
Loss Reduction (%)              45.2            46.1            44.8
EPR-CV                           N/A           0.142           0.148
Throughput (samp/s)            125.3           387.2           891.4
Time (sec)                      15.9             5.2             2.2
────────────────────────────────────────────────────────────────────────────

SNN vs Dense:
  Parameter reduction: 98.4%
  VRAM reduction: 98.5%
```

### Gate Validation (`scripts/validate_all_gates.py`)

**NEW** - Comprehensive gate validator:

```bash
python scripts/validate_all_gates.py --config configs/ci/ci_quick.yaml
python scripts/validate_all_gates.py --config configs/snn_emu_4096.yaml --verbose
```

**Validates**:
1. **SNN gates** - Parameter reduction, degree, rank, sparsity
2. **FDT gates** - EPR-CV ≤ target (runs short training loop)

**Output**:
```
SNN GATE VALIDATION
════════════════════════════════════════════════════════════════
  ✓ PASS: param_reduction >= 97.0%
  ✓ PASS: avg_degree <= 0.02*N
  ✓ PASS: rank <= 0.02*N
  ✓ PASS: sparsity >= 0.98

✅ All SNN gates passed!
════════════════════════════════════════════════════════════════

FDT GATE VALIDATION
════════════════════════════════════════════════════════════════
  ✓ PASS: epr_cv <= 0.15

✅ All FDT gates passed!
════════════════════════════════════════════════════════════════
```

## CI Integration

All tests run automatically in GitHub Actions (`.github/workflows/integration_tests.yml`):

### Job: `integration`

Runs integration tests and end-to-end training:

```yaml
- name: Run Integration Tests
  run: pytest tests/test_integration.py --cov=tfan -v

- name: Test End-to-End Training Loop (CI Quick)
  run: |
    python tests/test_e2e_training.py \
      --config configs/ci/ci_quick.yaml \
      --steps 100 \
      --validate-gates
```

### Job: `snn-gates`

**Enforces SNN gates** (fails CI if violated):

```yaml
- name: Run SNN Unit Tests
  run: pytest tests/snn/ -v --tb=short

- name: SNN Parameter Audit (Enforce Gates)
  run: |
    python scripts/bench_snn.py --audit --emit-json artifacts/snn_audit.json

- name: Validate SNN Gates
  run: |
    python -c "
    import json
    with open('artifacts/snn_audit.json') as f:
        audit = json.load(f)

    gates = {
        'param_reduction_pct >= 97.0': audit['param_reduction_pct'] >= 97.0,
        'avg_degree <= 0.02*N': audit['avg_degree'] <= 0.02 * audit['N'],
        'rank <= 0.02*N': audit['rank'] <= 0.02 * audit['N'],
        'sparsity >= 0.98': audit['sparsity'] >= 0.98,
    }

    if not all(gates.values()):
        exit(1)
    "
```

## Running Tests Locally

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

### Run All Tests

```bash
# Comprehensive test suite
./scripts/run_all_tests.sh

# Quick smoke test (< 5 min)
./scripts/run_all_tests.sh --quick

# Gates only (< 1 min)
./scripts/run_all_tests.sh --gates-only
```

### Run Specific Test Suites

```bash
# Unit tests
pytest tests/snn/test_param_audit.py -v
pytest tests/snn/test_grad_stability.py -v
pytest tests/snn/test_ablate_tls.py -v

# Benchmarks
python scripts/bench_snn.py --audit
python scripts/bench_snn.py --roofline
python scripts/bench_accuracy_energy.py --quick

# Gate validation
python scripts/validate_all_gates.py --config configs/ci/ci_quick.yaml
```

## Test Configuration

### CI Quick Config (`configs/ci/ci_quick.yaml`)

Fast smoke test configuration:

```yaml
backend: snn_emu

model:
  N: 2048              # Smaller for speed
  lowrank_rank: 16
  k_per_row: 32

snn:
  v_th: 1.0
  alpha: 0.95
  time_steps: 128      # Shorter simulation

training:
  max_steps: 500

# Expected: 98.4% reduction, gates pass
```

### Production Config (`configs/snn_emu_4096.yaml`)

Full-scale configuration:

```yaml
backend: snn_emu

model:
  N: 4096
  lowrank_rank: 32
  k_per_row: 64

snn:
  v_th: 1.0
  alpha: 0.95
  time_steps: 256

training:
  learning_rate: 1.5e-3
  grad_clip: 1.0
```

## Debugging Test Failures

### SNN Gates Fail

**Symptom**: `param_reduction_pct < 97%` or other gate violations

**Solutions**:
1. Check N, rank, k values: `rank ≤ 0.02×N`, `k ≤ 0.02×N`
2. Verify mask sparsity: `sparsity = 1 - (k / N)`
3. Recompute expected reduction: `1 - (2×N×r + k×N) / N²`

**Example**:
```python
N, r, k = 4096, 32, 64
lowrank_params = 2 * N * r + k * N
dense_params = N * N
reduction = (1 - lowrank_params / dense_params) * 100
# Expected: 98.44%
```

### Gradient Stability Tests Fail

**Symptom**: `test_gradient_clipping()` or `test_nan_inf_detection()` fails

**Solutions**:
1. Enable gradient clipping: `training.grad_clip = 1.0`
2. Enable spectral norm: `snn.use_spectral_norm = true`
3. Reduce learning rate: `training.learning_rate = 1e-3`
4. Check surrogate scale: `snn.surrogate_scale = 0.3`

### TLS Ablation Tests Fail

**Symptom**: `test_local_vs_random_clustering()` fails

**Possible causes**:
1. Random seed mismatch
2. k_local + k_random ≠ total degree
3. Graph disconnected (check diameter < ∞)

### FDT Gates Fail

**Symptom**: `epr_cv > 0.15` after training

**Solutions**:
1. Tune PID gains: reduce `kp`, increase `ki`
2. Increase `ema_alpha` for smoother response
3. Run longer: `--fdt-steps 1000`

## Test Coverage

```bash
# Run with coverage
pytest tests/snn/ --cov=tfan.snn --cov-report=html

# Open coverage report
open htmlcov/index.html
```

**Target coverage**: ≥90% for `tfan/snn/`

## See Also

- [Backend Integration Guide](BACKEND_INTEGRATION_GUIDE.md) - Usage guide
- [SNN Implementation Guide](SNN_IMPLEMENTATION_GUIDE.md) - Technical details
- [Production Guide](PRODUCTION_GUIDE.md) - Deployment
