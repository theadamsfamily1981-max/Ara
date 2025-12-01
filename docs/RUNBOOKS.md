# TF-A-N Runbooks

Operational procedures for common failure scenarios and maintenance tasks.

## Table of Contents
- [Gate Violations](#gate-violations)
- [Performance Degradation](#performance-degradation)
- [System Failures](#system-failures)
- [Maintenance Tasks](#maintenance-tasks)
- [Emergency Procedures](#emergency-procedures)

## Gate Violations

### RB-001: TTW p95 Latency Exceeds 5ms

**Symptoms**:
- Alignment p95 latency > 5ms
- TTW bottleneck in profiling
- Alerts: `tfan_ttw_p95_latency_ms > 5.0`

**Diagnosis**:
```python
# Check TTW metrics
from tfan.mm import align_streams

_, metrics = align_streams(streams)
print(f"p95: {metrics.p95_latency_ms:.2f} ms")
print(f"p50: {metrics.p50_latency_ms:.2f} ms")
print(f"Coverage: {metrics.coverage:.1%}")

# Check trigger frequency
print(f"Total triggers: {sentry.total_triggers}")
print(f"Trigger hits: {sentry.trigger_hits}")
print(f"Hit rate: {sentry.trigger_hits / sentry.total_triggers:.1%}")
```

**Resolution Steps**:

1. **Reduce trigger sensitivity** (if triggers firing too often):
   ```yaml
   ttw:
     triggers:
       vfe_spike_threshold: 0.20  # up from 0.15
       entropy_jump_threshold: 0.30  # up from 0.25
   ```

2. **Reduce max iterations**:
   ```yaml
   ttw:
     max_iter: 30  # down from 50
   ```

3. **Skip low-priority modalities**:
   ```python
   # In custom dataloader/collate
   if 'imu' in streams and priority == 'low':
       del streams['imu']
   ```

4. **Use cached warps more aggressively**:
   ```python
   # Force use of last good warp if budget tight
   if estimated_latency > 4.0:
       aligned = sentry._apply_cached_warp(streams)
   ```

5. **If persistent, profile with**:
   ```bash
   nvprof python train.py --profile-ttw
   # Or use PyTorch profiler
   ```

**Escalation**: If p95 still > 5ms after tuning, consider:
- Hardware upgrade (faster CPU for TTW kernels)
- Reduce stream resolution/frame rate
- Queue for architecture review

---

### RB-002: PGU p95 Latency Exceeds 200ms

**Symptoms**:
- Proof verification slow
- Training throughput degraded
- Alerts: `tfan_pgu_p95_latency_ms > 200.0`

**Diagnosis**:
```python
metrics = pgu.get_metrics()
print(f"p95: {metrics['p95_latency_ms']:.2f} ms")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.1%}")
print(f"Timeout rate: {metrics['timeout_rate']:.1%}")
```

**Resolution Steps**:

1. **Increase cache size**:
   ```yaml
   pgu:
     cache_size: 20000  # up from 10000
   ```

2. **Reduce timeout for faster fallback**:
   ```yaml
   pgu:
     timeout_ms: 100  # down from 120
     fallback_timeout_ms: 150  # down from 180
   ```

3. **Warm cache with common queries**:
   ```python
   # Pre-populate cache before training
   common_payloads = [...]
   for payload in common_payloads:
       pgu.verify_update(payload)
   ```

4. **Switch to soft mode if non-safety-critical**:
   ```yaml
   pgu:
     mode: soft
   ```

5. **Simplify proof rules**:
   ```python
   # Remove expensive rules
   pgu.rules.rules = [r for r in pgu.rules.rules if r['name'] != 'complex_rule']
   ```

**Escalation**: If persistent:
- Review Z3 formulas for complexity
- Consider rule reduction or simplification
- Evaluate moving to approximation for non-critical updates

---

### RB-003: EPR-CV Exceeds 0.15

**Symptoms**:
- Training instability
- Loss oscillations
- Alerts: `tfan_epr_cv > 0.15`

**Diagnosis**:
```python
# Check EPR-CV trend
epr_cv_history = [m['epr_cv'] for m in trainer.metrics_history[-1000:]]
import matplotlib.pyplot as plt
plt.plot(epr_cv_history)
plt.axhline(0.15, color='r', linestyle='--')
plt.show()

# Check gradient stats
grad_variance = trainer._compute_grad_variance()
print(f"Gradient variance: {grad_variance:.4f}")
```

**Resolution Steps**:

1. **Increase PID proportional gain**:
   ```yaml
   fdt:
     pid:
       kp: 0.40  # up from 0.30
   ```

2. **Enable/verify gradient clipping**:
   ```yaml
   fdt:
     grad_clip_norm: 1.0
   ```

3. **Add batch jitter** (reduce variance):
   ```python
   # In dataloader
   import random
   actual_batch = int(batch_size * (1 + random.uniform(-0.1, 0.1)))
   ```

4. **Reduce learning rate temporarily**:
   ```python
   for g in trainer.optimizer.param_groups:
       g['lr'] *= 0.5
   ```

5. **Checkpoint and resume with cooler start**:
   ```python
   trainer.save_checkpoint("emergency_ckpt.pt")
   # Reload with reduced LR/temp
   trainer.load_checkpoint("emergency_ckpt.pt")
   trainer.current_lr *= 0.8
   trainer.current_temperature *= 1.2
   ```

**Escalation**: If EPR-CV > 0.20 sustained:
- Pause training
- Review data distribution for anomalies
- Consider architecture changes (add layer norm, dropout)

---

### RB-004: Topology Gate Failures

**Symptoms**:
- Wasserstein gap > 2% or Cosine < 0.90
- Frequent CAT fallback activations
- Alerts: `tfan_topology_wasserstein_gap > 0.02` or `tfan_topology_cosine < 0.90`

**Diagnosis**:
```python
# Check topology metrics
topo_metrics = topo_reg.compute_landscape(latents, return_diagrams=True)
approx_diagrams = topo_metrics['diagrams']

# Validate against target
passes, metrics = topo_reg.validate_against_exact(
    approx_diagrams[0],
    target_diagrams[0],
)
print(f"Wasserstein: {metrics['wasserstein_distance']:.4f}")
print(f"Cosine: {metrics['cosine_similarity']:.4f}")
```

**Resolution Steps**:

1. **Increase landmark retention**:
   ```yaml
   ssa:
     keep_ratio: 0.40  # up from 0.33
   ```

2. **Smooth latent representations**:
   ```yaml
   model:
     dropout: 0.15  # add dropout
     layer_norm: true  # add layer norm
   ```

3. **Queue for nightly exact PH audit**:
   ```python
   # Add to audit queue
   with open('artifacts/topology/audit_queue.txt', 'a') as f:
       f.write(f"{current_step}\n")
   ```

4. **Adjust topology regularization weight**:
   ```yaml
   topology:
     lambda_topo: 0.05  # down from 0.10 if over-regularizing
   ```

5. **Retrain topology gate thresholds** (if systematic):
   ```python
   # Collect statistics from recent runs
   wass_samples = [...]
   cos_samples = [...]
   new_thresholds = (
       np.percentile(wass_samples, 95),
       np.percentile(cos_samples, 5),
   )
   ```

**Escalation**:
- Run full exact PH analysis on current checkpoint
- Review if data distribution has shifted
- Consider retraining with updated topology targets

---

### RB-005: Memory Scaling α ≥ 1.0

**Symptoms**:
- Sub-linear memory scaling violated
- OOM on longer sequences than expected
- Alerts: `tfan_memory_alpha >= 1.0`

**Diagnosis**:
```bash
# Run memory fit
python scripts/memory_fit.py \
    --seq 1024 2048 4096 8192 16384 \
    --batch 1 \
    --output artifacts/memory/fit.json

# Check alpha
python -c "
import json
data = json.load(open('artifacts/memory/fit.json'))
print(f'Alpha: {data[\"alpha\"]:.3f}')
print(f'Gate: {\"PASS\" if data[\"alpha\"] < 1.0 else \"FAIL\"}')"
```

**Resolution Steps**:

1. **Increase sparsity**:
   ```yaml
   ssa:
     keep_ratio: 0.25  # down from 0.33
   ```

2. **Reduce attention window**:
   ```yaml
   ssa:
     window_size: 96  # down from 128
   ```

3. **Enable gradient checkpointing**:
   ```yaml
   model:
     gradient_checkpointing: true
   ```

4. **Use lower precision for KV cache**:
   ```python
   # In model forward
   k = k.half()  # FP16 instead of FP32
   v = v.half()
   ```

5. **Chunk long sequences**:
   ```python
   # Process in chunks
   chunk_size = 8192
   for i in range(0, seq_len, chunk_size):
       chunk = x[:, i:i+chunk_size]
       output_chunk = model(chunk)
   ```

**Escalation**:
- Review sparse attention implementation for memory leaks
- Profile memory allocations with PyTorch profiler
- Consider architectural changes (e.g., local-only attention)

---

## Performance Degradation

### RB-006: SSA Speedup < 3× at 16k/32k

**Symptoms**:
- Sparse attention slower than expected
- Benchmarks show speedup < 3×
- Training slower than baseline

**Diagnosis**:
```bash
# Run benchmark
python scripts/bench_attention.py \
    --seq 16384 32768 \
    --batch 4 \
    --report artifacts/bench/attention.json

# Check results
python -c "
import json
data = json.load(open('artifacts/bench/attention.json'))
for seq, speedup in zip(data['seq_lengths'], data['speedups']):
    if seq in [16384, 32768]:
        print(f'{seq}: {speedup:.2f}× {\"✓\" if speedup >= 3.0 else \"✗\"}')"
```

**Resolution Steps**:

1. **Verify CUDA compilation**:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"PyTorch compiled with CUDA: {torch.backends.cudnn.enabled}")
   ```

2. **Check if Flash Attention is used**:
   ```python
   try:
       import flash_attn
       print("Flash Attention available")
   except:
       print("Flash Attention NOT available - install for speedup")
   ```

3. **Profile attention kernel**:
   ```python
   with torch.profiler.profile() as prof:
       output, _ = attn(x)
   print(prof.key_averages().table(sort_by="cuda_time_total"))
   ```

4. **Optimize mask computation**:
   ```python
   # Pre-compute masks and cache
   mask_cache = {}
   def get_mask(seq_len):
       if seq_len not in mask_cache:
           mask_cache[seq_len] = build_sparse_mask(seq_len, ...)
       return mask_cache[seq_len]
   ```

5. **Vectorize landmark selection**:
   ```python
   # Ensure TLS uses vectorized operations
   # Avoid Python loops in landmark selection
   ```

**Escalation**:
- Deep dive into kernel profiling
- Consider custom CUDA kernels for critical paths
- Benchmark against other sparse attention libraries

---

## System Failures

### RB-007: CUDA Out of Memory

**Symptoms**:
- `RuntimeError: CUDA out of memory`
- Training crashes
- Inconsistent batch processing

**Immediate Actions**:

1. **Clear CUDA cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

2. **Reduce batch size**:
   ```yaml
   batch_size: 4  # down from 8
   gradient_accumulation_steps: 8  # maintain effective batch
   ```

3. **Enable mixed precision**:
   ```yaml
   mixed_precision: true
   ```

4. **Reduce sequence length**:
   ```yaml
   max_seq_len: 16384  # down from 32768
   ```

**Diagnostic Steps**:

```python
# Check memory usage
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Profile memory
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    profile_memory=True,
) as prof:
    output = model(batch)

print(prof.key_averages().table(sort_by="cuda_memory_usage"))
```

**Resolution**:
- See RB-005 for memory scaling fixes
- Consider gradient checkpointing
- Use FP16 or FP8 where safe

---

### RB-008: PGU Deadlock

**Symptoms**:
- Training hangs
- PGU verification never returns
- Z3 solver stuck

**Immediate Actions**:

1. **Kill training process**:
   ```bash
   pkill -f train.py
   ```

2. **Check for Z3 processes**:
   ```bash
   ps aux | grep z3
   pkill z3  # if hung
   ```

**Resolution**:

1. **Reduce Z3 timeout**:
   ```yaml
   pgu:
     timeout_ms: 80  # down from 120
   ```

2. **Enable aggressive timeout**:
   ```python
   # In PGU code, add hard timeout wrapper
   import signal
   def timeout_handler(signum, frame):
       raise TimeoutError("PGU hard timeout")

   signal.signal(signal.SIGALRM, timeout_handler)
   signal.alarm(1)  # 1 second hard limit
   try:
       result = solver.check()
   finally:
       signal.alarm(0)
   ```

3. **Switch to soft mode temporarily**:
   ```yaml
   pgu:
     mode: soft
   ```

4. **Simplify rules**:
   ```python
   # Remove complex constraints
   pgu.rules.rules = pgu.rules.rules[:5]  # Keep only top 5
   ```

**Prevention**:
- Set `fallback_timeout_ms` conservatively
- Monitor PGU latency metrics
- Test rules on synthetic data before production

---

## Maintenance Tasks

### RB-009: Nightly Topology Audit

**Schedule**: Daily at 05:00 UTC

**Procedure**:

```bash
#!/bin/bash
# nightly_topo_audit.sh

# 1. Collect samples
python tools/collect_samples.py \
    --checkpoint checkpoints/latest.pt \
    --n-samples 5000 \
    --output artifacts/topology/samples.npz

# 2. Run exact PH
python tools/nightly_ph_check.py \
    --samples artifacts/topology/samples.npz \
    --max-time-min 20 \
    --output artifacts/topology/audit_$(date +%Y%m%d).json

# 3. Validate gates
python tools/check_gates.py \
    --audit artifacts/topology/audit_$(date +%Y%m%d).json \
    --thresholds '{"wasserstein": 0.02, "cosine": 0.90}'

# 4. Alert if failed
if [ $? -ne 0 ]; then
    curl -X POST https://alerts.example.com/topology_fail \
        -d "date=$(date)" \
        -d "file=artifacts/topology/audit_$(date +%Y%m%d).json"
fi
```

**Expected Duration**: 15-20 minutes

**Alerts**: If Wasserstein > 2% or Cosine < 0.90

---

### RB-010: Weekly Benchmark Run

**Schedule**: Weekly on Monday 06:00 UTC

**Procedure**:

```bash
#!/bin/bash
# weekly_bench.sh

# 1. Attention benchmark
python scripts/bench_attention.py \
    --seq 8192 16384 32768 \
    --batch 4 \
    --n-runs 50 \
    --report artifacts/bench/attention_$(date +%Y%m%d).json

# 2. Memory scaling
python scripts/memory_fit.py \
    --seq 1024 2048 4096 8192 16384 32768 \
    --batch 1 \
    --report artifacts/memory/fit_$(date +%Y%m%d).json

# 3. TTW micro-bench
python scripts/bench_ttw.py \
    --n-samples 1000 \
    --report artifacts/bench/ttw_$(date +%Y%m%d).json

# 4. PGU latency bench
python scripts/bench_pgu.py \
    --n-queries 10000 \
    --report artifacts/bench/pgu_$(date +%Y%m%d).json

# 5. Validate all gates
python scripts/check_gates.py \
    --bench artifacts/bench \
    --memory artifacts/memory \
    --weekly

# 6. Generate report
python tools/generate_report.py \
    --bench-dir artifacts/bench \
    --output reports/weekly_$(date +%Y%m%d).pdf
```

**Expected Duration**: 1-2 hours

---

### RB-011: Cache Cycling

**Trigger**: Every 1000 batches

**Procedure**:

```python
# In trainer loop
if trainer.current_step % 1000 == 0:
    # Cycle PGU cache
    trainer.pgu.cache.cycle()

    # Log cache stats
    print(f"PGU cache hit rate: {trainer.pgu.cache.hit_rate():.1%}")
    print(f"Cache size: {len(trainer.pgu.cache.cache)}")

    # Reset hit/miss counters
    trainer.pgu.cache.reset_stats()
```

---

## Emergency Procedures

### RB-012: Emergency Stop

**Trigger**: Critical failure, data corruption, security incident

**Procedure**:

1. **Immediate shutdown**:
   ```bash
   # Stop all training processes
   pkill -9 -f train.py

   # Stop monitoring
   systemctl stop prometheus grafana

   # Mark emergency in logs
   echo "EMERGENCY STOP $(date)" >> /var/log/tfan/emergency.log
   ```

2. **Preserve state**:
   ```bash
   # Save current checkpoint
   cp checkpoints/latest.pt checkpoints/emergency_$(date +%Y%m%d_%H%M%S).pt

   # Archive logs
   tar czf logs/emergency_$(date +%Y%m%d_%H%M%S).tar.gz logs/

   # Snapshot artifacts
   cp -r artifacts artifacts_emergency_$(date +%Y%m%d_%H%M%S)
   ```

3. **Notify team**:
   ```bash
   curl -X POST https://alerts.example.com/emergency \
       -d "severity=critical" \
       -d "message=TF-A-N emergency stop initiated" \
       -d "timestamp=$(date)"
   ```

4. **Investigate**:
   - Review logs in `logs/emergency_*.tar.gz`
   - Check GPU health: `nvidia-smi`
   - Verify data integrity
   - Security scan if applicable

---

### RB-013: Rollback to Last Good Checkpoint

**Trigger**: Persistent gate failures, model divergence

**Procedure**:

1. **Identify last good checkpoint**:
   ```bash
   # List recent checkpoints with gates
   python tools/list_checkpoints.py \
       --dir checkpoints/ \
       --show-gates \
       --sort-by step

   # Find last passing all gates
   LAST_GOOD=$(python tools/find_last_good.py --dir checkpoints/)
   echo "Last good: $LAST_GOOD"
   ```

2. **Backup current state**:
   ```bash
   cp checkpoints/latest.pt checkpoints/backup_$(date +%Y%m%d_%H%M%S).pt
   ```

3. **Rollback**:
   ```python
   # In training script
   trainer.load_checkpoint(last_good_checkpoint)

   # Verify gates
   passes, results = trainer.validate_gates()
   assert passes, f"Rollback checkpoint also failing: {results}"

   # Resume training with conservative settings
   trainer.base_lr *= 0.8
   trainer.base_temperature *= 1.1
   ```

4. **Monitor closely**:
   - Check all gates every 100 steps
   - Enable verbose logging
   - Reduce checkpoint interval to 500 steps

---

## Contact & Escalation

- **On-call engineer**: `oncall@tfan.example.com`
- **Slack channel**: `#tfan-alerts`
- **Incident tracker**: `https://issues.example.com/tfan`

### Severity Levels

- **P0 (Critical)**: System down, data loss, security breach → Immediate response
- **P1 (High)**: Multiple gate failures, training blocked → 1 hour response
- **P2 (Medium)**: Single gate failure, degraded performance → 4 hour response
- **P3 (Low)**: Warnings, optimization opportunities → Next business day
