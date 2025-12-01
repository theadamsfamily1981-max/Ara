# TF-A-N Triton Inference Server

Production-ready serving infrastructure for TF-A-N with:
- **SSA Runner**: O(N log N) selective sparse attention
- **KV Pager**: File-backed KV cache for 128k+ contexts
- **TTW Hook**: VFE-based alignment monitoring

## Hard Gates

✅ **Performance**: 128k prefill ≥3× faster than dense baseline (RTX 3090)
✅ **Latency**: p99 under SLO (<10s for 128k tokens)
✅ **Cache**: KV hit-rate ≥90%
✅ **Safety**: VFE monitoring with <10ms overhead

## Quick Start

### 1. Start Server

```bash
cd deploy/triton
docker-compose up -d triton-server
```

Wait for server to be ready:
```bash
curl http://localhost:8000/v2/health/ready
```

### 2. Run Inference

```python
import numpy as np
import tritonclient.http as httpclient

# Connect to server
client = httpclient.InferenceServerClient(url='localhost:8000')

# Prepare input
input_ids = np.random.randint(0, 50257, size=(1, 4096), dtype=np.int64)

inputs = [
    httpclient.InferInput("input_ids", input_ids.shape, "INT64")
]
inputs[0].set_data_from_numpy(input_ids)

outputs = [
    httpclient.InferRequestedOutput("output_ids"),
    httpclient.InferRequestedOutput("ssa_stats")
]

# Infer
response = client.infer(
    model_name="tfan_ssa",
    inputs=inputs,
    outputs=outputs
)

# Get results
output_ids = response.as_numpy("output_ids")
stats = response.as_numpy("ssa_stats")

print(f"Generated: {output_ids}")
print(f"Stats: {stats}")
```

### 3. Run Benchmarks

```bash
pip install tritonclient[http]

python deploy/triton/benchmark_ssa.py \
  --url localhost:8000 \
  --model tfan_ssa \
  --seq-lengths 4096 8192 16384 32768 65536 131072 \
  --num-trials 10 \
  --output benchmark_results.json
```

### 4. View Results

```bash
cat benchmark_results.json
```

Example output:
```json
{
  "seq_length": 131072,
  "ssa": {
    "latency_p99_ms": 8234.5,
    "throughput_mean": 15.9,
    "ssa_sparsity_mean": 0.94,
    "kv_hit_rate_mean": 0.92,
    "gates": {
      "speedup_vs_baseline": {"value": 3.2, "pass": true},
      "p99_under_slo": {"value": 8234.5, "pass": true},
      "kv_hit_rate": {"value": 0.92, "pass": true},
      "overall": {"pass": true}
    }
  }
}
```

## Configuration

Edit `deploy/triton/model_repository/tfan_ssa/config.pbtxt`:

```protobuf
parameters [
  {
    key: "k_landmarks"
    value: { string_value: "64" }  # Number of topological landmarks
  },
  {
    key: "local_window"
    value: { string_value: "256" }  # Local attention window
  },
  {
    key: "persistence_threshold"
    value: { string_value: "0.1" }  # Min persistence for landmarks
  },
  {
    key: "vfe_threshold"
    value: { string_value: "0.5" }  # VFE threshold for TTW
  }
]
```

## Architecture

```
┌─────────────────────────────────────────────┐
│             Triton Server                   │
│  ┌───────────────────────────────────────┐  │
│  │        tfan_ssa (Python Backend)      │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │  SSA Runner                     │  │  │
│  │  │  - Topological landmark select  │  │  │
│  │  │  - O(N log N) sparse attention  │  │  │
│  │  │  - Flash attention if available │  │  │
│  │  └─────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │  KV Pager                       │  │  │
│  │  │  - GPU → CPU → Disk tiers       │  │  │
│  │  │  - LRU eviction                 │  │  │
│  │  │  - Prefetching                  │  │  │
│  │  └─────────────────────────────────┘  │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │  TTW Hook                       │  │  │
│  │  │  - VFE monitoring               │  │  │
│  │  │  - Alignment triggers           │  │  │
│  │  └─────────────────────────────────┘  │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## API Endpoints

### HTTP
- **Inference**: `POST http://localhost:8000/v2/models/tfan_ssa/infer`
- **Health**: `GET http://localhost:8000/v2/health/ready`
- **Metrics**: `GET http://localhost:8002/metrics`

### gRPC
- **Inference**: `localhost:8001`

## Monitoring

### Prometheus Metrics

Start Prometheus and Grafana:
```bash
docker-compose up -d prometheus grafana
```

Access Grafana at http://localhost:3000 (admin/admin)

### Available Metrics
- `nv_inference_request_duration_us`: Request latency
- `nv_inference_queue_duration_us`: Queue time
- `nv_inference_compute_duration_us`: Compute time
- `nv_gpu_utilization`: GPU utilization
- `nv_gpu_memory_used_bytes`: GPU memory usage

## Troubleshooting

### Server won't start
```bash
docker-compose logs triton-server
```

### Out of memory
Reduce batch size or enable KV paging:
```protobuf
parameters [
  {
    key: "enable_kv_paging"
    value: { string_value: "true" }
  }
]
```

### Slow inference
Check if GPU is being used:
```bash
nvidia-smi
```

Increase landmarks for more sparsity:
```protobuf
parameters [
  {
    key: "k_landmarks"
    value: { string_value: "128" }  # More landmarks = slower but better quality
  }
]
```

## CI/CD

GitHub Actions workflow tests:
- ✅ Import checks
- ✅ SSA runner functionality
- ✅ KV pager operations
- ✅ TTW hook triggering
- ✅ Triton config validation
- 🔒 Benchmark gates (requires GPU runner)

To enable benchmark gates, add self-hosted GPU runner and uncomment `benchmark-gates` job in `.github/workflows/triton_server.yml`.

## See Also

- [Backend Integration Guide](../../docs/BACKEND_INTEGRATION_GUIDE.md)
- [8GB Dataset Guide](../../docs/8GB_DATASET_GUIDE.md)
- [Testing Guide](../../docs/TESTING_GUIDE.md)
