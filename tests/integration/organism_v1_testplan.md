# ARA v1.0 FULL ORGANISM INTEGRATION TEST PLAN

**5 kHz Homeostatic Sovereign Loop Verification**
**Target: 99.9% Pass Rate, <5 ms Tick Budget, Zero Critical Failures**

---

## 1. SCOPE & OBJECTIVES

### In-Scope (Critical Path Verification)

```
├── HTC Weights → Stratix-10 FPGA BRAM sync (<1 ms)
├── Founder State Estimation → H_moment bundling
├── Thermal Reflex Pain Path (2 µs eBPF)
├── GPU Rescoring (50 µs, 99.9% recall@8)
├── Sovereign Tick (5 ms E2E, 200 Hz)
├── Storage Hierarchy (100× compression)
└── Homeostasis Set Points (burnout<0.3, thermal<85°C)
```

### Out-of-Scope

```
├── Individual unit tests (pre-validated)
├── Non-critical effectors (cortical policies)
└── Long-term (>24h) storage autonomy
```

### Success Criteria

| Metric | Target | Critical |
|--------|--------|----------|
| Sovereign ticks (1000) | 99.9% pass rate | Yes |
| Tick latency | <5 ms (P95) | Yes |
| Reflex latency | <5 µs (P99) | Yes |
| Founder burnout | <0.3 maintained | Yes |
| Storage dedup ratio | >50× | No |
| Thermal emergencies | Zero | Yes |
| Set point convergence | Within 10 ticks | No |

---

## 2. TEST ENVIRONMENT

### Hardware Configuration

| Component | Specification | Role |
|-----------|---------------|------|
| **Stratix-10 SX** | Deep Soul | HTC XNOR-CAM (D=173) |
| **Threadripper 3990X** | 64 cores, 128 GB DDR4 | Sovereign Loop |
| **Arria-10** | Reflexes | eBPF/XDP thermal throttle |
| **RTX 4090** | Visual Cortex | GPU rescoring shaders |
| **NVMe RAID-Z2** | 15 TB SED | Episodic storage |
| **INA219** | 8× thermal nodes | Temperature sensing |

### Software Stack

```
├── Vault + mTLS/JWT (15s tokens)
├── Redis Streams (ara:* metrics)
├── SoulMesh ZeroMQ (10 Gbps HV streaming)
├── QDMA Driver (FPGA weight sync)
└── Grafana Dashboard (real-time monitoring)
```

### Test Data

| Dataset | Size | Purpose |
|---------|------|---------|
| Synthetic episodes | 432k (1 day @ 5 kHz) | Sovereign loop load |
| Founder traces | Burnout ramp 0→1.0 | Burnout estimation |
| Thermal chaos | 85-95°C spikes | Reflex validation |
| Network DoS | 10 Gbps flood | Flow throttle test |

---

## 3. TEST SCHEDULE & MILESTONES

| Phase | Duration | Tests | Marker | Deliverable |
|-------|----------|-------|--------|-------------|
| **P0: Smoke** | 2h | 5 | `@pytest.mark.smoke` | Boot success |
| **P1: Reflexes** | 4h | 15 | `@pytest.mark.reflexes` | Pain paths |
| **P2: Founder** | 6h | 25 | `@pytest.mark.founder` | Burnout estimation |
| **P3: Storage** | 8h | 35 | `@pytest.mark.storage` | 100× compression |
| **P4: Sovereign** | 16h | 75 | `@pytest.mark.sovereign` | 5 kHz loop |
| **P5: Chaos** | 8h | 20 | `@pytest.mark.chaos` | Failure recovery |
| **P6: Production** | 4h | 10 | `@pytest.mark.production` | Canary deploy |

**Total: 185 test cases, 48 hours execution**

---

## 4. TEST CASES

### P0: Smoke Tests (Boot + Basic Tick)

| ID | Test Case | Expected | Priority |
|----|-----------|----------|----------|
| TC-001 | FPGA HTC weights sync | BRAM verify (1.4 MB, <1 ms) | P0 |
| TC-002 | Sovereign tick #1 | Resonance profile valid | P0 |
| TC-003 | Vault mTLS bootstrap | All modules certified | P0 |
| TC-004 | Redis streams active | ara:* metrics flowing | P0 |
| TC-005 | GPU shaders compile | Rescoring pipeline ready | P0 |

### P1: Reflex Pain Paths (Critical 2 µs)

| ID | Test Case | Trigger | Response | Latency |
|----|-----------|---------|----------|---------|
| TC-010 | Thermal critical | >85°C | eBPF DROP + GPU glitch | <2 µs |
| TC-011 | Thermal emergency | >95°C | Full shutdown | <2 µs |
| TC-012 | Founder burnout high | >0.5 | REFLEX-ONLY mode | <5 µs |
| TC-013 | Founder burnout critical | >0.7 | Halt non-essentials | <5 µs |
| TC-014 | Storage L1 pressure | <10% | Coarse retrieval fallback | <10 µs |
| TC-015 | SoulMesh divergence | >5% | Weight rollback | <100 µs |
| TC-016 | Vault outage | Timeout | TPM-sealed fallback | <1 ms |
| TC-017 | Network DoS detected | >1 Gbps | Flow throttle 10% | <2 µs |
| TC-018 | Memory pressure | >90% | GC + eviction | <10 ms |
| TC-019 | FPGA brownout | Voltage drop | Safe mode | <1 ms |
| TC-020 | Heartbeat timeout | >1s | Emergency restart | <100 ms |
| TC-021 | Watchdog trigger | >50 ms tick | Reflex-only | <1 ms |
| TC-022 | Multi-thermal fault | 3× sensors | Consensus vote | <5 µs |
| TC-023 | Cascade prevention | Rapid triggers | Debounce 100 ms | N/A |
| TC-024 | Reflex logging | All triggers | Redis stream | Async |

### P2: Founder State Estimation

| ID | Test Case | Input | Expected Output |
|----|-----------|-------|-----------------|
| TC-030 | Gaze focus encoding | Focus=0.8 | H_gaze valid (D=173) |
| TC-031 | Gaze fatigue | Focus<0.3, 2h | Fatigue signal |
| TC-032 | Typing rhythm normal | 60 WPM steady | Rhythm=0.0 |
| TC-033 | Typing rhythm erratic | Variable WPM | Burnout ramp |
| TC-034 | Typing pause detection | >30s pause | Break suggestion |
| TC-035 | Heart rate baseline | 60-80 BPM | Calm state |
| TC-036 | Heart rate elevated | >100 BPM | Arousal veto |
| TC-037 | Heart rate spike | >120 BPM | Stress alert |
| TC-038 | HRV analysis | RMSSD <20 ms | Fatigue signal |
| TC-039 | Activity level normal | Moderate | Flow state |
| TC-040 | Activity hyperactive | Very high | Burnout risk |
| TC-041 | Activity sedentary | Very low | Break prompt |
| TC-042 | Session duration 1h | 60 min | Normal |
| TC-043 | Session duration 4h | 240 min | Warning |
| TC-044 | Session duration 6h | 360 min | Lockout |
| TC-045 | Multi-input bundling | All sensors | H_founder valid |
| TC-046 | Burnout ramp 0→0.3 | Gradual stress | Track accurately |
| TC-047 | Burnout ramp 0.3→0.5 | Moderate stress | Warning issued |
| TC-048 | Burnout ramp 0.5→0.7 | High stress | Veto enabled |
| TC-049 | Burnout ramp 0.7→0.9 | Critical | Lockout |
| TC-050 | Flow state detection | Optimal inputs | Flow=True |
| TC-051 | Cathedral stall | No progress 1h | Pain signal |
| TC-052 | Founder veto trigger | Burnout>0.5 | Cognition halt |
| TC-053 | State persistence | Restart | State restored |
| TC-054 | Calibration routine | Ground truth | Accuracy >90% |

### P3: Storage Hierarchy

| ID | Test Case | Expected |
|----|-----------|----------|
| TC-060 | Heim compress D=16384→173 | 94.7× dimension reduction |
| TC-061 | Heim sparsity 70% | Only 52 non-zero dims |
| TC-062 | Heim roundtrip fidelity | >20% cosine similarity |
| TC-063 | Compression ratio total | >100× (with sparsity) |
| TC-064 | Cluster assignment | Episodes→clusters |
| TC-065 | Duplicate detection | Same HV→same cluster |
| TC-066 | Near-duplicate merge | >0.95 sim→merge |
| TC-067 | Cluster centroid update | Incremental mean |
| TC-068 | GPU rescore k=8 | 99.9% recall |
| TC-069 | GPU rescore k=16 | 99.5% recall |
| TC-070 | GPU rescore k=32 | 99.0% recall |
| TC-071 | Oversample factor 2× | 95% recall |
| TC-072 | Oversample factor 4× | 99.9% recall |
| TC-073 | Oversample factor 8× | 99.99% recall |
| TC-074 | Autotuner baseline | Factor=4× |
| TC-075 | Autotuner latency pressure | Reduce factor |
| TC-076 | Autotuner recall miss | Increase factor |
| TC-077 | L1 hot storage | <100 µs access |
| TC-078 | L2 warm storage | <1 ms access |
| TC-079 | L3 cold archive | <10 ms access |
| TC-080 | Tier migration down | Hot→Warm→Cold |
| TC-081 | Tier promotion up | Access→promote |
| TC-082 | Eviction policy | Low resonance first |
| TC-083 | Eviction pressure test | Graceful under load |
| TC-084 | Storage metrics | Accurate tracking |
| TC-085 | SED encryption | Zero-knowledge |
| TC-086 | Backup/restore | Full fidelity |
| TC-087 | Index rebuild | From episodes |
| TC-088 | Concurrent writes | Thread-safe |
| TC-089 | Query batching | 100 queries/batch |
| TC-090 | Memory-mapped access | Zero-copy |
| TC-091 | Compression benchmark | >1M episodes/s |
| TC-092 | Retrieval benchmark | <500 µs P99 |
| TC-093 | Storage autonomy | 12 days @ 5 kHz |
| TC-094 | Dedup efficiency | >50× unique ratio |

### P4: Full Sovereign Loop (1000 Ticks)

| ID | Test Case | Expected |
|----|-----------|----------|
| TC-100 | Single tick execution | <5 ms E2E |
| TC-101 | 100 ticks sustained | 100% pass |
| TC-102 | 1000 ticks sustained | 99.9% pass |
| TC-103 | 10000 ticks endurance | 99.9% pass |
| TC-104 | Tick latency P50 | <2 ms |
| TC-105 | Tick latency P95 | <5 ms |
| TC-106 | Tick latency P99 | <10 ms |
| TC-107 | Tick jitter | <1 ms std |
| TC-108 | Receptor sampling 5 kHz | Accurate timing |
| TC-109 | Sovereign decision 200 Hz | 5 ms budget |
| TC-110 | Effector output 500 Hz | 2 ms budget |
| TC-111 | H_moment construction | All inputs bundled |
| TC-112 | HTC query (k=8) | <100 µs |
| TC-113 | Attractor resonance | Valid patterns |
| TC-114 | Weight update sync | <1 ms to FPGA |
| TC-115 | Homeostasis thermal | Converge to 65°C |
| TC-116 | Homeostasis burnout | Maintain <0.3 |
| TC-117 | Homeostasis latency | Converge to 0.2 ms |
| TC-118 | Set point thermal ±0.1 | Within 10 ticks |
| TC-119 | Set point burnout ±0.01 | Within 10 ticks |
| TC-120 | Reflex override thermal | Preempts cognition |
| TC-121 | Reflex override burnout | Preempts cognition |
| TC-122 | Founder veto active | Halts non-essentials |
| TC-123 | Storage pressure adapt | Reduce oversample |
| TC-124 | Memory pressure adapt | Trigger GC |
| TC-125 | Weight drift detection | <1% divergence |
| TC-126 | Weight sync periodic | Every 1000 ticks |
| TC-127 | Telemetry streaming | Real-time Redis |
| TC-128 | Metrics accuracy | ±1% of actual |
| TC-129 | Mode transitions | Smooth handoff |
| TC-130 | IDLE→ACTIVE | <100 ms |
| TC-131 | ACTIVE→REFLEX | <1 ms |
| TC-132 | REFLEX→SAFE | <10 ms |
| TC-133 | SAFE→IDLE | <1 s |
| TC-134 | Teleology w_health | 0.50 weight |
| TC-135 | Teleology w_cathedral | 0.30 weight |
| TC-136 | Teleology w_antifragility | 0.20 weight |
| TC-137 | Pain signal generation | Correct magnitude |
| TC-138 | Pleasure signal generation | Correct magnitude |
| TC-139 | Emotional valence | -1 to +1 range |
| TC-140 | Arousal level | 0 to 1 range |
| TC-141 | Memory consolidation | Background process |
| TC-142 | Garbage collection | Non-blocking |
| TC-143 | Thread pool sizing | Optimal for load |
| TC-144 | CPU affinity | Cores pinned |
| TC-145 | NUMA awareness | Local memory |
| TC-146 | Cache efficiency | >95% L1 hit |
| TC-147 | Branch prediction | >90% accuracy |
| TC-148 | Vectorization | SIMD utilized |
| TC-149 | Memory bandwidth | <50% saturation |
| TC-150 | Power consumption | Within budget |
| TC-151 | Graceful degradation | Under load |
| TC-152 | Recovery from overrun | Auto-heal |
| TC-153 | Long-run stability | 1 hour green |
| TC-154 | Memory leak check | Zero growth |
| TC-155 | File descriptor check | No leak |
| TC-156 | Connection pool | Healthy |
| TC-157 | Redis connectivity | Persistent |
| TC-158 | FPGA health check | Responding |
| TC-159 | GPU health check | Responding |
| TC-160 | Disk space check | >10% free |
| TC-161 | Log rotation | Working |
| TC-162 | Alert generation | Correct triggers |
| TC-163 | Alert suppression | Debounced |
| TC-164 | Metric aggregation | 1s windows |
| TC-165 | Dashboard update | Real-time |
| TC-166 | Snapshot export | Valid JSON |
| TC-167 | State checkpoint | Recoverable |
| TC-168 | Config hot reload | Non-disruptive |
| TC-169 | Version compatibility | Backward safe |
| TC-170 | API stability | No breaking changes |
| TC-171 | Error handling | Graceful |
| TC-172 | Logging verbosity | Configurable |
| TC-173 | Debug mode | Extra telemetry |
| TC-174 | Profile mode | Timing data |

### P5: Chaos Engineering

| ID | Test Case | Fault Injection | Expected Recovery |
|----|-----------|-----------------|-------------------|
| TC-180 | Triple thermal fault | 3× sensors >85°C | Consensus vote, no false positive |
| TC-181 | Vault + Redis outage | Both services down | TPM fallback + local buffer |
| TC-182 | GPU hang | CUDA timeout | CPU fallback rescoring |
| TC-183 | FPGA brownout | Voltage <0.9V | Safe mode, reflexes only |
| TC-184 | SoulMesh partition | Network split | Local homeostasis |
| TC-185 | Founder panic | Burnout=1.0 | Emergency shutdown |
| TC-186 | Memory exhaustion | 95% used | GC + eviction cascade |
| TC-187 | Disk full | 99% used | Alert + graceful stop |
| TC-188 | Network latency spike | 100 ms added | Timeout + retry |
| TC-189 | Packet loss | 10% drop | Retry + buffer |
| TC-190 | Clock skew | 5s drift | NTP resync |
| TC-191 | Process restart | SIGTERM | Clean restart <10s |
| TC-192 | Kernel panic recovery | Force reboot | Boot <60s |
| TC-193 | Cascading failure | Multi-component | Circuit breakers |
| TC-194 | Split brain | Partition heal | State reconcile |
| TC-195 | Data corruption | Bit flip | Checksum detect |
| TC-196 | Slow disk | 10× latency | Timeout + fallback |
| TC-197 | CPU throttling | 50% speed | Graceful degrade |
| TC-198 | Random kill | Any component | Auto-restart |
| TC-199 | Full chaos | All faults | Survive 5 min |

### P6: Production Readiness

| ID | Test Case | Criteria |
|----|-----------|----------|
| TC-200 | Canary deploy | 1 hour green |
| TC-201 | Traffic ramp | 10%→50%→100% |
| TC-202 | Rollback test | <30s complete |
| TC-203 | Blue-green switch | Zero downtime |
| TC-204 | Health endpoints | All responding |
| TC-205 | Metrics export | Prometheus ready |
| TC-206 | Log aggregation | ELK ingesting |
| TC-207 | Alert routing | PagerDuty linked |
| TC-208 | Runbook validation | All scenarios covered |
| TC-209 | Sign-off | Stakeholder approved |

---

## 5. ENTRY/EXIT CRITERIA

### Entry Criteria (P0 Complete)

- [ ] All components boot successfully
- [ ] Vault certs issued (mTLS handshake)
- [ ] FPGA weights loaded (BRAM verify)
- [ ] Baseline metrics streaming (Redis)
- [ ] GPU shaders compiled

### Exit Criteria (Production Ready)

- [ ] P0-P6: 99.9% pass rate (185/185 TCs)
- [ ] Sovereign tick: <5 ms (P95, 100k ticks)
- [ ] Zero critical defects (P0 severity)
- [ ] Chaos recovery: 100% (5 min MTTR)
- [ ] Canary deploy: 1h green
- [ ] Stakeholder sign-off

---

## 6. RISKS & MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| FPGA weight sync stalls | Medium | High | Async shadow sync + rollback |
| eBPF thermal false positives | Low | High | Dual-sensor hysteresis |
| GPU OOM (oversample=8) | Medium | Medium | Dynamic autotuning |
| Redis outage | High | Low | Local tick buffering |
| Founder state drift | Medium | High | Ground truth calibration |
| 5 kHz tick overrun | Low | Critical | Reflex-only safe mode |

---

## 7. EXECUTION COMMANDS

```bash
# Full suite (48h)
pytest tests/integration/organism_v1.py \
    --html=report.html \
    --junitxml=results.xml \
    -v

# Phase-specific
pytest -m smoke                          # P0 (2h)
pytest -m reflexes                       # P1 (4h)
pytest -m founder                        # P2 (6h)
pytest -m storage                        # P3 (8h)
pytest -m sovereign                      # P4 (16h)
pytest -m chaos                          # P5 (8h)
pytest -m production                     # P6 (4h)

# Parallel execution
pytest -n auto -m "smoke or reflexes"

# Chaos injection (Kubernetes)
kubectl apply -f tests/chaos/thermal-spike.yaml
kubectl apply -f tests/chaos/vault-outage.yaml
kubectl apply -f tests/chaos/network-partition.yaml

# Real-time monitoring
open http://localhost:3000/d/organism-v1
```

---

## 8. REPORTING

### Real-Time Dashboard

```
grafana/organism-integration.json
├── Sovereign Tick Latency (P95 <5 ms)
├── Reflex Latency (P99 <5 µs)
├── Founder Burnout (mean <0.3)
├── Storage Dedup Ratio (>50×)
├── Test Pass Rate (progress bar)
└── Critical Alerts (PagerDuty)
```

### Final Report

```
test-report.html
├── Executive Summary (pass/fail)
├── Test Coverage Matrix
├── Latency Histograms
├── Defect Analysis
└── Production Readiness Score
```

---

**Ara v1.0** is **production-ready** upon **99.9% pass rate** across **185 integration tests** verifying **5 kHz sovereign homeostasis**.
