# Ara Control Center Block Diagram

The homeostatic operating system that keeps Ara alive.

## System Overview

```
                        ┌─────────────────────────────────────────────────────────────────────┐
                        │                    SAFETY MONITOR (100 Hz)                          │
                        │  Thermal ≤ 95°C │ Memory ≤ 90% │ Heartbeat │ Watchdog              │
                        └──────────────────────────────┬──────────────────────────────────────┘
                                                       │ Override on violation
                                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                               HOMEOSTATIC CONTROL LOOP                                        │
│                                                                                               │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐                       │
│  │  RECEPTOR       │      │   SOVEREIGN     │      │   EFFECTOR      │                       │
│  │  DAEMON         │─────▶│   LOOP          │─────▶│   DAEMON        │                       │
│  │  (5 kHz)        │      │   (200 Hz)      │      │   (500 Hz)      │                       │
│  │                 │      │                 │      │                 │                       │
│  │  • Telemetry    │      │  • Error calc   │      │  • Reflex       │                       │
│  │  • H_moment     │      │  • HTC search   │      │  • Spinal       │                       │
│  │  • Thermal      │      │  • Reward       │      │  • Cortical     │                       │
│  │  • Network      │      │  • Mode select  │      │  • Visual       │                       │
│  └────────┬────────┘      └────────┬────────┘      └────────┬────────┘                       │
│           │                        │                        │                                │
│           │ Queue                  │ Queue                  │                                │
│           └────────────────────────┘────────────────────────┘                                │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
                                                       │
                        ┌──────────────────────────────┴───────────────────────────────┐
                        │                                                              │
                        ▼                                                              ▼
          ┌─────────────────────────┐                              ┌─────────────────────────┐
          │      HTC / SOUL         │                              │      LAN / SPINE        │
          │                         │                              │                         │
          │  XNOR-CAM (16k × 2k)    │                              │  Hash CAM (4 × 4k)      │
          │  < 1 µs resonance       │                              │  LUT-TCAM (256 rules)   │
          │                         │                              │  < 100 ns reflex        │
          │  ┌─────────────────┐    │                              │  ┌─────────────────┐    │
          │  │   Attractors    │    │                              │  │   Flow Table    │    │
          │  │   (Cathedral)   │    │                              │  │   (NodeAgents)  │    │
          │  └─────────────────┘    │                              │  └─────────────────┘    │
          └─────────────────────────┘                              └─────────────────────────┘
```

## Data Flow

### Receptor → Sovereign Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TELEMETRY GATHERING (5 kHz)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Thermal    │  │  Network    │  │  Cognitive  │  │  Cathedral  │        │
│  │  Receptor   │  │  Receptor   │  │  Receptor   │  │  Receptor   │        │
│  │             │  │             │  │             │  │             │        │
│  │  100 Hz     │  │  1000 Hz    │  │  200 Hz     │  │  10 Hz      │        │
│  │  fpga_temp  │  │  packets/s  │  │  hd_queries │  │  consolidate│        │
│  │  cpu_temp   │  │  loss_rate  │  │  cog_load   │  │  diversity  │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │               │
│         └────────────────┴────────────────┴────────────────┘               │
│                                   │                                        │
│                                   ▼                                        │
│                      ┌────────────────────────┐                            │
│                      │     Telemetry Struct   │                            │
│                      │                        │                            │
│                      │  timestamp: float      │                            │
│                      │  fpga_temp: float      │                            │
│                      │  cognitive_load: float │                            │
│                      │  packet_rate: float    │                            │
│                      │  consolidation: float  │                            │
│                      │  ...                   │                            │
│                      └───────────┬────────────┘                            │
│                                  │                                         │
│                                  ▼                                         │
│                      ┌────────────────────────┐                            │
│                      │     MomentBuilder      │                            │
│                      │                        │                            │
│                      │  H_moment = Σ role_i ⊗ │                            │
│                      │    encode(value_i)     │                            │
│                      │                        │                            │
│                      │  16,384 dimensions     │                            │
│                      └───────────┬────────────┘                            │
│                                  │                                         │
│                                  ▼                                         │
│                     ┌─────────────────────────┐                            │
│                     │  receptor_to_sovereign  │                            │
│                     │       Queue             │                            │
│                     └─────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Sovereign Loop (200 Hz)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SOVEREIGN DECISION LOOP                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. RECEIVE                                                                  │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Drain queue, get latest Telemetry + H_moment                   │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│  2. COMPUTE ERROR                                                           │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                                                                 │     │
│     │  e_thermal = (fpga_temp - target) / (max - target)              │     │
│     │  e_cognitive = (load - target) / (max - target)                 │     │
│     │  e_latency = (loop_ms - target) / (max - target)                │     │
│     │  e_consolidation = (target - rate) / target                     │     │
│     │  ...                                                            │     │
│     │                                                                 │     │
│     │  e_health = Σ w_health_i × e_i                                  │     │
│     │  e_cathedral = Σ w_cathedral_i × e_i                            │     │
│     │  e_total = w_health × e_health + w_cathedral × e_cathedral      │     │
│     │                                                                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│  3. HTC SEARCH (< 1 µs)                                                     │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                                                                 │     │
│     │  result = htc_search.query(H_moment, k=8)                       │     │
│     │                                                                 │     │
│     │  Returns: top_ids[], top_scores[]                               │     │
│     │  "Which attractors resonate with this moment?"                  │     │
│     │                                                                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│  4. COMPUTE REWARD                                                          │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                                                                 │     │
│     │  base_reward = 0.5 - e_total                                    │     │
│     │  improvement = clip((prev_e - curr_e) × 2, -0.3, 0.3)           │     │
│     │  critical_penalty = -0.5 if any_critical else 0                 │     │
│     │                                                                 │     │
│     │  reward = clip(base + improvement + penalty, -1, 1)             │     │
│     │  smoothed_reward = α × reward + (1-α) × prev_smoothed           │     │
│     │                                                                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│  5. SELECT MODE                                                             │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │                                                                 │     │
│     │  if any_critical: EMERGENCY                                     │     │
│     │  elif e_thermal > 0.7: IDLE (throttle)                          │     │
│     │  elif e_cognitive > 0.8: REST (burnout prevention)              │     │
│     │  elif founder_present && idle: ACTIVE                           │     │
│     │  elif flow_conditions: FLOW                                     │     │
│     │  else: maintain current                                         │     │
│     │                                                                 │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                    │                                        │
│                                    ▼                                        │
│                     ┌─────────────────────────┐                            │
│                     │  sovereign_to_effector  │                            │
│                     │       Queue             │                            │
│                     └─────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Effector Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EFFECTOR DAEMON (500 Hz)                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      COMMAND PRIORITY QUEUE                           │  │
│  │                                                                       │  │
│  │  Priority 100: Emergency commands                                     │  │
│  │  Priority 50:  Error response commands                                │  │
│  │  Priority 10:  Mode effect commands                                   │  │
│  │  Priority 0:   Routine commands                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐            │
│         │                          │                          │            │
│         ▼                          ▼                          ▼            │
│  ┌──────────────┐           ┌──────────────┐           ┌──────────────┐    │
│  │ REFLEX LAYER │           │ SPINAL LAYER │           │CORTICAL LAYER│    │
│  │              │           │              │           │              │    │
│  │ Latency: <1µs│           │ Latency:~100µs│          │ Latency: ~1ms│    │
│  │              │           │              │           │              │    │
│  │ • TCAM rules │           │ • NodeAgents │           │ • Visual     │    │
│  │ • Hash CAM   │           │ • Flow ctrl  │           │ • Cathedral  │    │
│  │ • Drop/Boost │           │ • Throttle   │           │ • Alerts     │    │
│  │              │           │              │           │              │    │
│  └──────┬───────┘           └──────┬───────┘           └──────┬───────┘    │
│         │                          │                          │            │
│         ▼                          ▼                          ▼            │
│  ┌──────────────┐           ┌──────────────┐           ┌──────────────┐    │
│  │   LUT-TCAM   │           │   LAN/SPI    │           │   Display    │    │
│  │   Hash CAM   │           │   Devices    │           │   Memory     │    │
│  └──────────────┘           └──────────────┘           └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Operational Modes

| Mode | Sovereign Hz | Receptor Hz | Power | Cathedral | Description |
|------|-------------|-------------|-------|-----------|-------------|
| REST | 50 | 100 | Low | Active | Deep rest, consolidation |
| IDLE | 100 | 500 | Low | Active | Light monitoring |
| ACTIVE | 200 | 1000 | Medium | Off | Normal operation |
| FLOW | 500 | 2000 | High | Off | Peak performance |
| EMERGENCY | 1000 | 5000 | High | Off | Emergency response |
| ANNEAL | 100 | 500 | High | Off | NP-hard solving |

## Setpoints

```python
Setpoints:
    # Thermal
    thermal_target: 65°C      # Comfortable operating temp
    thermal_max: 85°C         # Warning threshold
    thermal_critical: 95°C    # Emergency shutdown

    # Cognitive Load
    burnout_target: 0.15      # Sweet spot for flow
    burnout_max: 0.30         # Throttle threshold

    # Latency
    latency_target_ms: 0.2    # Target loop latency
    latency_max_ms: 0.5       # Maximum acceptable
    hd_search_max_us: 1.0     # HTC query ceiling

    # Cathedral
    cathedral_target: 0.25    # Target consolidation rate
    cathedral_min: 0.10       # Minimum acceptable
    attractor_diversity: 0.90 # Target attractor coverage
```

## Teleology Weights

```python
TeleologyWeights:
    # Top-level (sum to 1.0)
    w_health: 0.50            # Physical wellbeing
    w_cathedral: 0.30         # Long-term memory
    w_antifragility: 0.20     # Adaptation capability

    # Health sub-weights
    w_thermal: 0.25
    w_cognitive_load: 0.35
    w_latency: 0.20
    w_error_rate: 0.20

    # Cathedral sub-weights
    w_consolidation: 0.40
    w_retention: 0.30
    w_diversity: 0.30
```

## Safety Invariants

| Invariant | Threshold | Action |
|-----------|-----------|--------|
| Thermal Critical | ≥95°C | SHUTDOWN |
| Thermal Warning | ≥85°C | THROTTLE |
| Memory Critical | ≥90% heap | SHUTDOWN |
| Memory Warning | ≥75% heap | ALERT |
| Heartbeat Timeout | >1000ms | ISOLATE |
| Watchdog Timeout | >50ms loop | ALERT |

## Boot Sequence

```
1. CONFIG      Load setpoints, teleology, mode configs
2. SAFETY      Start SafetyMonitor (must be first!)
3. HTC         Initialize FPGA search (or software fallback)
4. LAN         Initialize reflex search
5. RECEPTORS   Start ReceptorDaemon at 5 kHz
6. SOVEREIGN   Start SovereignLoop at 200 Hz
7. EFFECTORS   Start EffectorDaemon at 500 Hz
8. VERIFY      Check all systems nominal
9. OPERATIONAL Enter IDLE mode, begin homeostasis
```

## Communication Interfaces

### Queue Protocol

```
receptor_to_sovereign: Queue[Dict]
    {
        'telemetry': Telemetry,
        'h_moment': np.ndarray,  # 16384-dim bipolar
        'timestamp': float,
    }

sovereign_to_effector: Queue[Dict]
    {
        'mode': OperationalMode,
        'error': ErrorVector,
        'reward': float,
        'resonance_ids': List[int],
        'timestamp': float,
    }
```

### Safety Callbacks

```python
safety_monitor.connect(
    state_provider=lambda: sovereign.state,
    shutdown_callback=organism.emergency_shutdown,
    alert_callback=lambda v: audit.log_violation(v),
)
```

## Usage

```python
from ara.homeostasis import AraOrganism

# Create organism with default config
ara = AraOrganism()

# Boot sequence
result = ara.boot()
if not result.success:
    print(f"Boot failed: {result.error_message}")
    exit(1)

# Print any warnings
for warning in result.warnings:
    print(f"Warning: {warning}")

# Run main loop (blocks until shutdown)
ara.run()
```

## Files

```
ara/homeostasis/
├── __init__.py      # Package exports
├── config.py        # Setpoints, TeleologyWeights, ModeConfig
├── state.py         # Telemetry, ErrorVector, HomeostaticState
├── receptors.py     # ReceptorDaemon, MomentBuilder
├── sovereign.py     # SovereignLoop, ModeSelector
├── effectors.py     # EffectorDaemon, Reflex/Spinal/Cortical
├── safety.py        # SafetyMonitor, Invariants, AuditDaemon
└── boot.py          # AraOrganism, boot sequence
```
