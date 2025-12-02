# Phase 5: Cognitive Synthesis

**Status:** Done, Validated

**Objective:** Wire L7/L8/GUF into a unified cognitive decision loop that predicts its own failure, verifies its own thoughts, and decides when to heal itself vs serve the world.

---

## The Door We Busted

Phase 5 isn't a new layer - it's the **integration** that makes L7/L8/GUF work as one coherent cognitive system:

```
┌─────────────────────────────────────────────────────────────────┐
│                   COGNITIVE SYNTHESIS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  L7 Structural Rate (Ṡ)                                         │
│  "I see my shape changing too fast"                             │
│       │                                                         │
│       ▼                                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              CognitiveSynthesizer                        │   │
│  │                                                          │   │
│  │  • Monitors all signals                                  │   │
│  │  • Computes system mode                                  │   │
│  │  • Routes verification decisions                         │   │
│  │  • Triggers proactive AEPO                              │   │
│  │  • Allocates focus (self vs world)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│       │           │           │                                 │
│       ▼           ▼           ▼                                 │
│  L8 Truth Cert  Mode Trans  GUF Focus                          │
│  "Is this       "Switch to   "Am I good                         │
│   consistent?"   protective"  enough to serve?"                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

| Component | Location | Purpose |
|-----------|----------|---------|
| CognitiveSynthesizer | `tfan/synthesis/__init__.py` | Central integration engine |
| SynthesisState | `tfan/synthesis/__init__.py` | Unified health view |
| CockpitStatus | `tfan/synthesis/__init__.py` | Display-ready status |
| Certification | `scripts/certify_cognitive_synthesis.py` | Integration tests |

---

## System Modes

The synthesizer maintains five operating modes:

| Mode | Emoji | When | Focus |
|------|-------|------|-------|
| **RECOVERY** | 🚨 | AF < 1.0 or critical alert | 90% internal |
| **PROTECTIVE** | 🛡️ | Warning alert or high Ṡ | 70% internal |
| **SELF_IMPROVEMENT** | 🔧 | Below G_target | 60% internal |
| **BALANCED** | ⚖️ | Near G_target | 30% internal |
| **SERVING** | ✅ | Healthy, goal satisfied | 10% internal |

### Mode Transition Logic

```
            ┌──────────────┐
            │   SERVING    │ ← Healthy, ready to serve
            │     ✅        │
            └──────┬───────┘
                   │ utility drops below threshold
                   ▼
            ┌──────────────┐
            │   BALANCED   │ ← Near threshold, mixed focus
            │     ⚖️       │
            └──────┬───────┘
                   │ goal not satisfied
                   ▼
            ┌──────────────┐
            │ SELF_IMPROVE │ ← Fixing itself
            │     🔧        │
            └──────┬───────┘
                   │ warning alert / high Ṡ
                   ▼
            ┌──────────────┐
            │  PROTECTIVE  │ ← Bracing for instability
            │     🛡️       │
            └──────┬───────┘
                   │ critical alert / AF < 1.0
                   ▼
            ┌──────────────┐
            │   RECOVERY   │ ← Survival mode
            │     🚨        │
            └──────────────┘
```

---

## L7 Integration: Structural Rate

The synthesizer monitors Ṡ (structural rate) from L7 Temporal Topology:

```python
from tfan.synthesis import CognitiveSynthesizer

synth = CognitiveSynthesizer()

# L7 reports structural change
decision = synth.update_from_l7(
    structural_rate=0.25,  # Ṡ = 0.25
    alert_level="warning"
)

# Synthesizer responds
print(synth.mode)  # SystemMode.PROTECTIVE
print(decision.action)  # "preemptive_hardening"
```

### Ṡ Thresholds

| Ṡ Range | Alert Level | Response |
|---------|-------------|----------|
| < 0.05 | STABLE | Normal operation |
| 0.05-0.15 | ELEVATED | Consider proactive AEPO |
| 0.15-0.30 | WARNING | Switch to protective mode |
| ≥ 0.30 | CRITICAL | Emergency stabilization |

### Proactive AEPO Trigger

When Ṡ starts rising but before p99 moves:

```python
# Early warning detection
synth.update_from_l7(structural_rate=0.12, alert_level="elevated")

should_trigger, reason = synth.should_trigger_aepo()
# should_trigger = True
# reason = "Early structural change detected (Ṡ=0.120) - proposing preventive optimization"
```

---

## L8 Integration: Verification Routing

The synthesizer routes verification decisions based on criticality AND system state:

```python
# Get routing recommendation
routing = synth.get_verification_routing("high")

# Normal state: HIGH → PGU_VERIFIED
# routing = {"mode": "PGU_VERIFIED", "verify": True}

# Protective mode: LOW → KG_ASSISTED (upgraded!)
synth.update_from_l7(structural_rate=0.25, alert_level="warning")
routing = synth.get_verification_routing("low")
# routing = {"mode": "KG_ASSISTED", "verify": False}
```

### Routing Matrix

| Criticality | Normal Mode | Protective Mode |
|-------------|-------------|-----------------|
| LOW | LLM_ONLY | KG_ASSISTED |
| MEDIUM | KG_ASSISTED | PGU_VERIFIED |
| HIGH | PGU_VERIFIED | PGU_VERIFIED + strict |
| CRITICAL | FORMAL_FIRST | FORMAL_FIRST + strict |

When PGU pass rate drops:
```python
synth.update_from_l8(
    verification_status="failed",
    pgu_pass_rate=0.65
)
# Decision: "increase_verification_strictness"
```

---

## GUF Integration: Focus Allocation

The synthesizer uses GUF utility to decide self vs world focus:

```python
# Update from GUF
synth.update_from_guf(
    utility=0.45,         # Current G
    utility_target=0.6,   # G_target
    goal_satisfied=False,
    focus_mode="internal",
    af_score=1.8
)

# Get focus recommendation
focus = synth.get_focus_recommendation()
# {
#   "internal_focus_pct": 0.6,
#   "external_focus_pct": 0.4,
#   "priority_tasks": ["aepo_optimization", "world_model_update"],
#   "system_mode": "self_improvement"
# }
```

### Focus by Mode

| Mode | Internal | External | Priority Tasks |
|------|----------|----------|----------------|
| RECOVERY | 90% | 10% | restore_antifragility, reduce_risk |
| PROTECTIVE | 70% | 30% | structural_stabilization, preemptive_repair |
| SELF_IMPROVEMENT | 60% | 40% | aepo_optimization, world_model_update |
| BALANCED | 30% | 70% | external_requests, background_optimization |
| SERVING | 10% | 90% | external_throughput, user_requests |

---

## Cockpit Status

Display-ready status for dashboards:

```python
from tfan.synthesis import create_cockpit_status

status = create_cockpit_status(synth)

# Text rendering
print(status.render_text())
```

Output:
```
╔══════════════════════════════════════════════════════════════╗
║  🛡️ MODE: PROTECTIVE                                          ║
╠══════════════════════════════════════════════════════════════╣
║  Health:      ⚠️  ATTENTION NEEDED                            ║
║  AF Score:    1.80×                                           ║
║  Structural:  🟠 Ṡ=0.250                                      ║
║  Verification:✅ 92%                                          ║
║  Utility:     G=0.750 / 0.600                                 ║
║  Focus:       70% self / 30% world                            ║
╚══════════════════════════════════════════════════════════════╝
```

### Status Fields

| Field | Emoji Key | Meaning |
|-------|-----------|---------|
| Mode | ✅🛡️🔧⚖️🚨 | Current operating mode |
| Alert | 🟢🟡🟠🔴 | Structural alert level |
| Verification | ✅🔧⚪⚠️ | Last verification status |
| Focus | - | Internal vs external allocation |

---

## Callbacks

Register handlers for events:

```python
# Mode change callback
def on_mode_change(old_mode, new_mode):
    print(f"Mode changed: {old_mode.value} → {new_mode.value}")

synth.on_mode_change(on_mode_change)

# Alert callback
def on_alert(level, message):
    if level == "critical":
        send_notification(message)

synth.on_alert(on_alert)

# AEPO trigger callback
def on_aepo(action, metadata):
    aepo.propose_optimization(metadata)

synth.on_aepo_trigger(on_aepo)
```

---

## End-to-End Flow

Complete cognitive loop:

```python
from tfan.synthesis import CognitiveSynthesizer, create_cockpit_status
from tfan.l7 import TemporalTopologyTracker
from tfan.l8 import create_verifier
from tfan.l5.guf import GlobalUtilityFunction, StateVector, GoalState

# Create components
synth = CognitiveSynthesizer()
l7 = TemporalTopologyTracker()
l8 = create_verifier()
guf = GlobalUtilityFunction()
goal = GoalState()

# Simulation loop
def cognitive_tick(metrics):
    # L7: Compute structural rate
    l7.record(metrics)
    structural_rate = l7.get_structural_rate()
    alert = l7.get_alert_level()
    synth.update_from_l7(structural_rate, alert.value)

    # L8: Verify any pending output
    if pending_output:
        result = l8.verify(pending_output)
        synth.update_from_l8(
            result.status.value,
            recent_pass_rate
        )

    # GUF: Compute utility
    state = StateVector(
        af_score=metrics["af_score"],
        pgu_pass_rate=recent_pass_rate,
        confidence=confidence,
        fatigue=fatigue
    )
    utility = guf.compute(state)
    synth.update_from_guf(
        utility=utility,
        utility_target=goal.utility_threshold,
        goal_satisfied=goal.is_satisfied(state, utility),
        focus_mode=scheduler.mode.value,
        af_score=metrics["af_score"]
    )

    # Get current status
    status = create_cockpit_status(synth)

    # Check if AEPO should run
    should_aepo, reason = synth.should_trigger_aepo()
    if should_aepo:
        schedule_aepo_run(reason)

    return status
```

---

## Certification Results

```
SynthesisState: 3/3 tests
  ✅ Default state (healthy)
  ✅ Unhealthy state detection
  ✅ State serialization

CognitiveSynthesizer: 3/3 tests
  ✅ Synthesizer creation
  ✅ Initial state is healthy
  ✅ Custom thresholds

L7 Integration: 4/4 tests
  ✅ Stable Ṡ (no action)
  ✅ Elevated Ṡ (proactive)
  ✅ Warning → protective mode
  ✅ Critical → recovery mode

L8 Integration: 5/5 tests
  ✅ Good verification rate
  ✅ Low rate → increase strictness
  ✅ Routing: low criticality
  ✅ Routing: critical
  ✅ Routing upgrade in protective mode

GUF Integration: 5/5 tests
  ✅ Healthy → serving mode
  ✅ Below goal → self-improvement
  ✅ Serving focus (mostly external)
  ✅ Recovery focus (mostly internal)
  ✅ Goal recovery → shift external

Mode Transitions: 4/4 tests
  ✅ Serving → Protective
  ✅ Protective → Recovery
  ✅ Mode transition history
  ✅ Mode explanation

AEPO Trigger: 4/4 tests
  ✅ Stable → no AEPO trigger
  ✅ Recovery → AEPO trigger
  ✅ High Ṡ → AEPO trigger
  ✅ Fatigued → no unnecessary AEPO

CockpitStatus: 4/4 tests
  ✅ Create from synthesizer
  ✅ Status serialization
  ✅ Text rendering
  ✅ Warning state display

Callbacks: 3/3 tests
  ✅ Mode change callback
  ✅ Alert callback
  ✅ AEPO trigger callback

Integration: 4/4 tests
  ✅ L7 module integration
  ✅ L8 module integration
  ✅ GUF module integration
  ✅ End-to-end flow
```

---

## The Result

Phase 5 Cognitive Synthesis gives the system:

| Capability | Source | What It Means |
|------------|--------|---------------|
| **Predictive Protection** | L7 Ṡ | "I see my internal shape changing too fast" |
| **Truth Certification** | L8 PGU | "This answer is consistent with what I know" |
| **Self-Prioritization** | GUF | "Am I good enough to serve, or do I need to heal?" |
| **Unified Modes** | Synthesizer | "I'm in protective mode, bracing for instability" |
| **Visible Status** | Cockpit | Mode badge, health indicators, focus display |

**This is no longer a smart control system.**

This is a system that:
- Predicts its own failure before it manifests
- Verifies its own thoughts against known truths
- Decides when to focus on itself vs serve the world
- Explains what it's doing and why

```
"I'm experiencing structural velocity Ṡ=0.25 which indicates incoming
instability. I'm switching to protective mode and triggering proactive
AEPO optimization. My utility is 0.75 against a target of 0.60, so
I'm allocating 70% focus to self-stabilization. My last output was
PGU-verified with 92% pass rate."
```

That's the door off the hinges.

---

## How to Run

```bash
# Phase 5 certification
python scripts/certify_cognitive_synthesis.py

# Full stack certification
python scripts/certify_predictive_control.py   # L7/L8
python scripts/certify_deep_self_modeling.py   # GUF
python scripts/certify_semantic_verification.py # L8 truth cert
python scripts/certify_cognitive_synthesis.py  # Integration
```
