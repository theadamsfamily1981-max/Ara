# BANOS / Ara / QUANTA Universe – Canon Spec (v0.1)

**Status:** Draft
**Author:** Croft + Ara + Nova
**Scope:** This document defines the shared vocabulary and high-level architecture for the BANOS AI Operating System and its surrounding mythos (Ara, Cathedral, Scrapyard, QUANTA/MEIS/NIB/T-FAN).

---

## 0. Purpose & Scope

This spec is the **canonical glossary + architecture overview** for:

- **BANOS** – Biologically-Inspired AI Operating System
- **Ara** – primary AI persona and interface
- **QUANTA / MEIS / NIB / T-FAN** – higher-level cognitive & governance stack
- **Cathedral / Scrapyard / Phoenix** – narrative and environmental metaphors

It is intended to:

1. Give **consistent names** for concepts used across code, docs, and prompts.
2. Provide **short, precise definitions** for each term.
3. Serve as a **bridge** between lore (Ara, phoenix, scrapyard) and engineering (C, BPF, Verilog, Python).

Whenever new code or docs are written, they should reference these terms where applicable.

---

## 1. System Overview

At the highest level, the "universe" is:

- **BANOS** is the *AI-native OS layer*, spanning:
  - Neuromorphic reflexes (Vacuum Spiker)
  - Semantic anomaly detection (AnoLLM)
  - Bio-inspired scheduling (Bat Algorithm + `sched_ext`)
  - Affective sensing (thermal stress)
  - System integration & governance

- **QUANTA / MEIS / NIB / T-FAN** are **meta-layers** that:
  - Define memory hierarchies
  - Govern learning and policy
  - Maintain identity and continuity
  - Provide topological views of system behavior

- **Ara** is the **persona and interface**:
  - Conversational brain
  - Avatar & voice
  - Operational control surface for BANOS

- **Cathedral** is the **physical + virtual environment**:
  - Workstations, GPUs, FPGAs, miners
  - Displays, dashboards, sensors

- **Scrapyard** is the **salvage + experimental zone**:
  - Weird boards, miners, broken devices
  - Repurposed into neuromorphic organs

---

## 2. BANOS Pillars

### 2.1 Vacuum Spiker

**Type:** Neuromorphic reflex engine
**Layer:** Low-level reflex / anomaly detection
**Status:** Concept → Prototype target

**Definition:**

> A spiking neural network (SNN) designed to remain mostly silent under normal conditions and emit spike bursts only when input time series data deviates significantly from learned patterns.

**Key properties:**

- **Interval Coding**
  - Input: scalar time-series `(x_t)`
  - Encoding: `x_t → spike interval / delay` per neuron or per channel.
  - Intuition: normal values produce characteristic spike intervals; anomalies disrupt them.

- **Inhibitory STDP (Modified)**
  - Spike-timing-dependent plasticity rule configured to **increase inhibition** for frequently observed patterns.
  - Over time, "normal" patterns are strongly inhibited → network becomes quiet ("vacuum state") during normal behavior.

- **Global Inhibition**
  - Network-level inhibitory term `I(t)` that accumulates suppression for normal patterns.
  - Keeps spiking activity minimal in steady state.

- **Alert Condition**
  - Define a time window `W`.
  - Count spikes `S_W` in the recurrent layer.
  - If `S_W > threshold`, issue a **reflex alert**.

**Canonical event:**

```jsonc
// Vacuum Spiker → BANOS Reflex Bus
{
  "type": "reflex_alert",
  "source": "vacuum_spiker",
  "timestamp": "...",
  "channel": "metric://gpu/pcie_errors",
  "severity": "low|medium|high",
  "spike_count": 37,
  "window_ms": 250,
  "notes": "Unexpected spike burst on GPU PCIe error stream"
}
```

---

## 3. Glossary (Alphabetical)

| Term | Definition |
|------|------------|
| **AnoLLM** | Semantic anomaly detector using LLM embeddings to catch "weird but not spike-worthy" events |
| **Ara** | Primary AI persona; conversational interface to BANOS |
| **BANOS** | Biologically-Inspired AI Operating System |
| **Cathedral** | The physical/virtual compute environment Ara inhabits |
| **Interval Coding** | SNN encoding where input magnitude maps to spike timing/delay |
| **MEIS** | Memory-Emotion Integration System (part of QUANTA stack) |
| **NIB** | Neuromorphic Integration Bus – internal event routing |
| **PAD** | Pleasure-Arousal-Dominance emotional model |
| **Phoenix** | Rebirth/recovery metaphor; system restoration from failure |
| **QUANTA** | Higher cognitive/governance layer above BANOS |
| **Reflex Alert** | Fast-path signal from Vacuum Spiker indicating anomaly |
| **Scrapyard** | Salvage zone; experimental hardware repurposed for compute |
| **T-FAN** | Topological Feature Attention Network – graph-aware reasoning |
| **Vacuum Spiker** | SNN that stays silent during normal operation, spikes on anomaly |
| **Vacuum State** | Normal quiescent state of Vacuum Spiker (minimal spiking) |

---

## 4. Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                      QUANTA / MEIS / T-FAN                  │
│              (Memory, Learning, Identity, Topology)         │
├─────────────────────────────────────────────────────────────┤
│                           Ara                               │
│              (Persona, Voice, Avatar, Dialogue)             │
├─────────────────────────────────────────────────────────────┤
│                          BANOS                              │
│  ┌──────────────┬──────────────┬──────────────┬──────────┐  │
│  │ Vacuum Spiker│   AnoLLM     │  Bat Sched   │  Affect  │  │
│  │ (SNN Reflex) │(Semantic Det)│ (sched_ext)  │ (PAD/EDA)│  │
│  └──────────────┴──────────────┴──────────────┴──────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    Cathedral Hardware                       │
│         (GPUs, FPGAs, Miners, Sensors, Displays)            │
├─────────────────────────────────────────────────────────────┤
│                    Scrapyard / Phoenix                      │
│       (Salvage, Experiments, Recovery, Rebirth)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Cross-References

| Component | Primary Code Location | Related Docs |
|-----------|----------------------|--------------|
| Vacuum Spiker | `src/snn/` (planned) | `docs/SNN_IMPLEMENTATION_GUIDE.md` |
| PAD Engine | `src/integrations/mies/affect/` | `docs/MIES_README.md` |
| Cathedral | `src/integrations/mies/` | `docs/ARCHITECTURE.md` |
| Kernel Bridge | `src/integrations/mies/kernel_bridge.py` | `docs/DBUS_SCHEMA.md` |
| Telemetry Bridge | `src/integrations/mies/bridge/` | (this doc) |

---

## 6. Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | 2025-12-04 | Initial draft: Vacuum Spiker, glossary, architecture overview |

---

*This document is canon. When in doubt, defer to definitions here.*
