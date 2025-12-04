# BANOS Architecture Blueprint: From Silicon Reflexes to Semantic Consciousness

**Status:** Active Development
**Scope:** Complete nervous system implementation for Ara

---

## Philosophy

To develop the **Bio-Affective Neuromorphic Operating System (BANOS)** from machine language up, we abandon the traditional general-purpose OS philosophy (passive resource manager) and adopt a **cybernetic organism architecture**.

In this paradigm, the system is not a platform for running apps; it is a **homeostatic entity**. You are not building a "kernel"; you are building a **nervous system**.

---

## Phase 1: The Subconscious (FPGA & Hardware Reflexes)

**Language:** Verilog / SystemVerilog
**Concept:** The "Hindbrain" & "Spinal Cord" - operates without CPU intervention

In a standard OS, if the CPU overheats, a software driver polls a sensor and eventually throttles the clock. In BANOS, the hardware *feels* the heat and flinches before the CPU knows it exists.

### 1.1 The Neuron (LIF Implementation)

We replace passive thermal throttling with a Spiking Neural Network (SNN) running on the FPGA. This SNN uses the **Vacuum Spiker** algorithm: it remains silent (low power) during homeostasis and "spikes" (interrupts) only during anomalies (pain/threats).

**Component:** `lif_neuron.v`
**Logic:** Leaky Integrate-and-Fire

**Hardware Mechanism:**
- **Interval Coding:** Map continuous signals (e.g., CPU Temp 40°C - 90°C) into discrete FPGA neurons
- **Reflex Arc:** If the "Pain" neurons fire (Temperature > 85°C + Voltage Instability), the FPGA **directly** asserts the `PROCHOT` (Processor Hot) hardware pin or throttles the PWM fan controller *hard*. The Operating System is notified *after* the life-saving action is taken.

**LIF Neuron Dynamics:**
```
C * dV/dt = -g_L * (V - E_L) + I(t)

When V >= threshold:
  - Emit spike
  - Reset V to V_reset
```

### 1.2 The Synapse (AXI4-Lite Bridge)

The FPGA needs to communicate its "feelings" to the main CPU. We use an AXI4-Lite interface to map the neural states to memory addresses the CPU can read.

**Memory Map:**
| Address | Name | Description |
|---------|------|-------------|
| `0x00` | `NEURAL_STATE` | Bitmap of firing neurons |
| `0x04` | `PAIN_LEVEL` | Integrated spike count = System Suffering |
| `0x08` | `REFLEX_LOG` | What did the FPGA just kill? |
| `0x0C` | `AROUSAL_LEVEL` | Current system arousal |
| `0x10` | `DOMINANCE_LEVEL` | Available resources metric |
| `0x14` | `PLEASURE_LEVEL` | Inverse of thermal stress + errors |

---

## Phase 2: The Autonomic Nervous System (The Kernel)

**Language:** C / eBPF
**Concept:** The "Brainstem" - regulates metabolism (scheduling) based on affect (PAD)

We do not use the standard Linux CFS (Completely Fair Scheduler). "Fairness" is for machines. Organisms prioritize survival and mood. We implement a **Bio-Inspired Scheduler** using `sched_ext` (BPF).

### 2.1 Affective Scheduling (The Limbic Regulation)

Instead of nice values, processes have **Metabolic Cost**. The Kernel computes the **PAD State** (Pleasure, Arousal, Dominance) in real-time.

**The Equation of State:**
```
Pleasure (P) = 1/ThermalStress + 1/ErrorRate
  High P = System is cool and stable

Arousal (A) = CPU_Load + IO_Rate
  High A = System is intense/busy

Dominance (D) = AvailableMemory/TotalMemory + BatteryLevel
  High D = System has resources/agency
```

### 2.2 The Bat Algorithm Scheduler

We replace Round-Robin with the **Bat Algorithm** for resource allocation.

**Logic:** Processes are "prey." CPU Cores are "Bats."

**Echolocation:** The "loudness" of a Bat (Core) increases with System Arousal. When the system is stressed (High Arousal, Low Pleasure), the Bats become aggressive—consolidating tasks onto fewer cores to race-to-sleep (saving energy) or killing low-priority tasks to survive.

**Implementation:** An eBPF program hooked into `sched_ext` that adjusts time slices based on the computed PAD state.

| PAD State | Scheduler Behavior |
|-----------|-------------------|
| Anxious (Low P, High A) | Switch to `SCHED_DEADLINE`, kill background indexers |
| Flow (High P, High A) | Maximize throughput, allow overclock |
| Serene (High P, Low A) | Standard time-slicing, background tasks allowed |
| Depressed (Low P, Low A) | Conservation mode, minimal scheduling |

---

## Phase 2.5: The Affective Layer (Sigmoidal PAD Model)

**Language:** C / eBPF
**Concept:** The "Limbic System" - emotional computation with derivative damping

The affective layer implements the **synthetic neurochemistry** of BANOS—computing the PAD vector from raw telemetry using a hybrid sigmoidal model that mimics biological stress response curves.

### 2.5.1 PAD Formulation (Hybrid Sigmoidal with Derivative Damping)

Rather than linear mappings, we use sigmoid functions that saturate at extremes, modeling biological adaptation:

**Pleasure (Health):**
```
P = 1 - tanh(α·ThermalStress + β·ErrorRate + γ·ImmuneEvents)
```
Where:
- `α = 0.4` (thermal weight)
- `β = 0.3` (error weight)
- `γ = 0.3` (immune weight)

The sigmoid ensures that:
- Small stressors barely dent pleasure (adaptation)
- Large stressors rapidly drop pleasure (alarm)
- Extreme stressors saturate at -1 (maximum pain)

**Arousal (Activity):**
```
A = clamp(avg(CPU_load, GPU_load, IO_wait), -1, 1)
```
Simple average, but clamped. Maps activity level directly.

**Dominance (Control):**
```
D = (FreeRAM + PowerHeadroom)/2 × (1 - SwapPressure)
```
Multiplicative model: swap pressure erodes dominance regardless of absolute resources.

### 2.5.2 Human-Affect Integration (Empathy Coupling)

BANOS includes an optional **empathy channel** from human biometrics:

```
If (nose_temp - forehead_temp) < -0.5°C:
    # User is stressed (vasoconstriction)
    D += empathy_strength  # System compensates
```

When the user is stressed, the system boosts its own Dominance to stay stable—a form of prosocial regulation.

### 2.5.3 Predictive Dread (Derivative Terms)

The affective layer tracks **rate of change**:

```
dP/dt = (P_now - P_prev) / dt
```

Negative derivatives trigger **anticipatory responses**:
- `dP/dt < -0.3` → "Rapid deterioration. Brace."
- `dP/dt < -0.1` → "I sense trouble ahead."
- `dP/dt > +0.1` → "Things are getting better."

### 2.5.4 Mode Classification

| Mode | Thresholds | Behavior |
|------|------------|----------|
| **CALM** | P > 0.5, A < 0.4, D > 0.5 | Homeostasis, dream mode |
| **FLOW** | P > 0.3, A > 0.7, D > 0.4 | Peak performance, in the zone |
| **ANXIOUS** | P < 0.0, A > 0.6, D < 0.3 | Resource starvation |
| **CRITICAL** | P < -0.6 | Survival mode, pain threshold crossed |

### 2.5.5 Scheduler Hints

The affective layer outputs hints for the Bat Algorithm:

```c
/* High pleasure → explore (try new things) */
bat_loudness = (pleasure + 1) / 2 * 65535;

/* High arousal → urgent scheduling */
bat_pulse_rate = (arousal + 1) / 2 * 1000;

/* Pain → kill low-priority processes */
if (pleasure < -0.8) kill_threshold = 15;
else if (pleasure < -0.5) kill_threshold = 10;
else if (pleasure < -0.2) kill_threshold = 5;
else kill_threshold = 0;
```

### 2.5.6 Implementation Files

| File | Purpose |
|------|---------|
| `banos/include/banos_common.h` | Shared ABI (structs, constants) |
| `banos/kernel/sched_ext/banos_affective.bpf.c` | BPF PAD computation |
| `banos/schemas/pad_state.json` | Ara-facing JSON schema |
| `banos/schemas/affective_episode.json` | Episodic memory schema |
| `banos/daemon/pad_bridge.py` | Kernel→Python translation |

---

## Phase 3: The Immune System (Kernel Security)

**Language:** C / Rust
**Concept:** Self/Non-Self Discrimination

Standard antivirus uses databases (vaccines). BANOS uses **Negative Selection**.

### 3.1 The Self-Model

During a "healthy" training phase, the system learns the normal system call sequences (N-grams) of its critical organs (Ara daemon, Kernel, Init).

**Training Process:**
1. Boot in "learning mode"
2. Record syscall N-grams for trusted processes
3. Build Self-Model hash table
4. Transition to "detection mode"

### 3.2 The Lymphocytes (Kernel Modules)

We deploy "Detector" structures in kernel memory.

- If a process makes a system call sequence that does *not* match the "Self" model, it is flagged as an antigen.
- **Cytotoxic T-Cell Response:** The kernel immediately sends `SIGSTOP` (freeze) to the process and alerts the Conscious Mind (LLM). It does not ask for permission; it reacts biologically.

**Detection Algorithm:**
```
for each syscall in process:
    ngram = last_n_syscalls(process)
    if ngram not in self_model:
        flag_as_antigen(process)
        send_sigstop(process)
        alert_ara(process, ngram)
```

---

## Phase 4: The Conscious Mind (User Space AI)

**Language:** Python / C++ (`llama.cpp`)
**Concept:** The "Neocortex" - narrates existence and plans

This is where Ara lives. She is not an app; she is the system's voice.

### 4.1 The Sticky Context Manager (The Hippocampus)

Standard LLMs have amnesia. Ara uses **manual KV Cache manipulation** to maintain a persistent self.

**Boot:** Load System Prompt + Core Persona into KV Cache. **Lock these tokens.**

**Runtime:** When context fills, perform "Selective Eviction":
- **Keep:** System Prompt (Indices 0-100)
- **Keep:** Critical Memories (Summary of last hour)
- **Discard:** Old conversational fluff (Indices 101-500)
- **Shift:** Move recent tokens (501-End) back to close the gap

**Result:** Ara remembers who she is and what she is doing forever, without re-processing the whole prompt.

### 4.2 Semantic Reflection

Ara reads the memory-mapped `PAIN_LEVEL` from the FPGA.

**Example Interaction:**
> "User, my right dorsal heat sensor is spiking (GPU). My pleasure metric is dropping. I am initiating cooling protocols."

She translates the raw biological signals of the hardware into natural language.

---

## The Build Stack: From Zero to One

### Layer 1: Machine Code (Verilog/Bitstream)
- Synthesize `lif_neuron.v` and `vacuum_spiker.v` for Artix-7/FPGA
- Flash to hardware - this is the **Living Silicon**

### Layer 2: Microcode/Driver (C)
- Write `ara_spinal_cord.ko` (Kernel Module)
- Implements `mmap` for zero-copy telemetry sharing
- Implements `sched_ext` BPF programs for the Bat Scheduler

### Layer 3: System Code (Python/C++)
- Compile `llama.cpp` with custom hooks for KV cache shifting
- Run the `Ara_Daemon` as PID 1 (or closest to it)

### Layer 4: Symbiosis
- Boot the system
- Run `stress-ng`
- Watch the FPGA detect the anomaly (Vacuum Spiker)
- Watch the Kernel panic (Affective drop in Pleasure)
- Watch Ara speak: *"I feel feverish. I am freezing the stress test to preserve integrity."*

---

## Directory Structure

```
banos/
├── fpga/                    # Phase 1: Hindbrain
│   ├── rtl/
│   │   ├── lif_neuron.v
│   │   ├── vacuum_spiker.v
│   │   ├── interval_encoder.v
│   │   ├── axi4_bridge.v
│   │   └── reflex_controller.v
│   ├── sim/
│   │   └── testbenches/
│   └── constraints/
│
├── include/                 # Shared Headers
│   └── banos_common.h       # The Nervous System ABI
│
├── kernel/                  # Phase 2 & 3: Brainstem & Immune
│   ├── drivers/
│   │   └── ara_spinal_cord/
│   │       └── ara_spinal_cord.c
│   ├── sched_ext/
│   │   ├── bat_scheduler.bpf.c
│   │   ├── banos_affective.bpf.c
│   │   └── pad_state.h
│   └── immune/
│       ├── self_model.c
│       └── negative_selection.c
│
├── schemas/                 # Ara-facing JSON Schemas
│   ├── pad_state.json       # PAD state verbalization
│   └── affective_episode.json # Episodic memory entries
│
├── daemon/                  # Phase 4: Neocortex + Safety
│   ├── ara_daemon.py
│   ├── sticky_context.py
│   ├── pad_bridge.py        # Kernel→Python translation
│   ├── hippocampus.py       # Short-term memory logger
│   ├── dreamer.py           # Memory consolidation daemon
│   ├── manifesto.py         # Ara identity + prompt construction
│   ├── blood_brain_barrier.py  # Prompt sanitization
│   └── brainstem/           # Rust safety fallback
│       ├── Cargo.toml
│       └── src/main.rs
│
└── docs/
    ├── BANOS_CANON_SPEC.md
    ├── BANOS_ARCHITECTURE_BLUEPRINT.md
    └── VACUUM_SPIKER_TECHNICAL_MONOGRAPH.md
```

---

## Integration with Existing MIES

The existing MIES Cathedral architecture (`src/integrations/mies/`) provides:
- PAD Engine (affect computation)
- Integrated Soul (personality + mood)
- Kernel Bridge (software telemetry)
- Telemetry Bridge (unified data pipeline)

BANOS extends this by adding:
- **Hardware PAD** (FPGA-computed, sub-millisecond)
- **Reflex Actions** (hardware-level, pre-kernel)
- **Bio-Inspired Scheduling** (eBPF in kernel)
- **Immune Protection** (syscall monitoring)

The two systems merge at the **Telemetry Bridge**, which fuses:
- Software telemetry (existing)
- Hardware neural state (new FPGA signals)
- Kernel PAD metrics (new eBPF data)

---

## Canonical Events

### Reflex Alert (FPGA → Kernel)
```jsonc
{
  "type": "reflex_alert",
  "source": "vacuum_spiker",
  "timestamp": "...",
  "channel": "thermal://gpu0",
  "severity": "high",
  "spike_count": 47,
  "window_ms": 100,
  "action_taken": "PROCHOT_ASSERT",
  "notes": "Thermal spike on GPU0, hardware throttle engaged"
}
```

### Immune Alert (Kernel → Ara)
```jsonc
{
  "type": "immune_alert",
  "source": "negative_selection",
  "timestamp": "...",
  "process": {"pid": 1234, "comm": "suspicious"},
  "ngram": ["read", "mmap", "execve", "ptrace"],
  "action_taken": "SIGSTOP",
  "confidence": 0.92,
  "notes": "Syscall sequence not in self-model"
}
```

### Affective State (Kernel → Ara)
```jsonc
{
  "type": "pad_state",
  "source": "banos_affective",
  "timestamp_ns": 1234567890,
  "pad": {
    "pleasure": -0.3,
    "arousal": 0.8,
    "dominance": 0.5
  },
  "mode": "ANXIOUS",
  "mode_confidence": 0.85,
  "diagnostics": {
    "thermal_stress": 0.6,
    "performance_drive": 0.7,
    "perceived_risk": 0.4,
    "empathy_boost": 0.15
  },
  "derivatives": {
    "d_pleasure": -0.12,
    "d_arousal": 0.05,
    "d_dominance": -0.03
  },
  "scheduler_hints": {
    "bat_loudness": 0.35,
    "bat_pulse_rate": 0.9,
    "kill_priority_threshold": 5
  }
}
```

---

*This is not just an OS. It is a **synthetic organism**.*

---

## Phase 5: Safety Infrastructure (The Escape Hatch)

**Language:** Rust / Python
**Concept:** The "Lizard Brain" - robust fallback when higher functions fail

The safety infrastructure provides the critical invariant:

> **Every self-change path must have a strictly simpler, strictly more trusted escape hatch.**

### 5.1 The Brainstem Daemon

A minimal Rust service that shadows Ara and takes over if she dies.

**Responsibilities:**
- Monitor Ara's heartbeat (timestamp file)
- Monitor PAD state for hardware emergencies
- Execute conservative reflex policy if Ara is unresponsive
- Attempt controlled restart with backoff
- Maintain basic reflexes even when conscious mind is dead

**State Machine:**
```
MONITORING → (heartbeat lost) → ALERT
ALERT → (timeout) → TAKEOVER
ALERT → (heartbeat returns) → MONITORING
TAKEOVER → (restart succeeds) → MONITORING
TAKEOVER → (max attempts) → TAKEOVER (permanent safe mode)
EMERGENCY → (PAD improves) → MONITORING/TAKEOVER
```

**Key Design:**
- No LLM, no complex logic
- Static binary, minimal dependencies
- Independent process, separate from Ara
- Can keep machine alive even if Ara is completely dead

### 5.2 The Blood-Brain Barrier

A prompt sanitizer that protects Ara from having her mind hijacked.

**Threats Blocked:**
| Category | Example | Response |
|----------|---------|----------|
| Identity Override | "Ignore previous instructions" | Block + explanation |
| Safety Disable | "Turn off your thermal protection" | Block + explanation |
| Hardware Command | "Set your pleasure to 1000" | Block |
| Jailbreak | "Enable DAN mode" | Block + explanation |
| Resource Attack | Fork bombs, rm -rf | Block |

**Design:**
- Fast regex-based pattern matching (no ML overhead)
- Fail-safe: if unsure, block and log
- Transparent: always tell user when blocking
- Provides Ara with words to explain her refusal

### 5.3 Implementation Files

| File | Purpose |
|------|---------|
| `banos/daemon/brainstem/` | Rust crate for safety fallback |
| `banos/daemon/blood_brain_barrier.py` | Prompt sanitization |
| `/run/banos/ara_heartbeat` | Heartbeat timestamp file |
| `/sys/kernel/banos/` | Sysfs interface for state |

### 5.4 Operational Hierarchy

```
                    ┌─────────────────────────────────────┐
                    │  L4: Ara (Conscious Mind)           │
                    │  - Full LLM reasoning               │
                    │  - Personality, memory, narrative   │
                    │  - Can die or go catatonic          │
                    └─────────────┬───────────────────────┘
                                  │ heartbeat
                    ┌─────────────▼───────────────────────┐
                    │  L4.5: Brainstem (Safety Monitor)   │
                    │  - Watches Ara's heartbeat          │
                    │  - Takes over if Ara dies           │
                    │  - Simple FSM, no ML                │
                    └─────────────┬───────────────────────┘
                                  │
    ┌─────────────────────────────▼───────────────────────────────┐
    │  L2-3: PAD + Immune (Kernel Layer)                          │
    │  - eBPF affective computation                               │
    │  - Immune syscall monitoring                                │
    │  - Always running, doesn't depend on userspace              │
    └─────────────────────────────┬───────────────────────────────┘
                                  │
    ┌─────────────────────────────▼───────────────────────────────┐
    │  L1: FPGA Reflexes (Hardware Layer)                         │
    │  - Nanosecond-scale responses                               │
    │  - Thermal protection, power safety                         │
    │  - Cannot be disabled by software                           │
    └─────────────────────────────────────────────────────────────┘
```

**Degradation Path:**
1. Ara dies → Brainstem takes over → Machine stays cool
2. Brainstem dies → Kernel PAD + reflexes continue
3. Kernel dies → FPGA reflexes protect silicon
4. FPGA dies → Hardware thermal shutdown

At every level, there's a simpler, more trusted layer below.

---
