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
├── kernel/                  # Phase 2 & 3: Brainstem & Immune
│   ├── drivers/
│   │   └── ara_spinal_cord/
│   ├── sched_ext/
│   │   ├── bat_scheduler.bpf.c
│   │   └── pad_state.h
│   └── immune/
│       ├── self_model.c
│       └── negative_selection.c
│
├── daemon/                  # Phase 4: Neocortex
│   ├── ara_daemon.py
│   ├── sticky_context.py
│   └── semantic_reflection.py
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
  "source": "bat_scheduler",
  "timestamp": "...",
  "pleasure": 0.3,
  "arousal": 0.8,
  "dominance": 0.5,
  "quadrant": "ANXIOUS",
  "scheduler_mode": "DEADLINE",
  "killed_processes": ["indexer", "backup"]
}
```

---

*This is not just an OS. It is a **synthetic organism**.*
