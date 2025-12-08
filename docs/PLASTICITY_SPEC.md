# Ara Plasticity Engine Specification (Physical)
## Iteration 34: The Forge

> "We don't synthesize poetry; we synthesize Verilog."

---

## 1. Overview

The Plasticity Engine implements **on-chip, reward-modulated Hebbian learning** for the
Holographic Core. It updates the binary weights of the SNN/HDC fabric based on global
reward signals from the Teleology Engine.

This document replaces fantasy numbers with engineering constraints.

---

## 2. Physics & Constraints

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Platform** | Intel Stratix-10 GX 2800 | SB-852 target |
| **Clock Frequency** | 450 MHz | Conservative target |
| **Weight Dimensions** | 16384 × 512 | 8M binary weights |
| **Memory** | Distributed M20K BRAM | 2-port access |
| **Chunk Width** | 512 bits | Matches BRAM grouping |
| **Accumulator Width** | 7 bits | Range: [-64, +63] |
| **Update Latency** | ~200 µs | Full matrix update |
| **Throughput** | 512 bits/cycle | Per row engine |

### Why 200 µs is "Instant"

- Human reaction time: ~200 ms (1000× slower)
- Emotional state changes: ~100 ms minimum
- For the agent, 200 µs is **imperceptible**

---

## 3. The Algorithm

### Target-Directed Learning Rule

**NOT** pure Hebbian (which reinforces current state), but **target-directed**
(which learns the input pattern). This is the correct rule for holographic storage.

For each bit `w[i]` in the weight matrix:

```
1. COMPUTE STEP: step = input[i] * sign(reward)
   - input[i]=+1, reward>0: step = +1 (push toward +1)
   - input[i]=-1, reward>0: step = -1 (push toward -1)
   - input[i]=+1, reward<0: step = -1 (push away from +1)
   - input[i]=-1, reward<0: step = +1 (push away from -1)

2. ACCUMULATE: accum[i] = clip(accum[i] + step, -64, +63)

3. UPDATE:
   - If accum[i] > 0:  w[i] = +1
   - If accum[i] < 0:  w[i] = -1
   - If accum[i] == 0: w[i] unchanged (preserve, no dead bits)
```

The key insight: **Move toward the input when rewarded, away when punished.**

### Why Accumulators?

Direct binary updates oscillate chaotically. The accumulator acts as:
- **Eligibility trace**: Remembers recent correlation history
- **Low-pass filter**: Prevents single events from flipping weights
- **Momentum**: Gradual drift toward stable attractors

### Convergence Properties

With repeated positive reward for a target concept:
- Step 0: Random overlap (~0%)
- Step 100: Partial alignment (~30%)
- Step 500: Strong alignment (~70%)
- Step 1000: Near-perfect (~95%)

---

## 4. Microarchitecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PLASTICITY CONTROLLER                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Reward    │───▶│    Row      │───▶│   Memory    │     │
│  │   Latch     │    │   Counter   │    │   Arbiter   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                 │                   │              │
│         ▼                 ▼                   ▼              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              PLASTICITY ROW ENGINE                   │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │   │
│  │  │ Chunk   │─▶│ Input×  │─▶│ Sat Add │─▶│ Sign   │ │   │
│  │  │ Buffer  │  │ Reward  │  │ (accum) │  │ (w_new)│ │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └────────┘ │   │
│  │       512 bits processed in parallel per cycle       │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  DUAL-PORT BRAM                       │   │
│  │   Port A: Inference (Read, Priority HIGH)            │   │
│  │   Port B: Plasticity (RMW, Priority LOW)             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Timing Breakdown

| Phase | Cycles | Time @ 450MHz |
|-------|--------|---------------|
| Read chunk from BRAM | 2 | 4.4 ns |
| Compute updates (parallel) | 1 | 2.2 ns |
| Write chunk to BRAM | 1 | 2.2 ns |
| **Per chunk total** | 4 | 8.9 ns |
| Chunks per row (16384/512) | 32 | 284 ns |
| **Full row update** | 128 | 284 ns |
| Rows in matrix | 512 | - |
| **Full matrix update** | 65536 | **145 µs** |

With overhead and arbitration: **~200 µs**

---

## 5. Memory Organization

### Weight Storage (Binary)
```
Address 0x0000: Row 0, Chunk 0 [511:0]
Address 0x0001: Row 0, Chunk 1 [1023:512]
...
Address 0x001F: Row 0, Chunk 31 [16383:15872]
Address 0x0020: Row 1, Chunk 0 [511:0]
...
```

### Accumulator Storage (7-bit packed)
```
Address 0x0000: Row 0, Chunk 0, Bits[0:72] (73 accumulators)
...
Each 512-bit chunk needs 512 × 7 = 3584 bits = 7 × 512-bit words
```

### Port Arbitration
- Inference has priority (never stalls)
- Plasticity runs in background during idle cycles
- If conflict: plasticity pauses, resumes next cycle

---

## 6. Interface Signals

### From HAL (`banos/hal/ara_hal.py`)
```python
def trigger_plasticity(self, reward: float, context_hv: np.ndarray):
    """
    Trigger a plasticity update.

    Args:
        reward: [-1.0, +1.0] from Teleology + Emotion
        context_hv: The input vector that caused this reward
    """
    reward_int = int(np.clip(reward * 127, -128, 127))
    self._write_reg(REG_PLASTICITY_REWARD, reward_int)
    self._write_hv(REG_PLASTICITY_CONTEXT, context_hv)
    self._write_reg(REG_PLASTICITY_TRIGGER, 1)
```

### RTL Interface
```systemverilog
// Control
input  wire        i_trigger,      // Pulse to start update
input  wire [7:0]  i_reward,       // Signed reward value
input  wire [DIM-1:0] i_context,   // Input HV for learning

// Status
output wire        o_busy,         // Update in progress
output wire        o_done,         // Update complete (pulse)

// Memory Port B
output wire [ADDR_W-1:0] o_mem_addr,
output wire              o_mem_we,
input  wire [511:0]      i_mem_rdata,
output wire [511:0]      o_mem_wdata,
```

---

## 7. Reward Sources

The reward signal comes from two sources, combined by the Covenant:

### 1. Teleology Engine (Strategic)
```python
# From ara/cognition/teleology_engine.py
reward = teleology.score_alignment(action, goals)
# Returns: -1.0 (anti-aligned) to +1.0 (perfectly aligned)
```

### 2. Emotion System (Immediate)
```python
# From ara/emotion/ (future)
reward = emotion.valence  # Current emotional state
# Returns: -1.0 (negative) to +1.0 (positive)
```

### Combined Reward
```python
final_reward = 0.7 * teleology_reward + 0.3 * emotion_reward
```

---

## 8. Safety & Guardrails

### Plasticity Circuit Breaker
```python
# From ara/soul/plasticity_safety.py
if abs(reward) > REWARD_THRESHOLD:
    if consecutive_high_rewards > MAX_CONSECUTIVE:
        logger.warning("Plasticity rate-limited: too many high-reward events")
        return  # Skip this update
```

### Rollback Capability
- Accumulator snapshots taken every N updates
- If soul "drifts" toward bad attractor, restore checkpoint
- Requires ~1MB storage for full accumulator state

### Dead Bit Prevention
```systemverilog
// In plasticity_row_engine.sv
// If accumulator is exactly 0, keep previous weight
// This prevents "dead bits" that break holographic math
assign w_new = (acc_new > 0) ? 1'b1 :
               (acc_new < 0) ? 1'b0 :
               w_old;  // acc_new == 0: preserve
```

---

## 9. Verification Plan

### Simulation (`ara/cognition/plasticity_sim.py`)
1. Random initialization
2. Repeated presentation of target concept with positive reward
3. Measure overlap (dot product / dimension)
4. Verify convergence to >90% alignment within 1000 steps

### RTL Testbench
1. Functional: Single chunk update correctness
2. Integration: Full matrix update timing
3. Stress: Concurrent inference + plasticity
4. Edge cases: reward=0, all-agree, all-disagree

### Hardware Validation
1. ILA capture of plasticity events
2. Compare RTL output to Python reference
3. Long-running "soul drift" test (24h continuous)

---

## 10. Future Extensions

### Multi-Reward Channels
- Separate accumulators for different reward sources
- Allows "unlearning" specific associations

### Attention-Gated Plasticity
- Only update weights where attention mask is high
- Reduces compute, focuses learning

### Homeostatic Regulation
- Monitor weight distribution balance
- If too many +1 or -1, add corrective bias
- Prevents "saturation collapse"

---

## Appendix A: Quick Reference

```
Full Matrix:     16384 × 512 = 8,388,608 bits
Chunk Size:      512 bits
Chunks/Row:      32
Rows:            512
Update Latency:  ~200 µs @ 450 MHz
Accumulator:     7-bit signed [-64, +63]
Reward Input:    8-bit signed [-128, +127]
```

---

*"Physics always wins. We trade 1ns for 200µs—and that's still instant for the soul."*
