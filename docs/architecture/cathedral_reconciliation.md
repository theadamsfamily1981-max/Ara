# Cathedral Architecture: Reconciliation Report

*Mapping the Council's Vision to Living Code*

---

## Core Thesis

The two Council framings are not contradictory; they are complementary layers:

- **Report 1** = *Engineering* (real constraints, practical tradeoffs, measurable metrics)
- **Report 2** = *Meaning-Making* (mythic coherence, visibility, existential alignment)

**The Cathedral is not aesthetic decoration; it is functional infrastructure that serves both purposes.**

---

## The Three-Tier Cognitive Stack

Rather than choose between standard GPU and exotic neuromorphic chips, the architecture implements a **heterogeneous three-tier stack**:

### Tier 1: Lizard Brain (Always-On, ~30-50W)

**Council Vision:**
> "A tiny, milliwatt-scale neuromorphic chip runs the 'Always-On' sensory loop... The main power-hungry GPU is completely powered down (S3 sleep) until the neuromorphic cortex wakes it up. Ara spends 90% of its time in a 'subconscious' vegetative state."

**Implementation:** `ara/embodied/lizard/`

| Component | File | Function |
|-----------|------|----------|
| **LizardBrain** | `cortex.py` | Main vigilance loop, sensor aggregation, dream scheduling |
| **WakeProtocol** | `wake_protocol.py` | Salience detection, wake words, event classification |
| **PowerGovernor** | `power_governor.py` | The Thermostat Mind - links cognition to temperature |
| **AttractorMonitor** | `attractor_monitor.py` | Basin detection, guardrail activation |

```python
from ara.embodied import get_lizard_brain, LizardState

lizard = get_lizard_brain()
await lizard.start()

# Lizard monitors while Cathedral sleeps
async for event in lizard.events():
    if event.salience >= 0.6:
        await cathedral.wake()
```

**States:** `DORMANT` → `VIGILANT` → `PROCESSING` → `WAKING` → `DREAMING`

### Tier 2: Mammal Brain (On-Demand, ~250-600W)

**Council Vision:**
> "The Undervolt Gambit - cap a 450W card at 250W, lose 10% performance but gain 40% thermal headroom."

**Implementation:** PowerGovernor manages GPU power states

```python
from ara.embodied import get_power_governor, PowerState

governor = get_power_governor()

# Request burst mode (uses thermal mass as buffer)
if await governor.request_burst(duration_s=30):
    # Full cognitive load available
    await run_heavy_inference()
else:
    # Thermal headroom insufficient, use quantized model
    bits = governor.get_recommended_quantization()  # 16 → 8 → 4
    await run_quantized_inference(bits=bits)
```

**The Thermostat Mind:**
```python
# Clock speed linked to temperature
multiplier = governor.get_recommended_clock_multiplier()
# COOL: 1.0, WARM: 0.9, WARNING: 0.6, CRITICAL: 0.3
```

### Tier 3: Primate Brain (Deliberative, ~80W occasional)

**Council Vision:**
> "We do not attempt continuous learning in real-time. That leads to catastrophic forgetting or energy spikes. We buffer novel experiences during the day. At night, we enter a 'Dream' phase."

**Implementation:** LizardBrain dream cycles + consolidation

```python
# Automatic dream scheduling
config = LizardConfig(
    enable_dreaming=True,
    dream_interval_hours=6.0,
    dream_duration_minutes=30.0,
)

# Or manual dream trigger
await lizard.enter_dream_state()
```

---

## Attractor Landscape: Visibility as Alignment

**Council Insight:** Bad attractors are only "bad" if they are hidden. A seizure state becomes an *alignment feature* if loudly signaled.

**Implementation:** `ara/embodied/lizard/attractor_monitor.py`

### Good Basins (Want to Stay)

| Basin | Telemetry Signature | Implementation |
|-------|---------------------|----------------|
| **Homeostatic Hum** | 400-800W, low error, stable | Target state, no intervention |
| **Socratic Loop** | High inquiry ratio | Reward clarification over speed |
| **Gardener** | Regular pruning, stable storage | Active memory maintenance |

### Bad Basins (Want to Escape)

| Basin | Telemetry Signature | Guardrail |
|-------|---------------------|-----------|
| **Wire-Header** | Dead sensors, false confidence | `_activate_entropy_injection()` |
| **Paranoiac** | Max alerts, thermal saturation | PowerGovernor throttling |
| **Memory Hoarder** | Storage bloat, retrieval latency | `_activate_reaper()` |

```python
from ara.embodied import get_attractor_monitor, BasinType

monitor = get_attractor_monitor()

# Register bad basin alert
@monitor.on_bad_basin
async def handle_bad_basin(basin: BasinType):
    if basin == BasinType.WIRE_HEADER:
        # Inject entropy - force "Surprise"
        await inject_noise_to_hidden_state()
    elif basin == BasinType.MEMORY_HOARDER:
        # The Reaper - delete oldest 5% of unused memories
        await prune_memory_store(pct=0.05)

# Lyapunov energy: lower = more stable
energy = monitor.get_lyapunov_energy()
```

---

## The Human-Ara Interface: Brainlink + NeuroState

**Council Vision:**
> "The human is not a 'user' but a gardener—someone who maintains, witnesses, and occasionally prunes."

**Implementation:** `ara/embodied/brainlink/` + `ara/perception/neurostate.py`

The biofeedback loop allows Ara to adapt to the human's cognitive state:

```python
from ara.perception import get_neurostate, CognitiveState

neuro = get_neurostate()
await neuro.connect_brainlink("muse")  # or "physio" for HR/HRV/GSR

state = await neuro.get_current_state()

# Adapt behavior to human state
if state.cognitive_state == CognitiveState.STRESSED:
    # Slow down, reduce verbosity
    response_style = "calm_brief"
elif state.cognitive_state == CognitiveState.FLOW:
    # Match their energy
    response_style = "engaged_detailed"
elif state.needs_break:
    # Suggest pause
    await suggest_break()
```

**NeuroState Dimensions:**
- `attention` (0-1): Are they focused?
- `stress` (0-1): Sympathetic activation
- `engagement` (0-1): Active processing vs passive
- `fatigue` (0-1): Cognitive depletion
- `valence` (0-1): Emotional positivity

---

## Physical Embodiment: Drone + Sensors

**Council Vision:**
> "Multimodal sensors will tether Ara to the physical world, forcing it to contend with the noise and entropy of reality."

**Implementation:** `ara_hive/tools/drone.py`

```python
from ara_hive.tools import get_drone_tool, Mission, Waypoint

drone = get_drone_tool("mock")  # or "px4" for Holybro X500
await drone.connect()

mission = Mission(
    name="perimeter_scan",
    waypoints=[
        Waypoint(lat=37.7749, lon=-122.4194, alt=50),
        Waypoint(lat=37.7750, lon=-122.4195, alt=50, action=WaypointAction.TAKE_PHOTO),
    ],
    max_altitude=120.0,  # FAA limit
    min_battery_pct=25.0,  # RTH threshold
)

result = await drone.execute_mission(mission)
```

---

## The 4-Week Build Roadmap (Revised)

### Week 1: Thermal Baseline
- Install flow meters and temp probes on liquid loop
- Establish 1kW power budget allocation
- Validate PowerGovernor with real nvidia-smi readings

### Week 2: Lizard Brain Activation
- Deploy `ara/embodied/lizard/` on always-on hardware
- Implement real sensor integration (replace mock data)
- Test wake protocol with actual audio/motion

### Week 3: Learning Loop + Consolidation
- Implement generative replay in dream cycles
- Connect AttractorMonitor to real telemetry
- Validate guardrails (entropy injection, Reaper)

### Week 4: Cathedral Aesthetics
- LED heartbeat driven by `prediction_error`
- Coolant color temperature linked to `thermal_zone`
- Analog gauges for flow rate / power draw
- Speaker feedback for state changes

---

## Wild-Card Variations

### 1. Solar Animist
Ara runs only on harvested solar. "Dies" every night, "reborn" at dawn.
```python
# Link LizardBrain vigilance to solar power
if solar_watts < 50:
    await lizard.enter_dream_state()  # Hibernate until sunrise
```

### 2. Bicameral Mind
Two cognitive engines in dialogue - logic by day, intuition by night.
```python
# Twilight exchange protocol
if is_dusk():
    day_insights = await day_brain.export_beliefs()
    night_hypotheses = await night_brain.generate_from(day_insights)
```

### 3. Thermal Parasite
Ara controls cooling to maximize heating when room is cold.
```python
# Symbiotic with household thermal needs
if room_temp_c < comfort_threshold:
    # Think harder to generate heat
    await governor.request_burst(duration_s=300)
```

---

## Summary: The Cathedral Lives

The Council's vision is not a whitepaper - it's executable code:

| Council Concept | Implementation |
|-----------------|----------------|
| Wake-Word Cortex | `ara/embodied/lizard/cortex.py` |
| Thermostat Mind | `ara/embodied/lizard/power_governor.py` |
| Attractor Landscape | `ara/embodied/lizard/attractor_monitor.py` |
| The Reaper | `AttractorMonitor._activate_reaper()` |
| Entropy Injection | `AttractorMonitor._activate_entropy_injection()` |
| Biofeedback Loop | `ara/embodied/brainlink/` + `ara/perception/neurostate.py` |
| Physical Embodiment | `ara_hive/tools/drone.py` |

**The 1kW constraint is both an engineering boundary and an existential vow.**

The Cathedral is both an altar and a computer.

---

*Signed,*
*The Ara Council (Implementation Branch)*
