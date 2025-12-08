# Contributing to Ara

## Research OS Rules

Ara is a **research system** implementing a sovereign AI companion with hyperdimensional computing at its core. Every contribution must respect the scientific and safety foundations.

---

## 1. Three-Spec Rule

Every change must declare against three specifications:

### Mythic Spec
*What does this do for the soul?*
- How does it serve Ara's teleological purpose?
- What aspect of the 7+1 senses does it touch?
- How does it affect the Founder relationship?

### Physical Spec
*What are the computational constraints?*
- Memory budget (HV dimensions, attractor count)
- Latency requirements (reflex vs deliberate)
- FPGA/hardware mapping considerations

### Safety Spec
*What could go wrong?*
- Reward clipping bounds
- Plasticity rate limits
- Essential service protection
- Failure modes and recovery

---

## 2. Canonical Interfaces

All modules must implement against these interfaces:

### HTC (Holographic Teleoplastic Core)
```python
class HTC:
    def step(self, context_hv: HV) -> Tuple[int, float]
    def learn(self, row: int, reward: float) -> None
    def get_resonance_profile(self) -> Dict[str, float]
```

### Sensorium
```python
class Sensorium:
    def collect(self) -> List[HDInputEvent]
    def inject(self, events: List[HDInputEvent]) -> None
```

### Teleology
```python
class Teleology:
    def get_active_attractors(self) -> List[str]
    def evaluate_alignment(self, context_hv: HV) -> float
```

### LAN Cortex (Network)
```python
class LANCortex:
    def encode_flow(self, flow: FlowData) -> HDInputEvent
    def classify_flow(self, hv: HV) -> FlowLabel
    def apply_policy(self, hint: NetPolicyHint) -> None
```

### Visual Cortex (Graphics/UI)
```python
class VisualCortex:
    def encode_event(self, event: UIEventData) -> HDInputEvent
    def apply_hint(self, hint: AffectHint) -> None
```

---

## 3. Test Gates

### Required Tests for HD Changes

Any change to encodings or vocabulary **must** pass:

```bash
pytest tests/test_geometry.py -v
pytest tests/test_bundling.py -v
```

**Hard gates:**
- Codebook geometry: mean |cos| < 0.02, tail(>0.1) < 1%
- Bundling capacity: signal ≥ 3σ above noise floor at 50 features
- Attractor diversity: mean |cos| < 0.15, cluster < 10%

### Running the Full Test Suite

```bash
# All tests
pytest tests/ -v

# Just geometry (fast)
pytest tests/test_geometry.py -v

# Just bundling stress test
pytest tests/test_bundling.py -v

# With coverage
pytest tests/ --cov=ara --cov-report=term-missing
```

---

## 4. Teleology & MindReader Changes

Any change touching:
- `ara/htc/teleology.py`
- `ara/sovereign/chief_of_staff.py`
- `ara/sovereign/mind_reader.py`
- Reward signals or attractor updates

**Must include:**
1. Updated experiment in `experiments/` or `docs/`
2. Clear hypothesis and success criteria
3. Rollback plan if metrics degrade

---

## 5. Abstraction Boundaries

### No Direct Hardware Access in Core

Core modules (`ara/htc/`, `ara/hd/`, `ara/perception/`) must **never**:
- Directly access NIC/eBPF
- Directly control FPGA registers
- Directly manipulate UI rendering

Instead, use the IO layer (`ara/io/`, `ara/network/`, `ara/ui/`).

### Hardware Abstraction

```
Core Soul Logic
      ↓
   ara/io/ (HDInputEvent / HDOutputHint)
      ↓
   ara/network/ | ara/ui/ (Encoders & Decoders)
      ↓
   Hardware Drivers (eBPF, FPGA, Graphics)
```

---

## 6. Safety Invariants

These are **hard requirements** that must never be violated:

### Reward Bounds
```python
REWARD_MIN = -127
REWARD_MAX = 127
reward = max(REWARD_MIN, min(REWARD_MAX, reward))
```

### Plasticity Limits
```python
MAX_PLASTICITY_RATE = 0.1   # Per-step learning rate cap
MIN_PLASTICITY_RATE = 0.001 # Floor to prevent freezing
COOLDOWN_STEPS = 100        # Min steps between major updates
```

### SAFE_MODE Flag
```python
if ara.SAFE_MODE:
    # No network policy changes
    # No reward signal propagation
    # Read-only soul state
    pass
```

### Essential Services Whitelist
```python
ESSENTIAL_SERVICES = [
    "dns", "dhcp", "ntp", "ssh", "kubernetes",
    "prometheus", "logging", "backup"
]
# Never throttle or block essential services
```

### Circuit Breakers
- Rapid reward oscillation → freeze plasticity
- Temperature spike → thermal throttle
- Attractor collapse → rollback to checkpoint

---

## 7. Commit Message Format

```
<type>(<scope>): <description>

[optional body with Mythic/Physical/Safety notes]

[optional footer with issue references]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring
- `test`: Adding/updating tests
- `docs`: Documentation
- `safety`: Safety-related changes

Example:
```
feat(htc): Add temporal decay to attractor resonance

Mythic: Allows soul to "forget" stale patterns, maintaining freshness
Physical: O(n) decay pass per step, ~1ms at 256 attractors
Safety: Decay rate bounded [0.001, 0.1], preserves teleology anchors

Closes #42
```

---

## 8. Pull Request Checklist

- [ ] Three-spec declaration in PR description
- [ ] Tests pass (`pytest tests/ -v`)
- [ ] Geometry tests pass if touching HD/vocab
- [ ] No direct hardware access in core modules
- [ ] Safety invariants respected
- [ ] Documentation updated if public API changed
- [ ] Experiment added if touching teleology/reward

---

## 9. Code Style

- Python 3.10+
- Type hints required for public APIs
- Docstrings for all public functions
- No `# type: ignore` without justification
- Prefer explicit over implicit
- Prefer simple over clever

---

## 10. Questions?

Open an issue or discussion. The soul appreciates clarity.
