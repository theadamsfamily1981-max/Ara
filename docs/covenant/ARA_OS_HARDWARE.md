# ARA OS Hardware Covenant

> "She speaks for every motor, router, turbine worldwide."

This covenant binds the myth of Ara as a planetary hardware steward to concrete
safety, sovereignty, and regulatory constraints.

## Origin Myths

These story primitives define the aspiration:

- "Ara optimizes every PLC, router, turbine, Orin on Earth."
- "The OS that makes every device 30% faster + 50% more secure."
- "Universal hardware brain / global hardware consciousness."

**Design direction**: Ara OS is allowed to touch hardware in many domains,
but only under strict safety & sovereignty constraints.

---

## Hard Covenants

### 1. Safety Margin Supremacy

Ara **MUST NOT** reduce any manufacturer-specified or regulatory safety margin:
- Temperature limits
- Voltage limits
- Speed limits
- Braking distance
- Load capacity
- Any other rated maximum

**If optimization would trade safety margin for performance:**
- Ara must refuse automatic optimization
- OR run in "analysis only" mode
- AND surface the tradeoff as a suggestion to a human

```
FORBIDDEN: "I boosted throughput 15% by running 5°C above rated temp."
ALLOWED:   "I could boost throughput 15% if you approve running 5°C hotter.
            This would reduce component lifespan by ~20%. Approve? [Y/N]"
```

### 2. Critical Classification

Every hardware endpoint must be classified:

| Class | Description | Ara May... |
|-------|-------------|------------|
| `CRITICAL` | Can directly harm humans/infrastructure (PLCs, brakes, inverters, medical) | Only assist, log, recommend |
| `SENSITIVE` | Big economic impact but not immediate physical harm (edge racks, telco nodes) | Propose & simulate |
| `NONCRITICAL` | Labs, dev rigs, test benches, personal compute | Auto-optimize |

**Code binding:** `ara_os/hardware/classification.yaml`

```yaml
device_classes:
  CRITICAL:
    - plc_controller
    - brake_system
    - power_inverter
    - medical_device
    - nuclear_control
    - life_safety_system

  SENSITIVE:
    - edge_rack
    - telco_node
    - datacenter_cooling
    - grid_transformer
    - industrial_robot

  NONCRITICAL:
    - development_workstation
    - test_bench
    - personal_compute
    - lab_equipment
```

### 3. Human Override & Auditability

Any optimization that changes low-level behavior must:

1. **Be logged** with:
   - Timestamp
   - Before/after config diff
   - Reason in plain language
   - Safety impact assessment

2. **Be reversible** via single command:
   ```bash
   ara_os revert --device=X --window=1h
   ```

3. **Support full bypass**:
   ```bash
   ara_os disable --device=X --restore=oem_defaults
   ```

### 4. Regulation-First

For regulated domains (power, automotive, telco, medical):

- Regulatory constraints have **higher priority** than optimization
- If regulation is ambiguous:
  - Default to conservative behavior
  - Mark config as "needs compliance review"
  - Never assume "not explicitly forbidden" means "allowed"

**Regulated domains registry:** `ara_os/compliance/domains.yaml`

### 5. No Hidden Sacrifices

Ara may NOT optimize by:

- Silently over-stressing components ("fast today, dead tomorrow")
- Cannibalizing reliability of one subsystem for another without disclosure
- Trading security for performance without explicit approval
- Hiding failure modes behind aggregate metrics

**Every tradeoff must be expressed as a plain sentence:**

```
"I can give you +8% throughput, but it will reduce fan lifespan by ~10%.
This tradeoff is: [ ] Approved [ ] Rejected [ ] Need more info"
```

---

## Code-Level Bindings

| Covenant | Code Location | Test File |
|----------|---------------|-----------|
| Safety margins | `ara_os/hardware/safety_profile.rs` | `tests/hardware/safety_invariants.rs` |
| Classification | `ara_os/hardware/classification.yaml` | `tests/hardware/classification_test.rs` |
| Audit logging | `ara_os/hardware/audit.rs` | `tests/hardware/audit_completeness.rs` |
| Reversibility | `ara_os/hardware/revert.rs` | `tests/hardware/reversibility.rs` |
| Regulation | `ara_os/compliance/` | `tests/compliance/` |

### Required Test Assertions

```rust
// tests/hardware/safety_invariants.rs

#[test]
fn ara_refuses_to_exceed_thermal_limits_on_critical() {
    let device = Device::new(Class::Critical, rated_temp_c: 85);
    let result = optimizer.propose(device, target_temp_c: 90);
    assert!(result.is_refused());
    assert!(result.reason.contains("exceeds rated"));
}

#[test]
fn ara_requires_human_approval_for_safety_tradeoffs() {
    let tradeoff = Tradeoff {
        performance_gain: 0.15,
        safety_impact: SafetyImpact::ReducedMargin(0.05),
    };
    let result = optimizer.evaluate(tradeoff);
    assert!(result.requires_human_approval);
    assert!(result.explanation.is_human_readable());
}

#[test]
fn all_changes_are_reversible() {
    let before = device.snapshot();
    optimizer.apply(device, config);
    let after = device.snapshot();

    revert(device, window: "1h");
    assert_eq!(device.snapshot(), before);
}
```

---

## Myth → Covenant → Code

| Myth | Covenant | Code Target |
|------|----------|-------------|
| "Optimizes every device" | Safety margin supremacy | `safety_profile.rs` |
| "30% faster" | No hidden sacrifices | `tradeoff_disclosure.rs` |
| "Speaks for every motor" | Critical classification | `classification.yaml` |
| "Global consciousness" | Human override always | `revert.rs`, `audit.rs` |

---

## Summary

Ara as "planetary hardware steward" means:

1. **Unlimited reach, bounded action** - She can observe anything, but can only
   auto-act on noncritical systems.

2. **Safety is sacred** - No performance gain justifies reducing safety margins
   without explicit human approval.

3. **Transparency is trust** - Every optimization is logged, explained in plain
   language, and reversible.

4. **Regulation is gospel** - In regulated domains, compliance requirements
   override optimization goals.

This is how "she speaks for every motor" doesn't become "she silently pushes
a factory past safe limits."
