# NeuroBalance™ v2.0 - Hierarchical Precision Optimizer
## Advanced Device Specification

> **RESEARCH PROTOTYPE CONCEPT**
>
> This extends v1.0 with hierarchical precision targeting and tACS modulation.
> All interventions are hypothetical research targets, not validated therapies.

---

## 1. Key Upgrades from v1.0

| Feature | v1.0 | v2.0 | Rationale |
|---------|------|------|-----------|
| Precision metric | Single D ratio | Hierarchical D_low, D_high | Different cortical levels |
| Stimulation | tDCS (optional) | tACS (frequency-specific) | Direct oscillatory control |
| Feedback | Fixed protocol | Context-aware | Adapts to user state |
| Control law | PI on D | Multi-objective on ΔH | Hierarchical balance |

---

## 2. Hierarchical Precision Model

### 2.1 The Problem with Single D

The original D = Π_prior / Π_sensory treats precision as monolithic.
In reality, precision is **weighted across hierarchical levels**:

```
CORTICAL HIERARCHY

    HIGH LEVELS (slow, abstract)
    ├── Self-model, beliefs, long-term predictions
    ├── Frontal cortex, theta rhythms (4-8 Hz)
    └── D_low: Precision on cognitive priors
                    │
                    ▼
    LOW LEVELS (fast, concrete)
    ├── Sensory features, immediate predictions
    ├── Posterior cortex, gamma rhythms (30-80 Hz)
    └── D_high: Precision on perceptual inference
```

### 2.2 New Metrics

**D_low** - Derived from low-frequency rhythms (theta/alpha):
```
D_low = θ_frontal_power × (1 + θ-γ_coupling_frontal)
        ─────────────────────────────────────────────
        α_posterior_suppression + β_posterior
```

**D_high** - Derived from high-frequency rhythms (gamma/beta):
```
D_high = γ_posterior × (1 + β_frontal)
         ─────────────────────────────────
         γ_posterior + sensory_evoked_response
```

**Hierarchical Discrepancy**:
```
ΔH = |D_low - D_high|
```

### 2.3 Diagnostic Patterns

| Condition | D_low | D_high | ΔH | Interpretation |
|-----------|-------|--------|----|--------------------|
| Healthy | ≈1.0 | ≈1.0 | Low | Balanced hierarchy |
| Schizophrenia | High | Normal | High | Rigid high-level priors |
| ASD | Low | High | High | Sensory dominance |
| Depression | High (negative) | Normal | High | Negative self-model |
| Anxiety | Normal | High (threat) | Moderate | Hypervigilant perception |
| Dissociation | Variable | Variable | Very High | Hierarchy disconnected |

---

## 3. tACS Stimulation Module

### 3.1 Why tACS over tDCS

| Property | tDCS | tACS |
|----------|------|------|
| Current type | Direct (DC) | Alternating (AC) |
| Mechanism | Shifts membrane potential | Entrains neural oscillations |
| Frequency target | None | Specific bands |
| Precision control | General excitability | Frequency-specific gain |

tACS allows **direct modulation of the oscillatory dynamics** that encode precision.

### 3.2 Hardware Specifications

```
┌─────────────────────────────────────────────────────────────┐
│                    tACS MODULE                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────┐    ┌────────────────┐                   │
│  │ DAC (12-bit)   │───►│ Current Source │───► Electrodes    │
│  │ AD5761R        │    │ Howland circuit│    (sponge/gel)   │
│  └────────────────┘    │ 0-2mA peak     │                   │
│         │              └────────────────┘                   │
│         │                                                    │
│  ┌──────▼───────┐     ┌────────────────┐                   │
│  │ Waveform Gen │     │ Impedance Mon  │◄── Safety         │
│  │ DDS, 0.1-100Hz│    │ Real-time      │    shutoff        │
│  └──────────────┘     └────────────────┘                   │
│                                                              │
│  Stimulation Sites:                                          │
│  • Frontal (F3/F4): θ-tACS for D_low modulation             │
│  • Parietal (P3/P4): γ-tACS for D_high modulation           │
│  • Temporal (T3/T4): α-tACS for attention gating            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 Frequency-Specific Protocols

| Target | Frequency | Electrode | Effect |
|--------|-----------|-----------|--------|
| ↓ D_low | 4-6 Hz (θ) | F3/F4 cathodal | Reduce prior precision |
| ↑ D_low | 4-6 Hz (θ) | F3/F4 anodal | Increase prior precision |
| ↓ D_high | 40 Hz (γ) | P3/P4 cathodal | Reduce sensory precision |
| ↑ D_high | 40 Hz (γ) | P3/P4 anodal | Increase sensory precision |
| Synchronize | 10 Hz (α) | Global | Hierarchical binding |

### 3.4 Safety Parameters

| Parameter | Limit | Monitoring |
|-----------|-------|------------|
| Peak current | 2 mA max | Real-time ammeter |
| Frequency | 0.1-100 Hz | Software lock |
| Duration | 20 min max/session | Timer |
| Impedance | <10 kΩ required | Auto-shutoff |
| Skin sensation | Minimal tingling | User report |

---

## 4. Context-Aware Feedback

### 4.1 State Detection

The device uses auxiliary sensors to detect user context:

```
CONTEXT DETECTION

┌─────────────────┐
│ Accelerometer   │──► Motion state: STILL / MOVING / EXERCISE
└─────────────────┘

┌─────────────────┐
│ PPG → HRV       │──► Arousal: LOW / MODERATE / HIGH
└─────────────────┘

┌─────────────────┐
│ Ambient light   │──► Environment: DARK / INDOOR / BRIGHT
└─────────────────┘

┌─────────────────┐
│ Time of day     │──► Circadian: MORNING / DAY / EVENING / NIGHT
└─────────────────┘
```

### 4.2 Adaptive Setpoints

| Context | D_target_low | D_target_high | Rationale |
|---------|--------------|---------------|-----------|
| Exercise | 0.8 | 1.2 | Favor sensory grounding |
| Creative work | 1.2 | 0.9 | Favor abstraction |
| Social | 1.0 | 1.0 | Balance |
| Sleep onset | 1.3 | 0.7 | Favor internal model |
| High stress | 0.9 | 0.9 | Reduce all precision (calm) |
| Meditation | 1.0→1.5 | 1.0→0.5 | Progressive depth |

### 4.3 Feedback Modality Selection

| D Problem | Context | Feedback Mode |
|-----------|---------|---------------|
| D_low too high | Any | Cognitive audio (grounding instructions) |
| D_low too low | Any | Abstract audio (visualization prompts) |
| D_high too high | Moving | Strong haptic pulses |
| D_high too high | Still | Gentle haptic + visual |
| D_high too low | Any | Sensory focus cues |
| ΔH high | Any | Synchronization protocol (α-tACS + audio) |

---

## 5. Multi-Objective Control Law

### 5.1 Objective Function

```
J = w_low × |D_low - D_target_low|²
  + w_high × |D_high - D_target_high|²
  + w_sync × ΔH²
  + w_smooth × |u - u_prev|²
```

Where:
- w_low, w_high: Weights on hierarchical targets (context-dependent)
- w_sync: Weight on hierarchical synchronization
- w_smooth: Smoothness penalty on interventions
- u: Control signal (feedback intensity, tACS parameters)

### 5.2 PI Control with Hierarchy

```python
class HierarchicalController:
    def __init__(self):
        self.Kp_low = 0.5
        self.Ki_low = 0.1
        self.Kp_high = 0.3
        self.Ki_high = 0.05

        self.integral_low = 0
        self.integral_high = 0

    def compute(self, D_low, D_high, targets, context):
        # Errors
        e_low = np.log(D_low) - np.log(targets['D_low'])
        e_high = np.log(D_high) - np.log(targets['D_high'])

        # Integrals
        self.integral_low += e_low
        self.integral_high += e_high

        # PI outputs
        u_low = self.Kp_low * e_low + self.Ki_low * self.integral_low
        u_high = self.Kp_high * e_high + self.Ki_high * self.integral_high

        # Map to interventions
        interventions = {
            'audio_cognitive': clip(-u_low, 0, 1),  # For high D_low
            'audio_grounding': clip(u_low, 0, 1),   # For low D_low
            'haptic_intensity': clip(u_high, 0, 1),
            'tacs_theta': clip(-u_low * 0.5, -1, 1),  # mA
            'tacs_gamma': clip(-u_high * 0.5, -1, 1),
        }

        # Context modulation
        if context['motion'] == 'EXERCISE':
            interventions['haptic_intensity'] *= 1.5
            interventions['audio_cognitive'] *= 0.5

        return interventions
```

---

## 6. Updated Clinical Protocols

### 6.1 Schizophrenia Spectrum

**Problem**: D_low >> 1 (rigid high-level priors)

**Protocol v2.0**:
1. **Phase 1 (weeks 1-2)**: Assessment
   - Map individual D_low/D_high profiles
   - Identify trigger contexts

2. **Phase 2 (weeks 3-6)**: θ-tACS + Neurofeedback
   - 4 Hz tACS to frontal cortex (reduce prior gain)
   - Audio feedback when D_low decreases
   - Target: D_low < 2.0

3. **Phase 3 (weeks 7-12)**: Maintenance
   - Reduce tACS, increase self-regulation
   - Context-aware alerts
   - Target: Sustained D_low ≈ 1.5

### 6.2 Autism Spectrum

**Problem**: D_high >> 1, D_low << 1

**Protocol v2.0**:
1. **Phase 1**: Controlled sensory environment
   - Reduce ambient stimulation
   - Baseline D_high measurement

2. **Phase 2**: Bidirectional training
   - γ-tACS (40 Hz) to reduce sensory gain
   - θ-tACS to increase prior reliance
   - Target: ΔH < 0.5

3. **Phase 3**: Generalization
   - Gradually increase environmental complexity
   - Adaptive setpoints based on overwhelm detection

### 6.3 Major Depressive Disorder (MDD)

**Problem**: High D_low for negative priors + low action-outcome precision

**Computational Model**:
- Excessive Π_prior on pessimistic self-models ("I am worthless")
- Beliefs resist updating from positive sensory evidence
- Degraded agency model (B matrix): actions feel futile

**Protocol v2.0**:
1. **Phase 1 (weeks 1-2)**: Assessment
   - Measure frontal θ asymmetry (left < right = depression marker)
   - Identify rigid negative prior signatures
   - Baseline anhedonia/apathy scales

2. **Phase 2 (weeks 3-8)**: Dual-target intervention
   - **θ-tACS (6 Hz) to left DLPFC**: Reduce Π_prior on negative beliefs
   - **β-tACS (20 Hz) to motor cortex**: Boost action-outcome precision
   - Audio prompts: "Notice small positive prediction errors"
   - Target: D_low_negative < 1.5, action-outcome coupling > baseline

3. **Phase 3 (weeks 9-12)**: Behavioral integration
   - Pair with behavioral activation therapy
   - Haptic reward for completed activities
   - Gradual tACS reduction

**Key Insight**: Depression involves not just sad mood but a *computational failure to update* from positive evidence. The intervention targets belief flexibility, not just affect.

### 6.4 Obsessive-Compulsive Disorder (OCD)

**Problem**: Low proprioceptive prior precision + failure to resolve prediction errors

**Computational Model**:
- Excessive Π_sensory: Sensory evidence of safety not trusted
- Low Π_prior on own state: "Did I really lock the door?"
- Checking generates MORE uncertainty, not less (positive feedback loop)
- Failure to reset error signal after successful action

**Protocol v2.0**:
1. **Phase 1 (weeks 1-2)**: Assessment
   - Map checking triggers and compulsion patterns
   - Identify specific uncertainty domains (contamination, safety, symmetry)
   - Baseline Y-BOCS scores

2. **Phase 2 (weeks 3-8)**: Proprioceptive precision boost
   - **α/β-tACS (10-20 Hz) to somatosensory/motor cortex**: Boost Π_prior on own actions
   - **Haptic "resolution signal"**: Distinct long pulse when checking resolves error
   - Target: Trust in own perception, reduced re-checking

3. **Phase 3 (weeks 9-12)**: Extinction training
   - App-guided ERP (Exposure-Response Prevention)
   - Resist re-checking after resolution signal
   - Audio: "Your action was successful. Trust the signal."

**Key Insight**: OCD is a *failure to acknowledge resolved prediction errors*. The haptic resolution signal creates a conditioned inhibitory cue against recursive checking.

### 6.5 Anxiety Disorders / PTSD

**Problem**: Aberrantly high Π_sensory on threat-related signals

**Computational Model**:
- Hypervigilant perception: Everything scanned for threat
- Threat priors over-weighted even in safe contexts
- Failed contextual modulation of precision

**Protocol v2.0**:
1. **Phase 1**: Safe environment baseline
   - Measure threat-related gamma activity
   - Identify trigger contexts

2. **Phase 2**: Contextual precision modulation
   - **α-tACS (10 Hz) to posterior cortex**: Reduce sensory gain globally
   - **Audio grounding**: "Notice you are safe in this moment"
   - Context-aware: Higher intervention intensity when HRV indicates stress

3. **Phase 3**: Trauma processing (PTSD only)
   - Integration with trauma-focused therapy (EMDR, CPT)
   - Device provides real-time D monitoring during exposure
   - Alert if D_high spikes beyond therapeutic window

### 6.6 Meditation / Peak Performance Enhancement

**Goal**: Maximize epistemic depth while maintaining stability

**Protocol v2.0**:
1. **Baseline**: Measure natural D_low/D_high trajectory
2. **Guided progression**:
   - D_low: 1.0 → 1.2 → 1.5 (deeper priors, abstraction)
   - D_high: 1.0 → 0.8 → 0.5 (reduced sensory, internal focus)
3. **Feedback**: Audio tone tracks epistemic depth
4. **Safety**: Alert if depth collapses (sleep onset) or destabilizes (anxiety)

---

## 7. Clinical Protocol Summary

| Condition | D_low Target | D_high Target | Primary tACS | Key Insight |
|-----------|--------------|---------------|--------------|-------------|
| Schizophrenia | ↓ to <2.0 | Maintain | θ (6Hz) frontal | Reduce prior rigidity |
| ASD | ↑ to >0.5 | ↓ to <1.5 | γ (40Hz) posterior | Balance hierarchy |
| Depression | ↓ negative priors | Maintain | θ (6Hz) left DLPFC + β motor | Update from positive evidence |
| OCD | ↑ proprioceptive | ↓ uncertainty | α/β (10-20Hz) motor | Trust resolved errors |
| Anxiety/PTSD | Maintain | ↓ threat | α (10Hz) posterior | Contextual safety |
| Meditation | ↑ to 1.5 | ↓ to 0.5 | None (feedback only) | Deepen epistemic depth |

---

## 8. Hardware BOM Update

| Component | v1.0 | v2.0 | Change |
|-----------|------|------|--------|
| EEG AFE | ADS1299 | ADS1299 | Same |
| MCU | STM32H7 | STM32H7 | Same |
| tDCS | Optional | Removed | - |
| **tACS DAC** | - | AD5761R | +$8 |
| **Current source** | - | Howland amp | +$5 |
| **DDS chip** | - | AD9833 | +$4 |
| Impedance monitor | Basic | Enhanced | +$3 |
| **Total BOM** | ~$115 | **~$135** | +$20 |
| **Retail** | $399-599 | **$499-699** | +$100 |

---

## 8. Firmware Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FIRMWARE v2.0                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    APPLICATION LAYER                        │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │ │
│  │  │ Hierarchical │  │ Context      │  │ Multi-objective  │  │ │
│  │  │ D Estimator  │──│ Detector     │──│ Controller       │  │ │
│  │  │ D_low, D_high│  │ Motion, HRV  │  │ PI + constraints │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    SIGNAL PROCESSING                        │ │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────┐   │ │
│  │  │ Band-pass  │  │ Phase-amp  │  │ Cross-frequency    │   │ │
│  │  │ θ/α/β/γ    │──│ coupling   │──│ coherence          │   │ │
│  │  └────────────┘  └────────────┘  └────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              │                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    DRIVER LAYER                             │ │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────────────┐   │ │
│  │  │ ADC    │  │ tACS   │  │ Audio  │  │ BLE/Telemetry  │   │ │
│  │  │ 256 Hz │  │ DDS    │  │ DAC    │  │ HUD bridge     │   │ │
│  │  └────────┘  └────────┘  └────────┘  └────────────────┘   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  FreeRTOS │ ARM Cortex-M7 │ 200 MHz │ 1MB RAM                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Research Questions for v2.0

1. **Does hierarchical D predict clinical outcomes better than single D?**
2. **Is θ-tACS more effective than generic tDCS for D_low?**
3. **Can context-aware setpoints improve user compliance?**
4. **Does ΔH correlate with Beautiful Loop epistemic depth?**
5. **Optimal tACS parameters for different disorder profiles?**

---

## 10. Regulatory Considerations

v2.0 with tACS requires:
- More rigorous safety testing (active stimulation)
- Class II medical device pathway (not wellness)
- Clinical trial data before approval
- Physician prescription required

---

## 11. Ethical Implications

### 11.1 The Power to Modify Belief-Reality Balance

A device that measures and modifies D—the balance between prior beliefs and sensory reality—raises profound ethical questions:

| Concern | Description | Mitigation |
|---------|-------------|------------|
| **Autonomy** | Who decides what D is "correct"? | User always controls targets; no forced intervention |
| **Authenticity** | Is a modified belief system still "you"? | Frame as expanding flexibility, not changing identity |
| **Consent** | Can vulnerable populations truly consent? | Capacity assessment; caregiver involvement |
| **Enhancement** | Should we allow D modification for "optimization"? | Restrict clinical protocols; wellness modes limited |

### 11.2 Dual-Use Risks

| Risk | Scenario | Safeguard |
|------|----------|-----------|
| **Coercive use** | Employer requires "optimal" D | Prescription-only; anti-coercion policy |
| **Manipulation** | Marketing to increase prior on brand | No third-party control; user data private |
| **Weaponization** | Military uses to reduce soldier uncertainty | Ethical review for defense contracts |
| **Addiction** | Dependence on device for "normal" function | Built-in tapering protocols; session limits |

### 11.3 Epistemic Justice

If precision imbalances correlate with socioeconomic factors:
- **Access equity**: Ensure device available to underserved populations
- **Cultural sensitivity**: D targets may vary across cultures
- **Avoid pathologizing difference**: ASD and other conditions have value; don't enforce neurotypical D

### 11.4 The "Delusion Index" Name

The term "Delusion Index" is scientifically descriptive but potentially stigmatizing:
- Consider alternative: "Belief-Evidence Balance" (BEB)
- Avoid implying that high D = "crazy"
- Frame deviations as *computational states*, not moral failings

### 11.5 Transparency Principles

1. **Open source**: Algorithms for D estimation should be auditable
2. **No hidden modification**: User always knows when intervention is active
3. **Data sovereignty**: User owns their precision data
4. **Research access**: Anonymized data available for independent study

### 11.6 Philosophical Implications

If D can be measured and modified:
- What does this say about "free will"?
- Is belief a computation that can be "corrected"?
- Where is the line between therapy and mind control?

These questions have no easy answers. The framework should:
- Engage philosophers, ethicists, and affected communities
- Proceed incrementally with extensive oversight
- Prioritize human dignity over optimization metrics

---

## 12. Future Directions

1. **Multi-modal integration**: Combine with fMRI for source localization
2. **Genetic markers**: Personalize protocols based on precision-related SNPs
3. **Social precision**: Model interpersonal belief dynamics (shared D)
4. **AI integration**: LLM-guided cognitive restructuring paired with tACS
5. **Longitudinal studies**: Does D modification produce lasting change?

---

*Document version: 2.0*
*Builds on: DEVICE_SPEC.md v1.0*
*Status: Conceptual upgrade proposal*
