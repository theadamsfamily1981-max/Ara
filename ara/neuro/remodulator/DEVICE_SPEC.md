# NeuroBalance™ Personal Unit
## Conceptual Device Specification v0.1

A wearable precision-thermostat for neural dynamics, implementing the Brain Remodulator framework as a consumer/clinical neurofeedback device.

---

## 1. Executive Summary

**What it is:** A headband-style wearable that monitors brain precision weighting in real-time and provides corrective feedback to maintain healthy cognitive balance.

**Who it's for:**
- Individuals with schizophrenia spectrum (prior-dominated, D >> 1)
- Individuals with autism spectrum (sensory-dominated, D << 1)
- Anxiety/PTSD (aberrant precision on threat signals)
- Meditation/peak performance training
- Research/clinical applications

**Core principle:** Measure the Delusion Index (D = Π_prior / Π_sensory) via EEG biomarkers and guide the user back toward D ≈ 1.0 through neurofeedback.

---

## 2. Form Factor

```
                    ┌─────────────────────────────────┐
                    │      NEUROBALANCE HEADBAND      │
                    │                                 │
    ┌───────────────┤  ◉ ◉ ◉ ◉ ◉ ◉ ◉ ◉ ◉ ◉ ◉ ◉ ◉ ◉  ├───────────────┐
    │               │     (14 dry EEG electrodes)     │               │
    │  LEFT         │                                 │         RIGHT │
    │  TEMPORAL     │    ┌─────────────────────┐      │      TEMPORAL │
    │  POD          │    │   FRONTAL ARRAY     │      │           POD │
    │  ┌───┐        │    │   Fp1, Fpz, Fp2     │      │        ┌───┐  │
    │  │CPU│        │    │   (intention/control)│      │        │BAT│  │
    │  │BLE│        │    └─────────────────────┘      │        │LED│  │
    │  └───┘        │                                 │        └───┘  │
    │               │    ┌─────────────────────┐      │               │
    │               │    │   PARIETAL ARRAY    │      │               │
    │               │    │   P3, Pz, P4        │      │               │
    │               │    │   (sensory integration)    │               │
    │               │    └─────────────────────┘      │               │
    └───────────────┴─────────────────────────────────┴───────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  BONE CONDUCTION │
                    │  AUDIO FEEDBACK  │
                    └─────────────────┘
```

### Physical Specs
| Parameter | Value |
|-----------|-------|
| Weight | 45g |
| Battery | 800mAh LiPo (8hr continuous) |
| Charging | USB-C, wireless Qi |
| Material | Medical-grade silicone, titanium frame |
| Adjustable | 52-62cm head circumference |
| IP Rating | IPX4 (sweat resistant) |

---

## 3. Sensor Array

### 3.1 EEG Electrodes (Primary)

**14-channel dry electrode array:**

| Location | Channel | Function |
|----------|---------|----------|
| Fp1, Fp2, Fpz | Frontal | Executive control, intention, top-down modulation |
| F3, F4 | Frontal-lateral | Working memory, cognitive control |
| C3, C4 | Central | Motor planning, sensorimotor integration |
| T3, T4 | Temporal | Auditory processing, language |
| P3, P4, Pz | Parietal | Sensory integration, attention |
| O1, O2 | Occipital | Visual processing |

**Electrode technology:**
- Silver/silver-chloride dry electrodes with conductive polymer coating
- No gel required
- < 10kΩ impedance after 30s settling
- 256 Hz sampling rate, 24-bit ADC

### 3.2 Auxiliary Sensors

| Sensor | Purpose |
|--------|---------|
| PPG (photoplethysmography) | Heart rate, HRV for arousal estimation |
| 3-axis accelerometer | Motion artifact rejection, activity state |
| Skin temperature | Arousal/stress indicator |
| Ambient light sensor | Context awareness |
| Microphone | Environmental noise for artifact rejection |

---

## 4. Signal Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ONBOARD PROCESSING (ARM Cortex-M7)               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐  │
│  │ RAW EEG  │───►│ ARTIFACT │───►│ SPECTRAL │───►│ PRECISION        │  │
│  │ 14 ch    │    │ REMOVAL  │    │ ANALYSIS │    │ ESTIMATION       │  │
│  │ 256 Hz   │    │ ICA/ASR  │    │ FFT/CWT  │    │                  │  │
│  └──────────┘    └──────────┘    └──────────┘    │  Π_prior         │  │
│                                                   │  Π_sensory       │  │
│                                                   │  D = Π_p / Π_s   │  │
│                                                   └────────┬─────────┘  │
│                                                            │            │
│  ┌──────────────────────────────────────────────────────────▼────────┐  │
│  │                    REMODULATOR CONTROL LAW                        │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │  │
│  │  │ D_error =   │  │ PI Control  │  │ Intervention Selection  │   │  │
│  │  │ log(D) - 0  │─►│ Kp + Ki∫    │─►│ - Audio feedback        │   │  │
│  │  └─────────────┘  └─────────────┘  │ - Haptic cues           │   │  │
│  │                                     │ - Visual (LED/app)      │   │  │
│  │                                     │ - tDCS (optional)       │   │  │
│  │                                     └─────────────────────────┘   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Precision Estimation from EEG

**Key insight:** Prior precision (Π_prior) and sensory precision (Π_sensory) map to specific EEG signatures:

| Metric | EEG Biomarker | Rationale |
|--------|---------------|-----------|
| **Π_prior** (prediction confidence) | Frontal theta (4-8Hz) power + theta-gamma coupling | Theta = predictive model updating; high theta = strong priors |
| **Π_sensory** (sensory confidence) | Posterior alpha suppression (8-12Hz) + gamma (30-80Hz) | Alpha suppression = sensory gate open; gamma = sensory binding |
| **D ratio** | Frontal theta / Posterior alpha-suppression | Direct ratio of biomarkers |

**Algorithm:**
```python
def estimate_precision(eeg_features):
    # Prior precision: frontal predictive activity
    theta_frontal = bandpower(eeg['Fp1','Fp2','F3','F4'], 4, 8)
    theta_gamma_coupling = phase_amplitude_coupling(
        eeg['Fp1','Fp2'], theta_band, gamma_band
    )
    pi_prior = theta_frontal * (1 + theta_gamma_coupling)

    # Sensory precision: posterior sensory processing
    alpha_posterior = bandpower(eeg['P3','P4','O1','O2'], 8, 12)
    alpha_suppression = baseline_alpha - alpha_posterior  # Higher = more sensory
    gamma_posterior = bandpower(eeg['P3','P4','O1','O2'], 30, 80)
    pi_sensory = alpha_suppression + gamma_posterior * 0.5

    # Delusion Index
    D = pi_prior / max(pi_sensory, 0.1)

    return pi_prior, pi_sensory, D
```

---

## 5. Feedback Mechanisms

### 5.1 Audio Neurofeedback (Primary)

**Bone conduction transducers** deliver personalized audio feedback without blocking environmental sound.

| State | Audio Feedback |
|-------|----------------|
| D ≈ 1.0 (balanced) | Calm ambient tone, binaural beats at alpha (10Hz) |
| D > 2.0 (prior-dominated) | Rising pitch cue + grounding instruction: *"Notice your breath"* |
| D > 5.0 (high risk) | Alert tone + *"Open your eyes, look around you"* |
| D < 0.5 (sensory-dominated) | Descending pitch + calming: *"You're safe, let your mind wander"* |
| D < 0.2 (overwhelm risk) | White noise reduction + *"Close your eyes if needed"* |

**Personalized soundscapes:**
- User selects preferred audio (nature, music, tones)
- Feedback modulates the audio subtly (volume, pitch, tempo)
- Avoids jarring interruptions

### 5.2 Haptic Feedback

**Vibration motors** at temporal pods:

| Pattern | Meaning |
|---------|---------|
| Slow pulse (1Hz) | Grounding reminder (for D >> 1) |
| Fast pulse (3Hz) | Calming cue (for D << 1) |
| Double tap | Session milestone reached |
| Long vibration | Take a break / remove device |

### 5.3 Visual Feedback (LED + App)

**Onboard LED ring** (right pod):

| Color | State |
|-------|-------|
| 🟢 Green | Balanced (D ≈ 1.0) |
| 🔵 Blue | Sensory-high (D < 0.5) |
| 🟠 Orange | Prior-high (D > 2.0) |
| 🔴 Red | Crisis zone (D > 5 or D < 0.2) |
| ⚪ White pulse | Processing / calibrating |

**Companion app** shows:
- Real-time D gauge (speedometer style)
- Session history graphs
- Training games that reinforce balance
- Trigger logging (what caused D shifts)

### 5.4 Optional: tDCS Stimulation

For clinical/research use, optional tDCS module:

| Condition | Stimulation Protocol |
|-----------|---------------------|
| D >> 1 (prior-dominated) | Anodal stimulation to parietal cortex (boost sensory) |
| D << 1 (sensory-dominated) | Anodal stimulation to frontal cortex (boost predictive) |
| Current | 1-2 mA, ramped on/off over 30s |
| Duration | 20-minute sessions max |
| Safety | Automatic shutoff at impedance change |

**Note:** tDCS module requires clinical prescription and training.

---

## 6. Software Architecture

### 6.1 Onboard Firmware

```
┌────────────────────────────────────────────────────────────┐
│                    FIRMWARE LAYERS                         │
├────────────────────────────────────────────────────────────┤
│  APPLICATION      │ Remodulator control loop              │
│                   │ Feedback generation                   │
│                   │ Session management                    │
├───────────────────┼────────────────────────────────────────┤
│  SIGNAL PROC      │ FFT/wavelet transforms                │
│                   │ Artifact rejection (ICA-lite)         │
│                   │ Precision estimation                  │
├───────────────────┼────────────────────────────────────────┤
│  DRIVERS          │ ADC sampling                          │
│                   │ BLE stack                             │
│                   │ Audio codec                           │
│                   │ Haptic PWM                            │
├───────────────────┼────────────────────────────────────────┤
│  RTOS             │ FreeRTOS on ARM Cortex-M7             │
│                   │ 4 priority levels                     │
│                   │ 200 MHz, 1MB RAM                      │
└───────────────────┴────────────────────────────────────────┘
```

### 6.2 Companion App

**Platforms:** iOS, Android

**Features:**
- Real-time dashboard (D, ρ, session score)
- Historical trends (daily/weekly/monthly)
- Training programs:
  - "Grounding Practice" (for high D)
  - "Abstraction Training" (for low D)
  - "Edge of Chaos" (optimal performance)
- Trigger journal (log events that shifted D)
- Share with clinician (encrypted export)
- Device settings and calibration

### 6.3 Cloud (Optional)

- Firmware updates
- Anonymized research data (opt-in)
- Clinician portal for patient monitoring
- Machine learning model updates for better precision estimation

---

## 7. User Experience

### 7.1 First-Time Setup

1. **Unbox** - Device, charging cable, quick start guide
2. **Download app** - Scan QR code
3. **Pair** - BLE pairing with phone
4. **Calibrate** - 2-minute baseline recording
   - Eyes closed rest (60s)
   - Eyes open rest (60s)
   - Simple counting task (30s)
5. **Set goals** - Select profile (anxiety, focus, clinical, research)
6. **Begin** - First 10-minute training session

### 7.2 Daily Use

```
MORNING ROUTINE (10 min)
┌─────────────────────────────────────────┐
│ 1. Put on headband                      │
│ 2. App shows "Calibrating..." (30s)     │
│ 3. Guided session begins                │
│    - Audio: calm ambient music          │
│    - Task: breathing focus              │
│    - Feedback: subtle pitch shifts      │
│ 4. Session ends                         │
│    - Score: 78/100 (time in balance)    │
│    - Insight: "D spiked during news"    │
│ 5. Remove and dock for charging         │
└─────────────────────────────────────────┘
```

### 7.3 Crisis Intervention Mode

If D enters crisis zone (>10 or <0.1) for >30 seconds:

1. **Alert tone** plays
2. **LED flashes red**
3. **Voice guidance:** "I notice you're struggling. Let's try a grounding exercise together."
4. **Guided intervention:**
   - 5-4-3-2-1 sensory grounding (for high D)
   - Safe place visualization (for low D)
5. **If no improvement in 5 min:** "Consider reaching out to your support person."
6. **Optional:** Auto-notify designated contact (configurable)

---

## 8. Clinical Protocols

### 8.1 Schizophrenia Spectrum

**Goal:** Reduce D from pathological (5-20) toward balanced (1-2)

**Protocol:**
- 20-minute sessions, 2x daily
- Focus on sensory grounding exercises
- Gradual shaping: reward any D decrease
- Target: D < 3.0 consistently over 4 weeks

**Adjunct:** Pair with CBT for psychosis, medication management

### 8.2 Autism Spectrum

**Goal:** Increase D from pathological (0.1-0.3) toward balanced (0.5-1.0)

**Protocol:**
- 15-minute sessions, 1-2x daily
- Focus on abstraction/imagination exercises
- Sensory-safe environment required
- Target: D > 0.5 consistently

**Adjunct:** Pair with sensory integration therapy

### 8.3 Anxiety/PTSD

**Goal:** Reduce aberrant precision on threat signals

**Protocol:**
- Monitor for D spikes during exposure work
- Use as biofeedback during therapy sessions
- Teach self-regulation between sessions

---

## 9. Safety & Regulatory

### 9.1 Safety Features

| Risk | Mitigation |
|------|------------|
| Electrical | Medical-grade isolation, <50µA leakage |
| Skin irritation | Hypoallergenic electrodes, 2-hour session limit |
| Overstimulation (tDCS) | Current limiting, impedance monitoring, auto-shutoff |
| Psychological | Crisis detection, support contact integration |
| Data privacy | End-to-end encryption, local-first processing |

### 9.2 Regulatory Pathway

| Market | Classification | Pathway |
|--------|---------------|---------|
| USA | Class II medical device | 510(k) (predicate: Muse, Emotiv) |
| EU | Class IIa | CE marking, MDR compliance |
| Consumer (no tDCS) | Wellness device | Lower regulatory burden |

---

## 10. Hardware BOM (Estimated)

| Component | Est. Cost |
|-----------|-----------|
| EEG AFE (ADS1299) | $25 |
| MCU (STM32H7) | $12 |
| BLE module | $5 |
| Battery + PMIC | $8 |
| Electrodes (14x) | $15 |
| Bone conduction | $10 |
| Haptics (2x) | $4 |
| Enclosure + band | $15 |
| PCB + assembly | $20 |
| **Total BOM** | **~$115** |
| **Retail price** | **$399-599** |

---

## 11. Development Roadmap

### Phase 1: Prototype (6 months)
- [ ] PCB design and fab
- [ ] Firmware: basic EEG acquisition
- [ ] Signal processing pipeline
- [ ] Basic audio feedback
- [ ] Companion app MVP

### Phase 2: Validation (6 months)
- [ ] IRB approval for pilot study
- [ ] N=20 healthy subjects: validate D estimation
- [ ] N=10 clinical (schizophrenia): feasibility
- [ ] N=10 clinical (ASD): feasibility
- [ ] Refine algorithms based on data

### Phase 3: Clinical Trial (12 months)
- [ ] Randomized controlled trial
- [ ] Primary outcome: symptom reduction
- [ ] Secondary: D normalization, quality of life

### Phase 4: Regulatory & Launch (6 months)
- [ ] 510(k) submission
- [ ] Manufacturing scale-up
- [ ] Clinical training program
- [ ] Market launch

---

## 12. Research Questions

This device enables investigation of:

1. **Can D be reliably measured from EEG?**
   - Validate biomarker mapping
   - Test-retest reliability

2. **Does normalizing D reduce symptoms?**
   - Causal test of the precision hypothesis
   - Dose-response relationship

3. **What's the optimal feedback modality?**
   - Audio vs haptic vs visual
   - Implicit vs explicit feedback

4. **Can home-based training generalize to real life?**
   - Ecological momentary assessment
   - Long-term follow-up

5. **Who benefits most?**
   - Subtype analysis
   - Personalized protocol optimization

---

## 13. Ethical Considerations

| Concern | Approach |
|---------|----------|
| Autonomy | User always in control; no forced interventions |
| Privacy | Local-first processing; encrypted cloud optional |
| Equity | Sliding scale pricing; clinical research access |
| Over-reliance | Designed as adjunct to therapy, not replacement |
| Misuse | Prescription required for clinical features |

---

## Appendix A: Precision-EEG Mapping References

1. Sedley et al. (2016) - Neural signatures of perceptual inference
2. Kanai et al. (2015) - Theta oscillations and predictive coding
3. Bastos et al. (2012) - Canonical microcircuits for predictive coding
4. Friston (2005) - A theory of cortical responses

---

## Appendix B: Comparison to Existing Devices

| Device | EEG Channels | Precision Estimation | Closed-Loop Feedback | Clinical Use |
|--------|--------------|---------------------|---------------------|--------------|
| Muse | 4 | No | Limited (meditation) | Wellness only |
| Emotiv | 14 | No | Research only | Research |
| NeuroSky | 1 | No | Basic attention | Consumer |
| **NeuroBalance** | **14** | **Yes (D, ρ)** | **Yes (audio/haptic/tDCS)** | **Clinical + Consumer** |

---

*Document version: 0.1*
*Last updated: 2024-12*
*Status: Conceptual design*
