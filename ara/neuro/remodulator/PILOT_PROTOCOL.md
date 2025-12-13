# NeuroBalance Pilot Clinical Protocol
## EXP-NB-001: Hierarchical Precision Metrics Validation

> **STATUS**: Pre-registration template for exploratory research
>
> This is NOT a proven treatment. This protocol tests whether the proposed
> metrics (D_low, D_high, ΔH) correlate with clinical phenomena and whether
> closed-loop stimulation affects those metrics in a measurable way.

---

## 1. Study Overview

### 1.1 Primary Objectives

1. **Metric Validation**: Do D_low and D_high reliably distinguish clinical groups?
2. **Test-Retest Reliability**: Are metrics stable within individuals across sessions?
3. **Stimulation Response**: Does tACS modulate the targeted metrics?
4. **Symptom Correlation**: Do metric changes track with symptom changes?

### 1.2 Study Design

**Phase 1** (N=60): Cross-sectional metric validation
- 20 healthy controls
- 20 schizophrenia spectrum
- 20 autism spectrum

**Phase 2** (N=30): Within-subject stimulation response
- 10 from each group
- Sham-controlled crossover design

### 1.3 Registration

Pre-register at OSF (osf.io) or ClinicalTrials.gov before data collection.
Specify primary outcomes, analysis plan, and stopping rules.

---

## 2. Inclusion/Exclusion Criteria

### 2.1 Healthy Controls

| Criterion | Specification |
|-----------|---------------|
| Age | 18-55 years |
| Psychiatric history | None (SCID-5 screen negative) |
| Neurological | No seizure history, no TBI |
| Medications | No psychotropic medications |
| Handedness | Right-handed (for consistency) |
| Vision/Hearing | Normal or corrected |

### 2.2 Schizophrenia Spectrum

| Criterion | Specification |
|-----------|---------------|
| Diagnosis | DSM-5 schizophrenia or schizoaffective |
| Confirmation | SCID-5 + chart review |
| Symptom stability | Stable medications ≥4 weeks |
| PANSS | Total 50-100 (moderate severity) |
| Exclusions | Active substance use, ECT in past 6 months |

### 2.3 Autism Spectrum

| Criterion | Specification |
|-----------|---------------|
| Diagnosis | DSM-5 ASD, confirmed by ADOS-2 |
| IQ | ≥70 (verbal and nonverbal) |
| Age at diagnosis | Any (adult or childhood) |
| Comorbidities | Exclude active psychosis, mania |
| Sensory | Document sensory sensitivities |

### 2.4 General Exclusions (All Groups)

- Implanted devices (pacemaker, DBS, cochlear)
- Skull defects or metal implants
- Pregnancy
- Skin conditions at electrode sites
- Inability to tolerate EEG cap for 90 minutes

---

## 3. Metrics and Measurements

### 3.1 Primary Metrics (EEG-Derived)

| Metric | Derivation | Expected Pattern |
|--------|------------|------------------|
| **D_low** | θ_frontal × (1 + θ-γ coupling) / (α_post + 0.5×β_frontal) | SCZ > HC > ASD |
| **D_high** | γ_post × (1 + 0.3×β_frontal) / (γ_post + 0.5×α_post) | ASD > HC > SCZ |
| **ΔH** | \|D_low - D_high\| | SCZ, ASD > HC |

### 3.2 EEG Acquisition

| Parameter | Specification |
|-----------|---------------|
| System | 14-channel NeuroBalance prototype or BioSemi 64-ch |
| Sampling | 512 Hz, 24-bit |
| Reference | Linked mastoids |
| Impedance | <10 kΩ |
| Duration | 5 min resting (eyes open), 5 min (eyes closed) |
| Artifact rejection | ICA for blinks, ASR for motion |

### 3.3 Frequency Bands

| Band | Range | Regions |
|------|-------|---------|
| Theta (θ) | 4-8 Hz | Fp1, Fp2, F3, F4 (frontal) |
| Alpha (α) | 8-12 Hz | P3, P4, Pz, O1, O2 (posterior) |
| Beta (β) | 12-30 Hz | F3, F4, C3, C4 |
| Gamma (γ) | 30-80 Hz | P3, P4, O1, O2 |

### 3.4 Phase-Amplitude Coupling

Compute θ-γ coupling using:
- Modulation Index (Tort et al., 2010)
- Phase from frontal θ, amplitude from posterior γ
- Report as z-score against surrogate distribution

### 3.5 Secondary Metrics

| Metric | Source | Purpose |
|--------|--------|---------|
| PANSS | Interview | Schizophrenia symptoms |
| AQ-10 | Self-report | Autism traits |
| DASS-21 | Self-report | Depression, anxiety, stress |
| SRS-2 | Informant | Social responsiveness |
| SPQ | Self-report | Sensory processing |

---

## 4. Behavioral Tasks

### 4.1 Precision Estimation Paradigm

**Roving MMN Task** (auditory oddball with changing standards)

- Estimates precision updating from P3a/MMN amplitude
- Prediction: SCZ shows reduced MMN (aberrant precision)
- Duration: 15 minutes

### 4.2 Bistable Perception Task

**Necker Cube / Binocular Rivalry**

- Measures prior-sensory balance from switch rate
- Prediction: SCZ = slower switches (strong priors), ASD = faster
- Duration: 10 minutes

### 4.3 Sensory Gating Task

**Prepulse Inhibition (PPI)**

- Measures sensory filtering
- Prediction: SCZ = reduced PPI
- Duration: 10 minutes

### 4.4 Rubber Hand Illusion (Optional)

- Measures body model precision
- Prediction: SCZ = enhanced illusion, ASD = reduced

---

## 5. Stimulation Protocol (Phase 2)

### 5.1 tACS Parameters

| Parameter | Specification |
|-----------|---------------|
| Device | NeuroBalance v2.0 prototype |
| Waveform | Sinusoidal |
| Current | 1.5 mA peak-to-peak |
| Duration | 20 minutes |
| Ramp | 30 seconds on/off |
| Sessions | 2 per participant (active + sham), 1 week apart |

### 5.2 Stimulation Conditions

**Condition A: Theta-tACS (6 Hz)**
- Electrodes: F3/F4 (anodal) ↔ Pz (return)
- Target: Reduce D_low in SCZ group

**Condition B: Gamma-tACS (40 Hz)**
- Electrodes: P3/P4 (anodal) ↔ Fz (return)
- Target: Reduce D_high in ASD group

**Condition C: Sham**
- 30 seconds ramp-on, then off (maintain blinding)
- Same electrode placement as active condition

### 5.3 Crossover Design

```
Week 1          Week 2
─────────────────────────────
Group A:  Active  →  Sham
Group B:  Sham    →  Active
```

Randomization stratified by diagnosis.

### 5.4 Blinding Assessment

Post-session questionnaire:
- "Do you think you received active or sham stimulation?"
- Record confidence (1-5)
- Check if blinding maintained (>60% incorrect = good blinding)

---

## 6. Pre-Registered Hypotheses

### 6.1 Primary Hypotheses (Phase 1)

| # | Hypothesis | Test | α |
|---|------------|------|---|
| H1 | D_low differs between groups | One-way ANOVA | 0.05 |
| H1a | D_low: SCZ > HC | Post-hoc t-test | 0.017 |
| H1b | D_low: ASD < HC | Post-hoc t-test | 0.017 |
| H2 | D_high differs between groups | One-way ANOVA | 0.05 |
| H2a | D_high: ASD > HC | Post-hoc t-test | 0.017 |
| H3 | ΔH > 0.5 in clinical groups | One-sample t-test | 0.05 |

### 6.2 Secondary Hypotheses (Phase 1)

| # | Hypothesis | Test |
|---|------------|------|
| H4 | D_low correlates with PANSS positive | Pearson r |
| H5 | D_high correlates with SPQ sensory subscale | Pearson r |
| H6 | Test-retest ICC > 0.7 for D_low, D_high | ICC(3,1) |

### 6.3 Stimulation Hypotheses (Phase 2)

| # | Hypothesis | Test |
|---|------------|------|
| H7 | θ-tACS reduces D_low (pre vs post) | Paired t-test |
| H8 | γ-tACS reduces D_high (pre vs post) | Paired t-test |
| H9 | Active > Sham for metric change | Mixed ANOVA |

---

## 7. Power Analysis

### Phase 1 (Metric Validation)

| Effect | Expected d | N per group | Power |
|--------|------------|-------------|-------|
| D_low: SCZ vs HC | 0.8 (large) | 20 | 0.80 |
| D_high: ASD vs HC | 0.8 (large) | 20 | 0.80 |

Justification: Based on Lawson et al. (2014) precision differences.

### Phase 2 (Stimulation)

| Effect | Expected d | N total | Power |
|--------|------------|---------|-------|
| Pre-post metric change | 0.5 (medium) | 30 | 0.75 |

Note: Exploratory—effect size uncertain.

---

## 8. Analysis Plan

### 8.1 Preprocessing

1. Band-pass filter: 1-100 Hz
2. Notch filter: 50/60 Hz
3. ICA: Remove blinks, saccades
4. ASR: Remove transient artifacts (threshold: 20 SD)
5. Epoch: -200 to 800 ms for tasks, 2s for resting
6. Reject: >100 µV amplitude

### 8.2 Metric Computation

```python
def compute_metrics(eeg_data):
    # Band power (Welch, 2s windows, 50% overlap)
    theta_frontal = bandpower(eeg['F3','F4'], 4, 8)
    alpha_post = bandpower(eeg['P3','P4','O1','O2'], 8, 12)
    gamma_post = bandpower(eeg['P3','P4','O1','O2'], 30, 80)
    beta_frontal = bandpower(eeg['F3','F4'], 12, 30)

    # Phase-amplitude coupling
    pac = compute_pac(eeg['Fz'], theta_band, eeg['Pz'], gamma_band)

    # Metrics
    D_low = theta_frontal * (1 + pac) / (alpha_post + 0.5 * beta_frontal)
    D_high = gamma_post * (1 + 0.3 * beta_frontal) / (gamma_post + 0.5 * alpha_post)
    delta_H = abs(D_low - D_high)

    return D_low, D_high, delta_H
```

### 8.3 Statistical Analysis

| Analysis | Software | Method |
|----------|----------|--------|
| Group comparison | R (lme4) | Mixed-effects ANOVA |
| Correlations | R | Pearson with bootstrapped CI |
| Reliability | R (psych) | ICC(3,1) |
| Effect sizes | R | Cohen's d with Hedges correction |

### 8.4 Multiple Comparison Correction

- Primary hypotheses: Bonferroni within family
- Exploratory: FDR (Benjamini-Hochberg)

---

## 9. Safety Monitoring

### 9.1 Adverse Events

| Event | Action |
|-------|--------|
| Skin irritation | Discontinue, document, refer if needed |
| Headache | Rest, document, continue if resolves |
| Mood change | Assess severity, report to IRB if significant |
| Seizure | Emergency protocol, exclude, report |

### 9.2 Stopping Rules

- Stop individual: 2+ moderate adverse events
- Stop study: Serious adverse event in >5% of participants

### 9.3 Data Safety Monitoring

- Independent safety monitor reviews after N=15, 30
- Unblinded interim analysis only for safety

---

## 10. Ethical Considerations

### 10.1 Informed Consent

- Plain-language description of procedures
- Emphasize: This is research, not treatment
- Describe risks: Skin sensation, possible headache
- Right to withdraw at any time

### 10.2 Vulnerable Populations

- Schizophrenia: Assess capacity, involve caregiver if appropriate
- ASD: Accommodate sensory needs, allow breaks
- All: Avoid coercion (no excessive compensation)

### 10.3 Data Protection

- De-identified EEG stored encrypted
- No re-identification without consent
- Retention: 10 years per institutional policy

---

## 11. Success Criteria

### What Would Constitute Initial Evidence?

| Outcome | Threshold | Interpretation |
|---------|-----------|----------------|
| H1 confirmed (D_low group diff) | p < 0.05, d > 0.5 | Metric has construct validity |
| H6 confirmed (ICC > 0.7) | ICC > 0.7 | Metric is reliable |
| H7 confirmed (tACS changes D_low) | p < 0.05, d > 0.3 | Metric is modifiable |
| H4 confirmed (symptom correlation) | r > 0.3, p < 0.05 | Metric clinically relevant |

### What Would Refute the Framework?

| Outcome | Threshold | Interpretation |
|---------|-----------|----------------|
| No group differences | p > 0.2, d < 0.2 | Metrics don't capture clinical variation |
| Poor reliability | ICC < 0.5 | Metrics too noisy for individual use |
| Sham = Active | p > 0.5 | tACS doesn't modulate metrics |

---

## 12. Timeline

| Phase | Duration | Milestone |
|-------|----------|-----------|
| IRB approval | 2 months | Protocol approved |
| Recruitment | 4 months | N=60 enrolled |
| Phase 1 data collection | 4 months | All baseline complete |
| Phase 1 analysis | 2 months | Primary results |
| Phase 2 data collection | 3 months | Stimulation complete |
| Phase 2 analysis | 2 months | Full results |
| Manuscript | 3 months | Submitted |

**Total**: ~20 months

---

## 13. Team and Resources

| Role | Expertise |
|------|-----------|
| PI | Clinical neuroscience, psychiatry |
| Co-I | EEG signal processing, tES |
| Co-I | Computational psychiatry |
| Coordinator | Recruitment, scheduling |
| RA | Data collection, preprocessing |
| Statistician | Analysis, power |

### Equipment

- NeuroBalance v2.0 prototype OR BioSemi + Neuroconn tACS
- Stimulus computer (PsychoPy)
- Shielded EEG room

### Budget Estimate

| Item | Cost |
|------|------|
| Equipment | $15,000 |
| Participant compensation | $6,000 |
| Personnel | $50,000 |
| Analysis/Software | $2,000 |
| Publication | $3,000 |
| **Total** | **~$76,000** |

---

## 14. Limitations and Caveats

1. **Exploratory**: Effect sizes uncertain; underpowered for small effects
2. **Generalizability**: Single site, specific inclusion criteria
3. **Metric validity**: D_low/D_high are model-derived, not gold standard
4. **Stimulation specificity**: tACS effects may be non-specific
5. **Blinding**: Participants may detect stimulation despite sham

---

## References

1. Lawson, R.P., et al. (2014). Adults with autism overestimate the volatility of the sensory environment. *Nature Neuroscience*.
2. Adams, R.A., et al. (2013). The computational anatomy of psychosis. *Frontiers in Psychiatry*.
3. Tort, A.B., et al. (2010). Measuring phase-amplitude coupling between neuronal oscillations. *Journal of Neurophysiology*.
4. Antal, A., & Herrmann, C.S. (2016). Transcranial alternating current and random noise stimulation. *Journal of Neuroscience*.

---

*Document version: 0.1*
*Status: Pre-registration template*
*Requires: IRB approval, institutional support, clinical collaborators*
