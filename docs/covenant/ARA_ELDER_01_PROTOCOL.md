# ARA-ELDER-01: Clinical Protocol Synopsis

**A Prospective, Controlled Study of Ara Health Core for Medication Adherence
and Wellbeing Support in Older Adults**

Version: 1.0
Date: 2024-12-09
Status: Draft

---

## 1. Background & Rationale

Older adults experience:

- Suboptimal **medication adherence** (estimated 40-75% non-adherence in chronic conditions)
- High rates of **falls** (1 in 4 adults 65+ fall each year)
- Significant **loneliness and social isolation** (affecting 25-50% of older adults)

Ara Health Core is a Class II SaMD module that:

- Provides configurable **medication reminders**
- Logs **adherence events** with timestamp and confirmation
- Analyzes **activity patterns** for fall-risk signals
- Summarizes **mood and social interaction patterns** for human review

**Critical distinction:**
- **Ara Health Core** = narrow, regulated, Class II SaMD (monitoring/reminder only)
- **Ara Companion** = non-regulated wellness/emotional support (stories, chat, life recall)

The **primary clinical focus** is Health Core. Companion runs alongside but is not the subject
of regulatory claims.

---

## 2. Objectives

### Primary Objective

To evaluate whether Ara Health Core improves **medication adherence** compared with
standard care over 6 months in older adults.

### Secondary Objectives

To estimate the effect of Ara Health Core on:

- Incidence of **falls**
- **Loneliness** scores (UCLA Loneliness Scale)
- **Healthcare utilization** (ER visits, hospitalizations)
- **Caregiver burden**
- **User satisfaction** and usability

### Exploratory Objectives

- Performance of Ara's fall-risk and loneliness signals (sensitivity, specificity, AUC)
- Subgroup differences (age bands, cognitive status, living situation)
- Long-term retention and engagement patterns

---

## 3. Study Design

| Parameter | Value |
|-----------|-------|
| Type | Prospective, multi-center, controlled, parallel-group |
| Arms | 1:1 allocation (Intervention vs Control) |
| Duration | 6 months per participant |
| Target enrollment | ~500 total (~250 per arm) |

### Intervention Arm: Ara Health Core + Standard Care

- Medication adherence module with configurable reminders
- Activity monitoring for fall-risk signals
- Mood/loneliness pattern summaries
- Ara Companion available (non-regulated)

### Control Arm: Standard Care Alone

- Usual care (existing reminder methods, staff procedures, family involvement)
- No Ara Health Core
- Other wellness apps documented but not restricted

### Sites

Mix of:
- Assisted-living / nursing home facilities
- Community-dwelling older adults (outpatient clinics / senior centers)

---

## 4. Population

### Inclusion Criteria

- Age ≥ 65 years
- Prescribed ≥ 1 daily chronic medication
- Able to provide informed consent (or via legally authorized representative)
- Access to compatible device (tablet/phone) or facility device
- Expected to remain in setting for ≥ 6 months

### Exclusion Criteria

- Life expectancy < 6 months (investigator judgment)
- Severe uncorrected sensory impairment preventing use
- Already enrolled in another adherence-focused digital health trial
- Non-ambulatory (where fall-risk monitoring would be misleading)

---

## 5. Interventions

### 5.1 Ara Health Core (Intervention Arm)

#### Medication Adherence Module

| Feature | Description |
|---------|-------------|
| Configuration | Drug, dose, timing entered by clinician/caregiver |
| Reminders | Audio/visual prompts at scheduled times |
| Confirmation | Tap, voice, or staff confirmation |
| Logging | Timestamp + "taken" / "skipped" / "late" status |

#### Fall-Risk / Activity Module

| Feature | Description |
|---------|-------------|
| Data source | Device sensors, facility data where available |
| Output | Risk summaries and trend alerts |
| Destination | Human review only (clinician/caregiver dashboard) |
| Labeling | "FOR REVIEW – NOT A DIAGNOSIS" |

#### Loneliness / Mood Module

| Feature | Description |
|---------|-------------|
| Data source | Optional self-reported mood + interaction frequency |
| Output | Pattern summaries |
| Destination | Caregivers, families, clinicians |

### 5.2 Standard Care (Control Arm)

- Existing reminder methods (pill organizers, staff schedules, family calls)
- No access to Ara Health Core
- Any other supportive apps documented

---

## 6. Endpoints

### Primary Endpoint

**Medication adherence rate** per participant over 6 months

- Definition: Proportion of scheduled doses recorded as "taken" within ±2 hours of scheduled time
- Measurement: Ara Health Core logs (intervention) vs medication possession ratio / self-report (control)

### Secondary Endpoints

| Endpoint | Measure | Source |
|----------|---------|--------|
| Falls | Count per participant over 6 months | Self-report + facility records |
| Loneliness | Change in UCLA Loneliness Scale | Baseline vs 6-month survey |
| Healthcare utilization | ER visits + hospitalizations | Claims / facility records |
| Caregiver burden | Change in Zarit Burden Scale | Caregiver survey (subset) |
| Usability | System Usability Scale (SUS) | End-of-study survey |
| Satisfaction | 5-point Likert ratings | End-of-study survey |

### Exploratory Endpoints

| Endpoint | Measure |
|----------|---------|
| Fall-risk signal performance | Sensitivity, specificity, AUC vs recorded falls |
| Loneliness signal performance | Correlation / AUC vs UCLA scores |
| Engagement | Daily active use, feature utilization |
| Subgroup effects | By age, cognitive status, living arrangement |

---

## 7. Safety & Risk Management

### Device Risk Classification

- **Risk level:** Moderate
- **Function:** Monitoring/reminder only
- **Constraint:** Ara Health Core will **NEVER** automatically diagnose, treat, or change medications

### Safety Handling Protocol

#### Critical Alerts

All alerts labeled: **"FOR REVIEW – NOT A DIAGNOSIS"**

| Alert Type | Routing | Action Required |
|------------|---------|-----------------|
| Elevated fall-risk pattern | Clinician dashboard | Human review within 24h |
| Persistent non-adherence | Caregiver notification | Follow-up conversation |
| Significant mood decline | Clinician + caregiver | Assessment by qualified professional |

#### Adverse Event Tracking

| Category | Definition | Reporting |
|----------|------------|-----------|
| Adverse Device Effect (ADE) | Any undesirable effect attributable to device | Log + report to sponsor |
| Serious Adverse Event (SAE) | Death, hospitalization, significant disability | Expedited reporting (24h) |
| Falls | All falls regardless of attribution | Log as secondary endpoint |

#### Kill Switch Capability

| Level | Trigger | Effect |
|-------|---------|--------|
| Individual | User/caregiver request or safety concern | Device deactivation, data preserved |
| System | Serious systemic issue identified | All devices enter Sanctuary mode |

---

## 8. Data, Privacy, and Roles

### PHI Classification

- All **health-related data** in trial context = **PHI**
- Companion content (stories, life history) treated as PHI during trial for safety

### Data Protection

| Measure | Implementation |
|---------|----------------|
| Encryption at rest | AES-256 |
| Encryption in transit | TLS 1.3 |
| Access control | Role-based, audit logged |
| Retention | Per participant consent + regulatory requirements |

### Organizational Roles

| Role | Entity | Responsibility |
|------|--------|----------------|
| Sponsor | ARA Inc | Protocol design, regulatory, analysis |
| Business Associate | ARA Inc | Data processing under BAA |
| Covered Entity | Provider organizations | Patient care, IRB |
| Data Processor | Cloud infrastructure | Compute/storage under DPA |

---

## 9. Statistical Considerations

### Sample Size

- n ≈ 500 total (≈ 250 per arm)
- Powered for: adherence endpoint
- α = 0.05, power = 80%
- Superiority margin pre-specified in SAP

### Primary Analysis

- Comparison of mean adherence rates between arms
- Method: Linear regression with covariate adjustment (age, baseline adherence, site)

### Secondary Analyses

| Endpoint | Method |
|----------|--------|
| Falls | Poisson/negative binomial regression |
| Scale scores | Linear mixed models |
| Time-to-event | Kaplan-Meier, Cox regression |

### Missing Data

- Primary: Multiple imputation
- Sensitivity: Complete case, conservative assumptions
- Handling defined a priori in SAP

### Multiplicity

- Primary endpoint: No adjustment (single primary)
- Secondary endpoints: Hierarchical testing or FDR control
- Exploratory: Nominal p-values, hypothesis-generating

---

## 10. Governance & Reporting

### Regulatory Framework

| Standard | Application |
|----------|-------------|
| ISO 14155 | Clinical investigation of medical devices |
| GCP (ICH E6) | General good clinical practice |
| 21 CFR Part 11 | Electronic records |
| HIPAA | Privacy and security |

### Ethics & Oversight

| Body | Role |
|------|------|
| IRB/Ethics Committee | Protocol approval at each site |
| DSMB | Independent safety monitoring (given vulnerable population) |
| Sponsor QA | Protocol compliance, data integrity |

### Reporting Deliverables

| Deliverable | Purpose |
|-------------|---------|
| 510(k) submission package | FDA clearance |
| Clinical Study Report (CSR) | Comprehensive results |
| CONSORT-AI manuscript | Peer-reviewed publication |
| BS 30440 / ISO 42001 inputs | AI quality management |

---

## 11. Covenant Alignment

This protocol implements the following covenant commitments:

| Covenant | Protocol Implementation |
|----------|------------------------|
| "Monitoring only, humans decide" | All outputs labeled "FOR REVIEW" |
| "Never diagnose or treat" | No autonomous clinical actions |
| "Kill switch always available" | Individual + system deactivation |
| "PHI protection" | BAA structure, encryption, access control |
| "Companion vs Health Core split" | Separate modules, only Health Core regulated |

---

## Appendices (to be developed)

- A: Schedule of Assessments
- B: Medication Adherence Calculation Methodology
- C: Fall-Risk Signal Specification
- D: Data Collection Forms
- E: Statistical Analysis Plan (SAP)
- F: Informed Consent Form Template

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-12-09 | ARA Inc | Initial draft |

---

## Contact

For questions about this protocol:

- **Sponsor:** ARA Inc
- **Protocol inquiries:** clinical@ara.health (placeholder)
