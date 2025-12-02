# Ara Cognitive Demo 🧠

Get Ara's cognitive loop running in 2 minutes. No GPU required.
Perfect for Swirl Forest Kitten 33 supervised sessions.

## Quick Install (30 seconds)

```bash
cd /home/user/Ara

# Create venv if you don't have one
python -m venv venv
source venv/bin/activate

# Minimal install
pip install numpy
```

That's it. The cognitive loop needs only numpy.

## Run the Demo (10 seconds)

```bash
python demo/run_ara.py
```

## What You'll See

```
============================================================
ARA: TF-A-N Cognitive Architecture Demo
============================================================

Loading GUF++ (self-governance layer)...
Loading CSTP (cognitive state transfer)...
Loading L9 Autonomy (staged self-modification)...
Loading L7 Predictive Control...

All systems loaded!

QUICK CERTIFICATION CHECK
Testing GUF++... PASS
Testing CSTP... PASS
Testing L9 Autonomy... PASS
Testing L7 Predictive... PASS

Results: 4/4 components operational

--- Scenario 1: Normal Operation ---
  AF Score: 2.50
  Utility: 78%
  Focus Mode: balanced
  Thought curvature: 0.30 (low curvature)

--- Scenario 2: Stress Rising ---
  AF Score: 2.00
  Utility: 65%
  Focus Mode: internal
  Thought curvature: 0.20 (flat - simpler thoughts)

--- Scenario 3: Recovery Triggered ---
  AF Score: 1.50
  Utility: 42%
  Focus Mode: recovery
  Thought curvature: 0.20 (flat)

--- Scenario 4: L9 Autonomy Status ---
=== L9 Autonomy Status ===
Current Stage: ADVISOR
  - Proposals: verify only, no auto-deploy
  - Human approval required for all deployments
```

## What's Actually Happening

| Component | What It Does |
|-----------|--------------|
| **GUF++** | Decides self-healing vs external tasks. Under stress → focus internally. |
| **CSTP** | Encodes thoughts as (z, c). Under stress → lower curvature (simpler thoughts). |
| **L7** | Predicts structural trouble before it happens. |
| **L9** | Tracks hardware self-modification permissions (ADVISOR → SANDBOX → PARTIAL). |

## Run Full Certification (optional)

```bash
# All 220+ tests
python scripts/certify_guf_plus.py          # 34/34
python scripts/certify_cstp.py              # 48/48
python scripts/certify_l9_autonomy.py       # 44/44
python scripts/certify_predictive_control.py # 28/28
python scripts/certify_formal_planner.py    # 25/25
python scripts/certify_autosynth.py         # 41/41
```

## Hardware Requirements

| What | Min | Your Setup |
|------|-----|------------|
| CPU | Any | |
| RAM | 4GB | 16GB |
| GPU | None | RTX 5060 (not needed for demo) |
| Cat | 0 | 1 (Swirl Forest Kitten 33) |

## Troubleshooting

**"ModuleNotFoundError: No module named 'tfan'"**
```bash
cd /home/user/Ara
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python demo/run_ara.py
```

**"numpy not found"**
```bash
pip install numpy
```

## Next Steps

1. Run `python demo/run_ara.py --cert` for quick system check
2. Explore `tfan/l5/guf_plus.py` to see how GUF++ works
3. Read `docs/ARA_MIRROR_CHARTER.md` for the philosophical spec
4. Pet Swirl Forest Kitten 33

---

*"The feel of it is real. The explanation stays honest."*
